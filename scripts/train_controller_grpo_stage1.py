import argparse
import json
import os
import re
import sys
from collections import Counter

import torch
import yaml
from datasets import load_dataset
from torch.optim import AdamW
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QWEN_DIR = os.path.join(REPO_ROOT, "qwen_vl")
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
for path in (QWEN_DIR, SCRIPTS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from qwen_vl_utils import process_vision_info
from train_adaptive_controller import build_qwen2vl_adaptive_model
from utils import Config, set_seed


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return Config(yaml.safe_load(f))


def format_m3cot_prompt(example):
    choices_str = "\n".join(
        f"{chr(65 + i)}.{{{choice.strip()}}}" for i, choice in enumerate(example["choices"])
    )
    return {
        "id": str(example.get("id", "")),
        "question_raw": (
            f"[Question]:{{{example['question'].strip()}}}\n"
            f"[Options]:\n{choices_str}\nAnswer:"
        ),
        "image_raw": example["image"],
        "gt_answer": str(example["answer"]).strip().upper(),
    }


def extract_m3cot_answer(text):
    matches = re.finditer(
        r"(?:the\s+answer\s+is|Answer:)\s*[\n\s]*([A-Z])",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    candidates = [m.group(1).upper() for m in matches]
    if candidates:
        return candidates[-1]
    fallback = re.search(r"\b([A-Z])\b", text)
    return fallback.group(1).upper() if fallback else None


def build_train_dataset(configs):
    dataset = load_dataset(getattr(configs, "dataset_name", "LightChen2333/M3CoT"))
    split = getattr(configs, "dataset_split", "train")
    train_dataset = dataset[split].filter(lambda ex: ex["image"] is not None)
    train_dataset = train_dataset.map(format_m3cot_prompt)
    train_dataset = train_dataset.shuffle(seed=int(getattr(configs, "seed", 0)))
    max_examples = int(getattr(configs, "max_train_examples", 100000000))
    if max_examples > 0:
        train_dataset = train_dataset.select(range(min(max_examples, len(train_dataset))))
    return train_dataset


def encode_example(processor, example, latent_n, device):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": example["image_raw"], "resized_height": 280, "resized_width": 280},
            {"type": "text", "text": example["question_raw"]},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    text = text + ("<|latent|>" * latent_n)
    image_inputs, video_inputs = process_vision_info(messages)
    return processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)


def summarize_trace(controller_trace):
    logprob_sum = None
    entropy_sum = None
    selected_count = 0
    action_count = 0.0
    for step in controller_trace:
        selected_count += int(step["selected_counts"][0].item())
        if "logprob_sum" in step:
            logprob_sum = step["logprob_sum"] if logprob_sum is None else logprob_sum + step["logprob_sum"]
            entropy_sum = step["entropy_sum"] if entropy_sum is None else entropy_sum + step["entropy_sum"]
            action_count += float(step["action_count"].detach().sum().item())
    if logprob_sum is None:
        raise RuntimeError("Sampled controller trace did not include logprob_sum.")
    return logprob_sum.squeeze(0), entropy_sum.squeeze(0), selected_count, action_count


def main():
    parser = argparse.ArgumentParser(description="Stage 1 controller-only GRPO for adaptive IVT-LR")
    parser.add_argument("--config", required=True)
    parser.add_argument("--controller_checkpoint_path", default=None)
    args = parser.parse_args()

    configs = load_config(args.config)
    set_seed(int(getattr(configs, "seed", 0)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, processor = build_qwen2vl_adaptive_model(configs, device)

    controller_path = args.controller_checkpoint_path or getattr(configs, "controller_checkpoint_path", None)
    if controller_path:
        controller_state = torch.load(controller_path, map_location=device)
        model.controller.load_state_dict(controller_state, strict=True)
        print(f"Loaded controller checkpoint from {controller_path}")

    model.train_controller_only()
    model.base_causallm.eval()
    model.controller.train()

    output_dir = getattr(configs, "output_dir", os.path.join(REPO_ROOT, "adaptive_controller_grpo_runs"))
    os.makedirs(output_dir, exist_ok=True)
    rollouts_per_prompt = int(getattr(configs, "grpo_rollouts_per_prompt", 5))
    controller_temperature = float(getattr(configs, "grpo_controller_temperature", 1.0))
    min_patches = int(getattr(configs, "grpo_min_patches", 0))
    max_new_tokens = int(getattr(configs, "max_new_tokens", 64))
    latent_n = int(getattr(configs, "latent_n", 3))
    lambda_patch = float(getattr(configs, "lambda_patch", 0.0002))
    correct_reward = float(getattr(configs, "grpo_correct_reward", 1.0))
    incorrect_reward = float(getattr(configs, "grpo_incorrect_reward", 0.0))
    entropy_coef = float(getattr(configs, "grpo_entropy_coef", 0.0))
    grad_clip_norm = float(getattr(configs, "grad_clip_norm", 1.0))
    log_every = int(getattr(configs, "log_every", 10))
    save_every = int(getattr(configs, "save_every", 250))
    max_steps = int(getattr(configs, "max_train_steps", 0))

    train_dataset = build_train_dataset(configs)
    optimizer = AdamW(
        [p for p in model.controller.parameters() if p.requires_grad],
        lr=float(getattr(configs, "controller_lr", 5e-5)),
        weight_decay=float(getattr(configs, "weight_decay", 0.0)),
    )

    correct_counter = Counter()
    reward_window = []
    patch_window = []
    loss_window = []
    trace_path = os.path.join(output_dir, "grpo_stage1_traces.jsonl")

    for global_step, example in enumerate(tqdm(train_dataset, desc="Stage1 controller GRPO"), start=1):
        inputs = encode_example(processor, example, latent_n, device)
        prompt_len = inputs["input_ids"].size(1)
        rollout_logprobs = []
        rollout_entropies = []
        rewards = []
        rollout_payloads = []

        for rollout_idx in range(rollouts_per_prompt):
            output_ids, controller_trace = model.generate_with_sampled_controller(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs["pixel_values"],
                image_grid_thw=inputs["image_grid_thw"],
                max_new_tokens=max_new_tokens,
                controller_temperature=controller_temperature,
                min_patches=min_patches,
            )
            generated_tokens = output_ids[0, prompt_len:]
            generated_text = processor.decode(generated_tokens, skip_special_tokens=True)
            pred = extract_m3cot_answer(generated_text)
            is_correct = pred == example["gt_answer"]
            logprob_sum, entropy_sum, selected_count, action_count = summarize_trace(controller_trace)
            reward = (correct_reward if is_correct else incorrect_reward) - lambda_patch * selected_count

            rollout_logprobs.append(logprob_sum)
            rollout_entropies.append(entropy_sum / max(action_count, 1.0))
            rewards.append(reward)
            correct_counter[int(is_correct)] += 1
            reward_window.append(reward)
            patch_window.append(selected_count)
            rollout_payloads.append(
                {
                    "rollout": rollout_idx,
                    "prediction": pred,
                    "correct": bool(is_correct),
                    "selected_count": selected_count,
                    "reward": reward,
                    "generated_text": generated_text,
                }
            )

        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
        advantages = rewards_t - rewards_t.mean()
        std = rewards_t.std(unbiased=False)
        if float(std.item()) > 1e-6:
            advantages = advantages / (std + 1e-6)
        logprobs_t = torch.stack(rollout_logprobs)
        entropies_t = torch.stack(rollout_entropies)
        loss = -(advantages.detach() * logprobs_t).mean() - entropy_coef * entropies_t.mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.controller.parameters(), grad_clip_norm)
        optimizer.step()
        loss_window.append(float(loss.detach().item()))

        if global_step <= int(getattr(configs, "num_debug_traces", 5)):
            with open(trace_path, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "step": global_step,
                            "id": example["id"],
                            "answer": example["gt_answer"],
                            "rewards": rewards,
                            "advantages": advantages.detach().float().cpu().tolist(),
                            "rollouts": rollout_payloads,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        if global_step % log_every == 0:
            total_rollouts = correct_counter[0] + correct_counter[1]
            acc = correct_counter[1] / max(total_rollouts, 1)
            avg_reward = sum(reward_window[-log_every * rollouts_per_prompt :]) / max(
                len(reward_window[-log_every * rollouts_per_prompt :]), 1
            )
            avg_patches = sum(patch_window[-log_every * rollouts_per_prompt :]) / max(
                len(patch_window[-log_every * rollouts_per_prompt :]), 1
            )
            avg_loss = sum(loss_window[-log_every:]) / max(len(loss_window[-log_every:]), 1)
            print(
                f"step={global_step} loss={avg_loss:.4f} reward={avg_reward:.4f} "
                f"rollout_acc={acc:.3f} avg_patches={avg_patches:.2f} "
                f"grad_norm={float(grad_norm):.4f}"
            )

        if global_step % save_every == 0:
            torch.save(
                model.controller.state_dict(),
                os.path.join(output_dir, f"controller_grpo_step_{global_step}.pt"),
            )

        if max_steps and global_step >= max_steps:
            break

    final_path = os.path.join(output_dir, "controller_grpo_final.pt")
    torch.save(model.controller.state_dict(), final_path)
    print(f"Saved final controller to {final_path}")


if __name__ == "__main__":
    main()
