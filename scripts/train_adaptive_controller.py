import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QWEN_DIR = os.path.join(REPO_ROOT, "qwen_vl")
if QWEN_DIR not in sys.path:
    sys.path.insert(0, QWEN_DIR)

from controller import PatchPointerController
from dataset import MyCollator, get_dataset, get_cot_latent_dataset
from qwen_adaptive_ivtlr import QwenAdaptiveIVTLR
from qwen_vl_utils import process_vision_info
from utils import Config, set_seed


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_qwen2vl_adaptive_model(configs, device):
    tokenizer = AutoTokenizer.from_pretrained(
        configs.model_id,
        use_fast=False,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")
    processor = AutoProcessor.from_pretrained(configs.model_id, tokenizer=tokenizer)

    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        configs.model_id,
        device_map=None,
        torch_dtype=torch.bfloat16 if getattr(configs, "bf16", True) else torch.float32,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    base_model.resize_token_embeddings(len(tokenizer))
    if getattr(configs, "use_lora", True):
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            r=getattr(configs, "lora_r", 64),
            lora_alpha=getattr(configs, "lora_alpha", 16),
            lora_dropout=getattr(configs, "lora_dropout", 0.05),
            bias="none",
            inference_mode=False,
        )
        base_model = get_peft_model(base_model, lora_config)

    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    image_token_id = tokenizer.convert_tokens_to_ids(processor.image_token)
    visual_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    visual_end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")

    controller = PatchPointerController(
        model_dim=base_model.get_input_embeddings().embedding_dim,
        controller_dim=getattr(configs, "controller_hidden_dim", None),
        max_steps=configs.max_controller_steps,
        use_step_embedding=getattr(configs, "use_step_embedding", True),
    )
    model = QwenAdaptiveIVTLR(
        base_model,
        latent_token_id=latent_id,
        start_latent_id=start_id,
        end_latent_id=end_id,
        eos_token_id=tokenizer.eos_token_id,
        image_token_id=image_token_id,
        visual_start_id=visual_start_id,
        visual_end_id=visual_end_id,
        controller=controller,
        teacher_k=configs.teacher_k,
        max_controller_steps=configs.max_controller_steps,
        patch_reuse_policy=getattr(configs, "patch_reuse_policy", "never"),
        processor_model_id=configs.model_id,
    )

    teacher_path = getattr(configs, "ivtlr_checkpoint_path", None) or getattr(
        configs, "teacher_checkpoint_path", None
    )
    if teacher_path:
        state_dict = torch.load(teacher_path, map_location="cpu")
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        result = model.load_state_dict(state_dict, strict=False)
        print(
            f"Loaded teacher checkpoint. missing={len(result.missing_keys)} "
            f"unexpected={len(result.unexpected_keys)}"
        )

    if getattr(configs, "freeze_base_model", True) or getattr(configs, "train_controller_only", True):
        model.train_controller_only()
    model.to(device)
    return model, tokenizer, processor


def build_m3cot_dataset(configs, tokenizer, processor):
    dataset = load_dataset(getattr(configs, "dataset_name", "LightChen2333/M3CoT"))
    split = getattr(configs, "dataset_split", "train")
    train_dataset = dataset[split].filter(lambda ex: "image" in ex and ex["image"] is not None)

    def process_example(example):
        rationale = example["rationale"].replace("\n", " ").strip()
        example["steps"] = rationale.split(". ")
        if example["steps"] and example["steps"][-1] == "":
            example["steps"].pop()
        if len(example["steps"]) > 3:
            total_steps = len(example["steps"])
            step_size = total_steps // 3
            remainder = total_steps % 3
            new_steps = []
            start = 0
            for i in range(3):
                end = start + step_size + (1 if i < remainder else 0)
                new_steps.append(". ".join(example["steps"][start:end]))
                start = end
            example["steps"] = new_steps

        choices_str = "[Options]:\n" + "\n".join(
            f"({chr(65 + i)}).{{{choice.strip()}}}"
            for i, choice in enumerate(example["choices"])
        )
        question = f"[Question]:{{{example['question'].strip()}}}\n{choices_str}\nAnswer:\n"
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": example["image"], "resized_height": 280, "resized_width": 280},
                {"type": "text", "text": question},
            ],
        }]
        example["question"] = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[example["question"]],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.tolist() for k, v in inputs.items()}
        example["input_ids"] = torch.tensor(inputs["input_ids"][0])
        example["image_grid_thw"] = torch.tensor(inputs["image_grid_thw"]).squeeze(0)
        example["pixel_values"] = torch.tensor(inputs["pixel_values"])
        del example["rationale"]
        del example["choices"]
        return example

    num_proc = int(getattr(configs, "num_proc", 8))
    train_dataset = train_dataset.map(process_example, num_proc=num_proc)
    max_size = int(getattr(configs, "max_train_examples", 100000000))
    return get_dataset(train_dataset, tokenizer, processor, max_size=max_size, num_proc=num_proc)


def compute_budget_weights(rewards, mode, temperature, eps=1e-8):
    if mode == "softmax":
        return F.softmax(rewards / max(temperature, eps), dim=-1)
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, keepdim=True, unbiased=False)
    advantages = (rewards - mean) / (std + eps)
    if mode == "positive_advantage":
        positive = torch.relu(advantages)
        return positive / positive.sum(dim=-1, keepdim=True).clamp(min=eps)
    if mode == "signed_advantage":
        return advantages
    raise ValueError(f"Unknown advantage_mode={mode}")


def save_debug_trace(path, batch_idx, teacher_out, rewards, weights, budgets):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "batch_idx": batch_idx,
        "budgets": list(budgets),
        "rewards": rewards.detach().float().cpu().tolist(),
        "weights": weights.detach().float().cpu().tolist(),
        "teacher_topk": [
            {
                "latent_step_idx": step.latent_step_idx,
                "ranked_patch_indices": step.ranked_patch_indices.detach().cpu().tolist(),
                "attention_scores": step.attention_scores.detach().float().cpu().tolist(),
            }
            for step in teacher_out.latent_traces
        ],
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Teacher-guided adaptive IVT-LR controller training")
    parser.add_argument("--config", default=os.path.join(REPO_ROOT, "configs/adaptive_controller_qwen2b.yaml"))
    args = parser.parse_args()

    config_dict = load_yaml(args.config)
    configs = Config(config_dict)
    set_seed(getattr(configs, "seed", 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, processor = build_qwen2vl_adaptive_model(configs, device)

    base_dataset = build_m3cot_dataset(configs, tokenizer, processor)
    collator = MyCollator(
        tokenizer,
        latent_id=tokenizer.convert_tokens_to_ids("<|latent|>"),
        label_pad_token_id=-100,
    )
    dataloader = DataLoader(
        get_cot_latent_dataset(
            scheduled_stage=int(getattr(configs, "scheduled_stage", configs.max_latent_stage)),
            base_dataset=base_dataset,
            configs=configs,
            start_id=tokenizer.convert_tokens_to_ids("<|start-latent|>"),
            latent_id=tokenizer.convert_tokens_to_ids("<|latent|>"),
            end_id=tokenizer.convert_tokens_to_ids("<|end-latent|>"),
            no_special_marker=True,
            shuffle=True,
        ),
        batch_size=int(getattr(configs, "batch_size_training", 1)),
        shuffle=False,
        num_workers=int(getattr(configs, "num_workers", 1)),
        collate_fn=collator,
    )

    optimizer = AdamW(
        [p for p in model.controller.parameters() if p.requires_grad],
        lr=float(getattr(configs, "controller_lr", 1e-4)),
        weight_decay=float(getattr(configs, "weight_decay", 0.0)),
    )
    budgets = [int(x) for x in getattr(configs, "budget_candidates", [2, 4, 6, 8, 10])]
    lambda_patch = float(getattr(configs, "lambda_patch", 0.002))
    temperature = float(getattr(configs, "reward_temperature", 1.0))
    advantage_mode = getattr(configs, "advantage_mode", "softmax")
    grad_clip_norm = float(getattr(configs, "grad_clip_norm", 1.0))
    log_every = int(getattr(configs, "log_every", 10))
    save_every = int(getattr(configs, "save_every", 500))
    output_dir = getattr(configs, "output_dir", os.path.join(REPO_ROOT, "adaptive_controller_runs"))
    os.makedirs(output_dir, exist_ok=True)
    debug_trace_path = os.path.join(output_dir, "debug_traces.jsonl")
    win_counter = Counter()
    avg_advantage = defaultdict(float)

    model.train()
    for global_step, batch in enumerate(tqdm(dataloader), start=1):
        batch = {k: v.to(device) for k, v in batch.items() if k != "idx"}
        with torch.no_grad():
            teacher_out = model(
                **batch,
                mode="teacher",
                return_trace=True,
            )
            rewards_by_budget = []
            logprobs_by_budget = []
            patch_counts = []
            for budget in budgets:
                rollout = model(
                    **batch,
                    mode="forced_budget",
                    teacher_trace=teacher_out.latent_traces,
                    forced_budget=budget,
                )
                patches_per_example = budget * max(len(teacher_out.latent_traces), 1)
                reward = rollout.answer_logprob - lambda_patch * patches_per_example
                rewards_by_budget.append(reward)
                logprobs_by_budget.append(rollout.answer_logprob)
                patch_counts.append(patches_per_example)
            rewards = torch.stack(rewards_by_budget, dim=-1)
            logprobs = torch.stack(logprobs_by_budget, dim=-1)
            weights = compute_budget_weights(rewards, advantage_mode, temperature).detach()

        ctrl_stats = model.controller_teacher_forcing_loss(
            teacher_out.latent_traces,
            budgets=budgets,
            budget_weights=weights,
        )
        loss = ctrl_stats["loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.controller.parameters(), grad_clip_norm)
        optimizer.step()

        winners = rewards.argmax(dim=-1).detach().cpu().tolist()
        for winner in winners:
            win_counter[budgets[winner]] += 1
        advantages = rewards - rewards.mean(dim=-1, keepdim=True)
        for i, budget in enumerate(budgets):
            avg_advantage[budget] += float(advantages[:, i].mean().item())

        if global_step % log_every == 0:
            reward_means = rewards.mean(dim=0).detach().float().cpu().tolist()
            logprob_means = logprobs.mean(dim=0).detach().float().cpu().tolist()
            print(
                f"step={global_step} loss={float(loss.detach()):.4f} "
                f"grad_norm={float(grad_norm):.4f} "
                f"patch_acc={float(ctrl_stats['patch_top1_accuracy']):.3f} "
                f"stop_acc={float(ctrl_stats['stop_accuracy']):.3f}"
            )
            print(f"  reward_by_K={dict(zip(budgets, reward_means))}")
            print(f"  answer_logprob_by_K={dict(zip(budgets, logprob_means))}")
            print(f"  best_K_counts={dict(win_counter)} avg_patch_count={sum(patch_counts) / len(patch_counts):.1f}")

        if global_step <= int(getattr(configs, "num_debug_traces", 5)):
            save_debug_trace(debug_trace_path, global_step, teacher_out, rewards, weights, budgets)

        if global_step % save_every == 0:
            ckpt_path = os.path.join(output_dir, f"controller_step_{global_step}.pt")
            torch.save(model.controller.state_dict(), ckpt_path)

        max_steps = int(getattr(configs, "max_train_steps", 0))
        if max_steps and global_step >= max_steps:
            break

    torch.save(model.controller.state_dict(), os.path.join(output_dir, "controller_final.pt"))
    denom = max(global_step, 1)
    avg_advantage = {k: v / denom for k, v in avg_advantage.items()}
    print(f"final_best_K_counts={dict(win_counter)}")
    print(f"final_avg_advantage_by_K={avg_advantage}")


if __name__ == "__main__":
    main()
