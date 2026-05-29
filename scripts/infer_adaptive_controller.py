import argparse
import json
import os
import re
import sys
import time
from datetime import timedelta

import torch
import yaml
from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QWEN_DIR = os.path.join(REPO_ROOT, "qwen_vl")
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
for path in (QWEN_DIR, SCRIPTS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from train_adaptive_controller import build_qwen2vl_adaptive_model
from utils import Config, set_seed


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return Config(yaml.safe_load(f))


def format_m3cot_prompt(example):
    question = example["question"].strip()
    answer = str(example["answer"]).strip()
    choices = example["choices"]
    choices_str = "\n".join(
        f"{chr(65 + i)}.{{{choice.strip()}}}" for i, choice in enumerate(choices)
    )
    prompt = f"[Question]:{{{question}}}\n[Options]:\n{choices_str}\nAnswer:"
    return {
        "id": example["id"],
        "question_raw": prompt,
        "image_raw": example["image"],
        "gt_answer": answer.upper(),
        "choices": choices,
        "domain": example.get("domain"),
        "topic": example.get("topic"),
    }


def format_scienceqa_prompt(example, idx):
    question = example["question"].strip()
    choices = example.get("choices", [])
    if choices:
        choices_str = "\n".join(
            f"({chr(65 + i)}).{{{choice.strip()}}}" for i, choice in enumerate(choices)
        )
        prompt = f"[Question]:{{{question}}}\n[Options]:\n{choices_str}\nAnswer:"
    else:
        prompt = f"[Question]:{{{question}}}\nAnswer:"
    return {
        "id": str(idx),
        "question_raw": prompt,
        "image_raw": example["image"],
        "gt_answer": int(example["answer"]),
        "choices": choices,
    }


def build_eval_dataset(configs):
    task = getattr(configs, "eval_task", "m3cot").lower()
    data_percent = float(getattr(configs, "data_percent", 100.0))
    sample_seed = int(getattr(configs, "sample_seed", 42))
    if task == "m3cot":
        dataset = load_dataset(getattr(configs, "eval_dataset_name", "LightChen2333/M3CoT"))
        split = getattr(configs, "eval_split", "test")
        eval_dataset = dataset[split].filter(lambda ex: ex["image"] is not None).map(format_m3cot_prompt)
    elif task == "scienceqa":
        dataset = load_dataset(getattr(configs, "eval_dataset_name", "derek-thomas/ScienceQA"))
        split = getattr(configs, "eval_split", "test")
        eval_dataset = dataset[split].map(
            lambda ex, idx: {"original_idx": idx, **ex},
            with_indices=True,
        )
        eval_dataset = eval_dataset.filter(lambda ex: "image" in ex and ex["image"] is not None)
        eval_dataset = eval_dataset.map(lambda ex: format_scienceqa_prompt(ex, ex["original_idx"]))
    else:
        raise ValueError("eval_task must be 'm3cot' or 'scienceqa'")

    if not (0 < data_percent <= 100):
        raise ValueError("data_percent must be in (0, 100].")
    if data_percent < 100:
        sample_size = max(1, int(len(eval_dataset) * data_percent / 100.0))
        eval_dataset = eval_dataset.shuffle(seed=sample_seed).select(range(sample_size))
    return eval_dataset


def extract_m3cot_answer(text):
    matches = re.finditer(
        r"(?:the\s+answer\s+is|Answer:)\s*[\n\s]*([A-Z])",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    candidates = [m.group(1).upper() for m in matches]
    return candidates[-1] if candidates else None


def extract_scienceqa_answer(text):
    digit_patterns = [
        r"Therefore,?\s*the\s+answer\s+is\s+(\d)",
        r"the\s+answer\s+is\s+(\d)",
        r"answer\s+is:?\s*(\d)",
    ]
    for pattern in digit_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    letter_patterns = [
        r"Therefore,?\s*the\s+answer\s+is\s+([A-Z])",
        r"the\s+answer\s+is\s+([A-Z])",
        r"answer\s+is:?\s*([A-Z])",
    ]
    for pattern in letter_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return ord(match.group(1).upper()) - ord("A")
    return -1


def summarize_controller_trace(trace):
    counts = []
    selected_indices = []
    for step in trace:
        count = int(step["selected_counts"][0].item())
        counts.append(count)
        indices = step["selected_patch_indices"][0]
        selected_indices.append([int(x) for x in indices.tolist() if int(x) >= 0])
    return {
        "selected_counts_by_latent_step": counts,
        "selected_patch_indices_by_latent_step": selected_indices,
        "total_selected_patches": int(sum(counts)),
        "num_latent_steps": len(counts),
        "avg_selected_per_latent_step": float(sum(counts) / len(counts)) if counts else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Adaptive-controller IVT-LR inference")
    parser.add_argument("--config", required=True)
    parser.add_argument("--controller_checkpoint_path", default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--summary_path", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    args = parser.parse_args()

    configs = load_config(args.config)
    set_seed(int(getattr(configs, "seed", 0)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, processor = build_qwen2vl_adaptive_model(configs, device)

    controller_path = args.controller_checkpoint_path or getattr(
        configs, "controller_checkpoint_path", None
    )
    if not controller_path:
        raise ValueError("Set controller_checkpoint_path in config or pass --controller_checkpoint_path.")
    controller_state = torch.load(controller_path, map_location=device)
    model.controller.load_state_dict(controller_state, strict=True)
    model.eval()

    eval_dataset = build_eval_dataset(configs)
    task = getattr(configs, "eval_task", "m3cot").lower()
    output_path = args.output_path or getattr(
        configs,
        "prediction_output_path",
        os.path.join(getattr(configs, "output_dir", "."), f"adaptive_{task}_predictions.jsonl"),
    )
    summary_path = args.summary_path or getattr(
        configs,
        "summary_output_path",
        os.path.join(getattr(configs, "output_dir", "."), f"adaptive_{task}_summary.json"),
    )
    max_new_tokens = args.max_new_tokens or int(getattr(configs, "max_new_tokens", 512))
    latent_n = int(getattr(configs, "latent_n", 3))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    correct = 0
    total = 0
    total_generated_tokens = 0
    total_generate_time = 0.0
    total_selected_patches = 0
    total_latent_steps = 0
    selected_count_hist = {}

    with open(output_path, "w", encoding="utf-8") as f_out:
        for ex in tqdm(eval_dataset, total=len(eval_dataset), desc=f"Adaptive {task} inference"):
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": ex["image_raw"], "resized_height": 280, "resized_width": 280},
                    {"type": "text", "text": ex["question_raw"]},
                ],
            }]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            text = text + ("<|latent|>" * latent_n)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(device)
            prompt_len = inputs["input_ids"].size(1)

            start_time = time.time()
            with torch.no_grad():
                output_ids, controller_trace = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    pixel_values=inputs["pixel_values"],
                    image_grid_thw=inputs["image_grid_thw"],
                    max_new_tokens=max_new_tokens,
                    output_controller_trace=True,
                )
            generate_time = time.time() - start_time
            total_generate_time += generate_time
            generated_tokens = output_ids[0, prompt_len:]
            generated_text = processor.decode(generated_tokens, skip_special_tokens=True)
            total_generated_tokens += int(generated_tokens.numel())

            if task == "scienceqa":
                pred = extract_scienceqa_answer(generated_text)
                is_correct = pred == int(ex["gt_answer"])
            else:
                pred = extract_m3cot_answer(generated_text)
                is_correct = pred == str(ex["gt_answer"]).upper()
            correct += int(is_correct)
            total += 1

            trace_summary = summarize_controller_trace(controller_trace)
            total_selected_patches += trace_summary["total_selected_patches"]
            total_latent_steps += trace_summary["num_latent_steps"]
            for c in trace_summary["selected_counts_by_latent_step"]:
                selected_count_hist[c] = selected_count_hist.get(c, 0) + 1

            result = {
                "id": ex["id"],
                "answer": ex["gt_answer"],
                "prediction": pred,
                "correct": bool(is_correct),
                "generated_text": generated_text,
                "controller": trace_summary,
            }
            if "choices" in ex:
                result["choices"] = ex["choices"]
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
            f_out.flush()

    accuracy = correct / total if total else 0.0
    avg_tokens = total_generated_tokens / total if total else 0.0
    avg_time = total_generate_time / total if total else 0.0
    avg_selected_per_example = total_selected_patches / total if total else 0.0
    avg_selected_per_latent = total_selected_patches / total_latent_steps if total_latent_steps else 0.0
    summary = {
        "task": task,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "avg_generated_tokens": avg_tokens,
        "total_generate_time_seconds": total_generate_time,
        "avg_generate_time_seconds": avg_time,
        "avg_selected_patches_per_example": avg_selected_per_example,
        "avg_selected_patches_per_latent_step": avg_selected_per_latent,
        "selected_count_histogram": selected_count_hist,
        "prediction_output_path": output_path,
        "controller_checkpoint_path": controller_path,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Total generate time: {timedelta(seconds=int(total_generate_time))}")


if __name__ == "__main__":
    main()
