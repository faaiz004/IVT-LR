import argparse
import os
import sys
from collections import deque

import torch
import yaml
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QWEN_DIR = os.path.join(REPO_ROOT, "qwen_vl")
if QWEN_DIR not in sys.path:
    sys.path.insert(0, QWEN_DIR)

from dataset import MyCollator, get_cot_latent_dataset
from train_adaptive_controller import build_m3cot_dataset, build_qwen2vl_adaptive_model
from utils import Config, set_seed


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_controller_if_present(model, configs, device):
    controller_path = getattr(configs, "controller_checkpoint_path", None)
    if not controller_path:
        raise ValueError("Stage 2 requires controller_checkpoint_path.")
    controller_state = torch.load(controller_path, map_location=device)
    model.controller.load_state_dict(controller_state, strict=True)
    print(f"Loaded controller checkpoint from {controller_path}")


def load_stage2_lora_if_present(model, configs, device):
    lora_path = getattr(configs, "lora_stage2_checkpoint_path", None)
    if not lora_path:
        return
    lora_state = torch.load(lora_path, map_location=device)
    result = set_peft_model_state_dict(model.base_causallm, lora_state)
    missing = len(getattr(result, "missing_keys", []))
    unexpected = len(getattr(result, "unexpected_keys", []))
    print(f"Loaded Stage 2 LoRA checkpoint from {lora_path}. missing={missing} unexpected={unexpected}")


def freeze_controller_train_lora_only(model):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.controller.parameters():
        param.requires_grad = False

    trainable = []
    for name, param in model.base_causallm.named_parameters():
        if "lora_" in name or ".lora_" in name:
            param.requires_grad = True
            trainable.append(param)

    if not trainable:
        raise RuntimeError("No LoRA parameters were found. Set use_lora: true for Stage 2.")
    return trainable


def save_lora(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(get_peft_model_state_dict(model.base_causallm), path)


def count_selected_patches(controller_trace):
    total = 0
    steps = 0
    for step in controller_trace:
        total += int(step["selected_counts"].sum().item())
        steps += int(step["selected_counts"].numel())
    return total, steps


def main():
    parser = argparse.ArgumentParser(description="Stage 2 adaptive IVT-LR LoRA tuning")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    configs = Config(load_yaml(args.config))
    set_seed(int(getattr(configs, "seed", 0)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer, processor = build_qwen2vl_adaptive_model(configs, device)
    load_controller_if_present(model, configs, device)
    load_stage2_lora_if_present(model, configs, device)

    trainable_params = freeze_controller_train_lora_only(model)
    trainable_count = sum(p.numel() for p in trainable_params)
    print(f"Stage 2 trainable LoRA parameters: {trainable_count:,}")

    model.base_causallm.train()
    model.controller.eval()

    base_dataset = build_m3cot_dataset(configs, tokenizer, processor)
    collator = MyCollator(
        tokenizer,
        latent_id=tokenizer.convert_tokens_to_ids("<|latent|>"),
        label_pad_token_id=-100,
    )
    latent_dataset = get_cot_latent_dataset(
        scheduled_stage=int(getattr(configs, "scheduled_stage", configs.max_latent_stage)),
        base_dataset=base_dataset,
        configs=configs,
        start_id=tokenizer.convert_tokens_to_ids("<|start-latent|>"),
        latent_id=tokenizer.convert_tokens_to_ids("<|latent|>"),
        end_id=tokenizer.convert_tokens_to_ids("<|end-latent|>"),
        no_special_marker=True,
        shuffle=True,
    )
    dataloader = DataLoader(
        latent_dataset,
        batch_size=int(getattr(configs, "batch_size_training", 1)),
        shuffle=False,
        num_workers=int(getattr(configs, "num_workers", 1)),
        collate_fn=collator,
    )

    optimizer = AdamW(
        trainable_params,
        lr=float(getattr(configs, "stage2_lora_lr", getattr(configs, "lora_lr", 2e-5))),
        weight_decay=float(getattr(configs, "weight_decay", 0.0)),
    )
    grad_accum = int(getattr(configs, "gradient_accumulation_steps", 1))
    grad_clip_norm = float(getattr(configs, "grad_clip_norm", 1.0))
    log_every = int(getattr(configs, "log_every", 10))
    save_every = int(getattr(configs, "save_every", 250))
    max_steps = int(getattr(configs, "max_train_steps", 0))
    output_dir = getattr(configs, "output_dir", os.path.join(REPO_ROOT, "adaptive_lora_stage2_runs"))
    os.makedirs(output_dir, exist_ok=True)

    loss_window = deque(maxlen=max(log_every, 1))
    logprob_window = deque(maxlen=max(log_every, 1))
    patch_window = deque(maxlen=max(log_every, 1))
    optimizer.zero_grad(set_to_none=True)
    update_step = 0

    for global_step, batch in enumerate(tqdm(dataloader, desc="Stage2 adaptive LoRA"), start=1):
        batch = {k: v.to(device) for k, v in batch.items() if k != "idx"}
        out = model(
            **batch,
            mode="adaptive",
        )
        answer_logprob = out.answer_logprob.mean()
        loss = -answer_logprob / grad_accum
        loss.backward()

        selected_total, latent_steps = count_selected_patches(out.controller_trace)
        loss_window.append(float((-answer_logprob).detach().item()))
        logprob_window.append(float(answer_logprob.detach().item()))
        patch_window.append(float(selected_total / max(latent_steps, 1)))

        if global_step % grad_accum == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            update_step += 1
        else:
            grad_norm = torch.tensor(0.0)

        if global_step % log_every == 0:
            print(
                f"step={global_step} update={update_step} "
                f"answer_nll={sum(loss_window) / len(loss_window):.4f} "
                f"answer_logprob={sum(logprob_window) / len(logprob_window):.4f} "
                f"avg_selected_per_latent={sum(patch_window) / len(patch_window):.2f} "
                f"grad_norm={float(grad_norm):.4f}"
            )

        if global_step % save_every == 0:
            save_lora(model, os.path.join(output_dir, f"stage2_lora_step_{global_step}.pt"))

        if max_steps and global_step >= max_steps:
            break

    if global_step % grad_accum != 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        update_step += 1
        print(f"final_partial_update={update_step} grad_norm={float(grad_norm):.4f}")

    final_path = os.path.join(output_dir, "stage2_lora_final.pt")
    save_lora(model, final_path)
    print(f"Saved Stage 2 LoRA checkpoint to {final_path}")


if __name__ == "__main__":
    main()
