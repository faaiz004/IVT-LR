from transformers import AutoTokenizer, AutoProcessor
from qwen_ivtlr import IVTLR  
from transformers import Qwen2VLForConditionalGeneration
import torch
import deepspeed
from peft import LoraConfig,get_peft_model
from qwen_vl_utils import process_vision_info
from datasets import load_dataset
from utils import set_seed
import re
import logging
import json
import os
import time
from datetime import timedelta
import argparse
from tqdm import tqdm
logging.basicConfig(
    filename='qwenvl_2b_infer_time.log',
    level=logging.DEBUG,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
import pdb

device = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_PATCH_REUSE_POLICY = "always"

def load_inference_model(checkpoint_path, patch_reuse_policy="never", patch_sampling_strategy="attention_topk"):
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        use_fast=False,
        trust_remote_code=True,
        padding_side="right"
    )
    
    tokenizer.add_special_tokens({
        "additional_special_tokens": [
            "<|start-latent|>",
            "<|end-latent|>",
            "<|latent|>"
        ]
    })
    
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager"
    )
    base_model.resize_token_embeddings(len(tokenizer))
    processor.tokenizer = tokenizer

    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        r=64,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        inference_mode=False
    )
    base_model = get_peft_model(base_model, lora_config)
    
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    image_token_id = tokenizer.convert_tokens_to_ids(processor.image_token)
    visual_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    visual_end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    
    model = IVTLR(
        base_model,
        latent_token_id=latent_id,
        start_latent_id=start_id,
        end_latent_id=end_id,
        eos_token_id=tokenizer.eos_token_id,
        image_token_id=image_token_id,
        visual_start_id=visual_start_id, 
        visual_end_id=visual_end_id,
        patch_reuse_policy=patch_reuse_policy,
        patch_sampling_strategy=patch_sampling_strategy,
        processor_model_id="Qwen/Qwen2-VL-2B-Instruct",
    )
    
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    print(state_dict.keys())
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=True)
    print(model)
    print("Successfully load")
    
    model = model.to(device)
    model.eval()
    return model, processor, tokenizer

def format_prompt(example):
    question = example["question"].strip()
    rationale = example["rationale"].replace("\n", " ").strip()
    answer = example["answer"].strip()
    choices = example["choices"]
    image = example["image"]

    choices_str = "\n".join([f"{chr(65+i)}.{{{choice.strip()}}}" for i, choice in enumerate(choices)])
    user_prompt = (
        f"[Question]:{{{question}}}\n"
        f"[Options]:\n{choices_str}\n"
        f"Answer:"
    )
    return user_prompt, rationale, answer, image

def process_func(example):
    prompt, rationale, answer, image = format_prompt(example)

    return {
        "question_raw": prompt,
        "image_raw": image,
        "gt_answer": answer,
        "id": example["id"],
        "choices": example["choices"],
        "domain": example["domain"],
        "topic": example["topic"]
    }

def build_eval_dataset(data_percent=100.0, sample_seed=42):
    dataset = load_dataset("LightChen2333/M3CoT")
    val_dataset = dataset["test"]
    val_dataset = val_dataset.filter(lambda e: e["image"] is not None).map(process_func)
    if data_percent >= 100:
        return val_dataset
    if data_percent <= 0:
        raise ValueError("data_percent must be in (0, 100].")
    sample_size = max(1, int(len(val_dataset) * (data_percent / 100.0)))
    val_dataset = val_dataset.shuffle(seed=sample_seed).select(range(sample_size))
    return val_dataset


def compute_latent_attention_trace(model, inputs, attn_threshold=None, attn_threshold_multiplier=5.0):
    seq_len = inputs["input_ids"].shape[1]
    position_ids = torch.arange(seq_len, device=inputs["input_ids"].device).unsqueeze(0)
    labels = inputs["input_ids"].clone()

    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=labels,
            position_ids=position_ids,
            pixel_values=inputs["pixel_values"],
            image_grid_thw=inputs["image_grid_thw"],
            return_latent_attn=True,
            latent_attn_threshold=attn_threshold,
            latent_attn_threshold_multiplier=attn_threshold_multiplier,
        )

    return outputs.latent_attn_trace


def compute_token_norms(model, inputs):
    seq_len = inputs["input_ids"].shape[1]
    position_ids = torch.arange(seq_len, device=inputs["input_ids"].device).unsqueeze(0)
    labels = inputs["input_ids"].clone()

    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=labels,
            position_ids=position_ids,
            pixel_values=inputs["pixel_values"],
            image_grid_thw=inputs["image_grid_thw"],
            return_token_norms=True,
        )

    if not outputs.token_norms:
        return None
    return outputs.token_norms[0]


def evaluate_and_save(
    eval_dataset,
    model,
    processor,
    output_path,
    latent_n=3,
    max_new_tokens=512,
    attn_trace_path=None,
    attn_threshold=None,
    attn_threshold_multiplier=5.0,
    analyze_token_norms=False,
    token_norms_path=None,
):
    model.eval()
    correct = 0
    total = 0
    total_generated_tokens = 0 
    total_generate_time = 0.0  
    attn_traces = []

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    norms_file = None
    if analyze_token_norms and token_norms_path:
        token_norms_dir = os.path.dirname(token_norms_path)
        if token_norms_dir:
            os.makedirs(token_norms_dir, exist_ok=True)
        norms_file = open(token_norms_path, "a", encoding="utf-8")

    try:
        with open(output_path, "a", encoding="utf-8") as f_out:
            for ex in tqdm(
                eval_dataset,
                total=len(eval_dataset),
                desc="Evaluating M3CoT",
                dynamic_ncols=True,
            ):
                input_text = ex["question_raw"]
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": ex["image_raw"], "resized_height": 280, "resized_width": 280},
                        {"type": "text", "text": input_text}
                    ]
                }]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                text = text + ("<|latent|>" * latent_n)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt"
                ).to(device)
                if attn_trace_path:
                    trace = compute_latent_attention_trace(
                        model,
                        inputs,
                        attn_threshold=attn_threshold,
                        attn_threshold_multiplier=attn_threshold_multiplier,
                    )
                    attn_traces.append({
                        "id": ex["id"],
                        "latent_attn": trace[0] if trace else [],
                    })
                if analyze_token_norms and norms_file is not None:
                    norms = compute_token_norms(model, inputs)
                    if norms is not None:
                        norms_result = {
                            "id": ex["id"],
                            "patch_norms": norms.get("patch_norms", []),
                            "reasoning_norms": norms.get("reasoning_norms", []),
                            "aggregate_stats": norms.get("aggregate_stats", {}),
                        }
                        norms_file.write(json.dumps(norms_result, ensure_ascii=False) + "\n")
                        norms_file.flush()

                input_ids = inputs["input_ids"]
                prompt_length = input_ids.shape[1]
                
                generate_start_time = time.time()
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=torch.tensor(inputs["input_ids"]), 
                        attention_mask=torch.tensor(inputs["attention_mask"]),
                        pixel_values=torch.tensor(inputs["pixel_values"]),
                        image_grid_thw=torch.tensor(inputs["image_grid_thw"]),
                        max_new_tokens=max_new_tokens
                    )
                generate_end_time = time.time()
                sample_generate_time = generate_end_time - generate_start_time
                total_generate_time += sample_generate_time
                            
                generated_tokens = outputs[0, prompt_length:]
                new_generated_text = processor.decode(generated_tokens, skip_special_tokens=True)
                output_text = processor.decode(outputs[0], skip_special_tokens=True)
                logging.debug(f"[OUTPUT] {output_text}")
                
                num_generated_tokens = len(generated_tokens)
                total_generated_tokens += num_generated_tokens

                cleaned_text = re.sub(
                    r'(?<=answer:)\s*(\n+\s*)?assistant\b',
                    '',
                    output_text,
                    flags=re.IGNORECASE
                )
                matches = re.finditer(
                    r'(?:the\s+answer\s+is|Answer:)\s*[\n\s]*([A-Z])',
                    cleaned_text,
                    flags=re.IGNORECASE | re.DOTALL
                )
                candidates = {match.group(1).upper() for match in matches}
                gt_answer = ex["gt_answer"].strip().upper()

                if gt_answer in candidates:
                    correct += 1
                    logging.debug(f"correct: True")
                total += 1
                logging.debug(f"[TOTAL] {total}")

                # pdb.set_trace()
                message_question = ex["question_raw"]
                message_question = message_question.replace("<image>", "", 1).replace("Answer:", "", 1).strip()
                message_question = message_question.split("Answer:")[0].strip()

                result = {
                    "id": ex["id"],
                    "choices": ex["choices"],
                    "answer": ex["gt_answer"],
                    "domain": ex["domain"],
                    "topic": ex["topic"],
                    "messages": [
                        message_question,
                        new_generated_text
                    ]
                }
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                f_out.flush()
                
            avg_generated_tokens = total_generated_tokens / total if total > 0 else 0
            avg_time_per_sample = total_generate_time / total if total > 0 else 0
        
            logging.info(f"[FINAL] Avg generated tokens per sample: {avg_generated_tokens:.1f}")
            logging.info(f"[FINAL] Total generate time: {total_generate_time:.2f}s ({timedelta(seconds=int(total_generate_time))})")
            logging.info(f"[FINAL] Avg generate time per sample: {avg_time_per_sample:.3f}s")
    finally:
        if norms_file is not None:
            norms_file.close()

    if attn_trace_path:
        attn_dir = os.path.dirname(attn_trace_path)
        if attn_dir:
            os.makedirs(attn_dir, exist_ok=True)
        with open(attn_trace_path, "w", encoding="utf-8") as f:
            json.dump(attn_traces, f, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen2-VL IVTLR inference on M3CoT")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to model state_dict checkpoint (.pth)")
    parser.add_argument("--latent_n", type=int, default=3, help="Number of <|latent|> tokens appended to the prompt")
    parser.add_argument("--patch_reuse_policy", type=str, default=DEFAULT_PATCH_REUSE_POLICY,
                        choices=["never", "next_step_only", "always"],
                        help="Patch selection reuse policy during generation")
    parser.add_argument("--patch_sampling_strategy", type=str, default="attention_topk",
                        choices=["attention_topk", "random_image_only", "all_image_patches"],
                        help="Patch sampling strategy for selecting visual tokens")
    parser.add_argument("--output_path", type=str, default="output/qwen2vl_2b.jsonl", help="Path to write JSONL predictions")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum generated tokens per sample")
    parser.add_argument("--attn_trace_path", type=str, default=None, help="Path to write JSON attention traces")
    parser.add_argument("--attn_threshold", type=float, default=None, help="Absolute attention threshold for patch logging")
    parser.add_argument("--attn_threshold_multiplier", type=float, default=5.0, help="Threshold multiplier for 1/seq_len baseline")
    parser.add_argument("--data_percent", type=float, default=100.0, help="Percentage of dataset to use for inference")
    parser.add_argument("--sample_seed", type=int, default=42, help="Random seed for dataset sampling")
    parser.add_argument(
        "--analyze_patch_reasoning_norms",
        action="store_true",
        help="Compute patch vs reasoning token norms and store per-example JSON",
    )
    parser.add_argument(
        "--token_norms_path",
        type=str,
        default="output/qwen2vl_2b_token_norms.jsonl",
        help="Path to write token norm JSONL",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model, processor, _ = load_inference_model(
        args.checkpoint_path,
        patch_reuse_policy=args.patch_reuse_policy,
        patch_sampling_strategy=args.patch_sampling_strategy,
    )
    if not (0 < args.data_percent <= 100):
        raise ValueError("--data_percent must be in (0, 100].")
    set_seed(args.sample_seed)
    val_dataset = build_eval_dataset(data_percent=args.data_percent, sample_seed=args.sample_seed)
    evaluate_and_save(
        val_dataset,
        model,
        processor,
        output_path=args.output_path,
        latent_n=args.latent_n,
        max_new_tokens=args.max_new_tokens,
        attn_trace_path=args.attn_trace_path,
        attn_threshold=args.attn_threshold,
        attn_threshold_multiplier=args.attn_threshold_multiplier,
        analyze_token_norms=args.analyze_patch_reasoning_norms,
        token_norms_path=args.token_norms_path,
    )


if __name__ == "__main__":
    main()
