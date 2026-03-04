from transformers import AutoTokenizer, AutoProcessor
from qwen_ivtlr import IVTLR  
from transformers import Qwen2VLForConditionalGeneration
import torch
import deepspeed
from peft import LoraConfig,get_peft_model
from qwen_vl_utils import process_vision_info
from datasets import load_dataset
import re
import logging
import json
import os
import time
from datetime import timedelta
logging.basicConfig(
    filename='qwenvl_32_infer_time.log',
    level=logging.DEBUG,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
import pdb

device = "cuda" if torch.cuda.is_available() else "cpu"
PATCH_REUSE_POLICY = "never"

def load_inference_model(checkpoint_path, patch_reuse_policy="never"):
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
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
        "Qwen/Qwen2-VL-7B-Instruct",
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

model, processor, tokenizer = load_inference_model("your_path", patch_reuse_policy=PATCH_REUSE_POLICY)

os.makedirs("output", exist_ok=True)

def format_prompt(example):
    question = example["question"].strip()
    answer = example["answer"] 
    choices = example.get("choices", [])
    image = example["image"]

    if choices:
        choices_str = "\n".join([f"({chr(65+i)}).{{{choice.strip()}}}" for i, choice in enumerate(choices)])
        user_prompt = (
            f"[Question]:{{{question}}}\n"
            f"[Options]:\n{choices_str}\n"
            f"Answer:"
        )
    else:
        user_prompt = f"[Question]:{{{question}}}\nAnswer:"
    
    return user_prompt, answer, image

def process_func(example, idx):
    prompt, answer, image = format_prompt(example)
    
    return {
        "idx": idx,  
        "question_raw": prompt,
        "image_raw": image,
        "gt_answer": answer,  
    }

dataset = load_dataset("derek-thomas/ScienceQA")
test_dataset = dataset["test"]

def has_image(example):
    return "image" in example and example["image"] is not None


test_dataset = test_dataset.map(lambda example, idx: {"original_idx": idx, **example}, with_indices=True)
test_dataset = test_dataset.filter(has_image)
test_dataset = test_dataset.map(lambda example: process_func(example, example["original_idx"]))

def evaluate_and_save(eval_dataset, model, processor):
    model.eval()
    correct = 0
    total = 0
    results = {} 
    total_generated_tokens = 0  
    total_generate_time = 0.0  
    
    output_json_path = "sqa_output/qwen_2_scienceqa.json"
    
    for ex in eval_dataset:
        idx = str(ex["idx"]) 
        input_text = ex["question_raw"]
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": ex["image_raw"], "resized_height": 280, "resized_width": 280},
                {"type": "text", "text": input_text}
            ]
        }]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text = text + "<|latent|>" + "<|latent|>" + "<|latent|>"
        
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(device)
        
        prompt_length = inputs["input_ids"].shape[1]
        
        generate_start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"], 
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs["pixel_values"],
                image_grid_thw=inputs["image_grid_thw"],
                max_new_tokens=512
            )
        generate_end_time = time.time()
        sample_generate_time = generate_end_time - generate_start_time
        total_generate_time += sample_generate_time
        generated_tokens = outputs[0, prompt_length:]
        generated_text = processor.decode(generated_tokens, skip_special_tokens=True)
        num_generated_tokens = len(generated_tokens)
        total_generated_tokens += num_generated_tokens
        
        pred_answer = extract_answer(generated_text)
        

        results[idx] = pred_answer
        

        gt_answer = ex["gt_answer"]  
        if pred_answer == gt_answer:
            correct += 1
        
        total += 1
        
    
    output_data = {"results": results}
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    accuracy = correct / total if total > 0 else 0
    avg_generated_tokens = total_generated_tokens / total if total > 0 else 0
    avg_time_per_sample = total_generate_time / total if total > 0 else 0
    
    
    logging.info(f"[FINAL] Total: {total}, Correct: {correct}, Accuracy: {accuracy:.2%}")
    logging.info(f"[FINAL] Avg generated tokens per sample: {avg_generated_tokens:.1f}")
    logging.info(f"[FINAL] Total generate time: {total_generate_time:.2f}s ({timedelta(seconds=int(total_generate_time))})")
    logging.info(f"[FINAL] Avg generate time per sample: {avg_time_per_sample:.3f}s")
    
    
    print(f"[FINAL] Total: {total}, Correct: {correct}, Accuracy: {accuracy:.2%}")
    print(f"Results saved to: {output_json_path}")
    
    return accuracy

def extract_answer(text):
    digit_patterns = [
        r'Therefore,?\s*the\s+answer\s+is\s+(\d)',  
        r'the\s+answer\s+is\s+(\d)', 
        r'answer\s+is:?\s*(\d)',  
    ]
    
    for pattern in digit_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            answer_idx = int(match.group(1))
            logging.debug(f"Extracted answer (digit): {answer_idx}")
            return answer_idx
    

    letter_patterns = [
        r'Therefore,?\s*the\s+answer\s+is\s+([A-Z])', 
        r'the\s+answer\s+is\s+([A-Z])',
        r'answer\s+is:?\s*([A-Z])',  
    ]
    
    for pattern in letter_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            letter = match.group(1).upper()

            answer_idx = ord(letter) - ord('A')
            logging.debug(f"Extracted answer (letter): {letter} -> index {answer_idx}")
            return answer_idx
    

    logging.warning(f"No answer pattern found in text: {text[:200]}")
    return -1  

evaluate_and_save(test_dataset, model, processor)
