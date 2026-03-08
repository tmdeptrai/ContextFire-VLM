
import argparse
import os
import time
import json
import re
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# --- Stage 1 Prompts: Image -> Caption ---
STAGE_1_SYSTEM_MESSAGE = """You are an expert Vision Language Model. Your task is to analyze the provided image and provide a detailed, objective summary of its contents.
Describe the environment, key objects, people, and any signs of fire or smoke. Be factual and comprehensive."""

STAGE_1_USER_QUERY = """Please summarize what you see in this image.
Respond only in this json format:
{ "caption": "..." }
"""

# --- Stage 2 Prompts: Caption -> Label ---
STAGE_2_SYSTEM_MESSAGE = """You are a classification model specialized in fire safety. Your task is to analyze a text caption describing a scene and classify the situation.
Based on the text, classify the fire situation as: 'no fire', 'controlled fire', or 'dangerous fire'.
- 'no fire': Scene has no fire, or only fire safety equipment (e.g., alarm, extinguisher).
- 'controlled fire': A fire is present but contained and not a threat (e.g., fireplace, campfire, candle).
- 'dangerous fire': A fire is uncontrolled and poses a threat (e.g., curtains on fire, building fire).
Focus only on the text provided."""

STAGE_2_USER_QUERY = """Based on the following caption, classify the situation.
Caption: "{caption}"

Respond only in this json format:
{ "label": "no fire" | "controlled fire" | "dangerous fire" }
"""

def encode_image(image_path):
    with open(image_path, "rb") as f:
        base64_bytes = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{base64_bytes}"

def process_image_2_stage_openai(client, model_name, image_path):
    try:
        start_time = time.time()
        data_url = encode_image(image_path)

        # --- Stage 1: Get Caption ---
        response_s1 = client.chat.completions.create(
            model=model_name,
            temperature=0.1,
            messages=[
                {"role": "system", "content": [{"type": "text", "text": STAGE_1_SYSTEM_MESSAGE}]},
                {"role": "user", "content": [{"type": "text", "text": STAGE_1_USER_QUERY}, {"type": "image_url", "image_url": {"url": data_url}}]}
            ]
        )
        output_s1 = response_s1.choices[0].message.content
        json_str_s1 = re.sub(r"^```json|```$", "", output_s1.strip(), flags=re.MULTILINE).strip()
        matches_s1 = re.findall(r'\{[^{}]+\}', json_str_s1, re.DOTALL)
        intermediate_caption = json.loads(matches_s1[-1]).get("caption", "No caption generated.") if matches_s1 else "No caption generated."

        # --- Stage 2: Get Label from Caption ---
        response_s2 = client.chat.completions.create(
            model=model_name,
            temperature=0.1,
            messages=[
                {"role": "system", "content": [{"type": "text", "text": STAGE_2_SYSTEM_MESSAGE}]},
                {"role": "user", "content": [{"type": "text", "text": STAGE_2_USER_QUERY.format(caption=intermediate_caption)}]}
            ]
        )
        output_s2 = response_s2.choices[0].message.content
        json_str_s2 = re.sub(r"^```json|```$", "", output_s2.strip(), flags=re.MULTILINE).strip()
        matches_s2 = re.findall(r'\{[^{}]+\}', json_str_s2, re.DOTALL)
        label = json.loads(matches_s2[-1]).get("label", "unknown").strip() if matches_s2 else "unknown"
        
        inference_time = time.time() - start_time
        return label, intermediate_caption, inference_time

    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return "unknown", "failed to generate caption", 0.0

def process_image_2_stage_transformers(model, processor, device, image_path):
    from PIL import Image
    import torch

    try:
        start_time = time.time()
        image = Image.open(image_path).convert("RGB")
        
        # --- Stage 1: Get Caption ---
        messages_s1 = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": STAGE_1_SYSTEM_MESSAGE + STAGE_1_USER_QUERY}]}]
        text_s1 = processor.apply_chat_template(messages_s1, tokenize=False, add_generation_prompt=True)
        inputs_s1 = processor(text=[text_s1], images=[image], padding=True, return_tensors="pt").to(device)
        
        with torch.no_grad():
            generate_ids_s1 = model.generate(**inputs_s1, max_new_tokens=512)
        output_s1 = processor.tokenizer.batch_decode(generate_ids_s1, skip_special_tokens=True)[0].split("assistant")[-1].strip()
        matches_s1 = re.findall(r'\{[^{}]+\}', output_s1, re.DOTALL)
        intermediate_caption = json.loads(matches_s1[-1]).get("caption", "No caption generated.") if matches_s1 else "No caption generated."

        # --- Stage 2: Get Label from Caption ---
        messages_s2 = [{"role": "user", "content": [{"type": "text", "text": STAGE_2_SYSTEM_MESSAGE + STAGE_2_USER_QUERY.format(caption=intermediate_caption)}]}]
        text_s2 = processor.apply_chat_template(messages_s2, tokenize=False, add_generation_prompt=True)
        inputs_s2 = processor(text=[text_s2], padding=True, return_tensors="pt").to(device) # No image input
        
        with torch.no_grad():
            generate_ids_s2 = model.generate(**inputs_s2, max_new_tokens=64)
        output_s2 = processor.tokenizer.batch_decode(generate_ids_s2, skip_special_tokens=True)[0].split("assistant")[-1].strip()
        matches_s2 = re.findall(r'\{[^{}]+\}', output_s2, re.DOTALL)
        label = json.loads(matches_s2[-1]).get("label", "unknown").strip() if matches_s2 else "unknown"
        
        inference_time = time.time() - start_time
        return label, intermediate_caption, inference_time

    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return "unknown", "failed to generate caption", 0.0

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a VLM in a 2-stage pipeline (Image->Caption->Label)")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model being evaluated")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV")
    parser.add_argument("--output_dir", type=str, default="results_2_stage", help="Directory to save results")
    parser.add_argument("--backend", type=str, choices=["openai", "transformers"], default="transformers", help="Backend to use")
    parser.add_argument("--api_url", type=str, default="http://127.0.0.1:8080/v1", help="API URL for openai backend")
    parser.add_argument("--model_path", type=str, help="HuggingFace model path for transformers backend")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    df = pd.read_csv(args.input_csv)
    print(f"Total samples to process: {len(df)}")
    
    # Backend Initialization
    client = None
    model, processor, device = None, None, None
    if args.backend == "openai":
        from openai import OpenAI
        client = OpenAI(base_url=args.api_url, api_key="sk-test", timeout=9999)
    elif args.backend == "transformers":
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor, BitsAndBytesConfig
        device = "cuda" if torch.cuda.is_available() else "cpu"
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        processor = Qwen2_5_VLProcessor.from_pretrained(args.model_path, use_fast=True)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True, quantization_config=bnb_config)
    
    predictions, intermediate_captions, inference_times, ground_truth = [], [], [], []
    
    for idx, row in df.iterrows():
        img_path = row['image_path']
        if not os.path.exists(img_path):
            img_path = os.path.join('fine_tune_dataset', 'test', 'images', os.path.basename(img_path))
        
        if os.path.exists(img_path):
            label, caption, inf_time = "", "", 0.0
            if args.backend == "openai":
                label, caption, inf_time = process_image_2_stage_openai(client, args.model_name, img_path)
            elif args.backend == "transformers":
                label, caption, inf_time = process_image_2_stage_transformers(model, processor, device, img_path)
            
            predictions.append(label)
            intermediate_captions.append(caption)
            inference_times.append(inf_time)
            ground_truth.append(row['label'])
            
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(df)} images...")
        else:
            print(f"Image not found: {img_path}")

    # --- Save and Analyze Results ---
    results_df = pd.DataFrame({
        'image_path': df['image_path'],
        'true_label': ground_truth,
        'predicted_label': predictions,
        'intermediate_caption': intermediate_captions,
        'inference_time': inference_times,
    })
    
    csv_out = os.path.join(args.output_dir, f"{args.model_name}_2_stage_results.csv")
    results_df.to_csv(csv_out, index=False)
    print(f"Results saved to {csv_out}")
    
    # Calculate and print metrics
    accuracy = accuracy_score(ground_truth, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(ground_truth, predictions, average='weighted', zero_division=0)
    print("Model Performance Metrics (2-Stage):")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

if __name__ == "__main__":
    main()
