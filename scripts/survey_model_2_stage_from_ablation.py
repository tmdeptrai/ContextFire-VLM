
import argparse
import os
import time
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# --- Prompts ---
STAGE_1_SYSTEM_MESSAGE = "You are an expert Vision Language Model. Your task is to analyze the provided image and provide a detailed, objective summary of its contents. Describe the environment, key objects, people, and any signs of fire or smoke. Be factual and comprehensive."
STAGE_1_USER_QUERY = """Please summarize what you see in this image. Respond only in this json format:
{ "caption": "..." }"""

STAGE_2_SYSTEM_MESSAGE = """You are a classification model specialized in fire safety. Your task is to analyze a text caption describing a scene and classify the situation.
Based on the text, classify the fire situation as: 'no fire', 'controlled fire', or 'dangerous fire'.
- 'no fire': Scene has no fire, or only fire safety equipment.
- 'controlled fire': A fire is present but contained and not a threat.
- 'dangerous fire': A fire is uncontrolled and poses a threat.
Focus only on the text provided."""
STAGE_2_USER_QUERY = """Based on the following caption, classify the situation.
Caption: "{caption}"

Respond only in this json format:
{ "label": "no fire" | "controlled fire" | "dangerous fire" }"""


def parse_json_from_string(text, image_path, expected_keys):
    try:
        json_str_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if json_str_match:
            json_str = json_str_match.group(0)
            data = json.loads(json_str)
            return {key: str(data.get(key, "parsing_key_error")).strip() for key in expected_keys}
        else:
            print(f"❌ No JSON block found for {image_path}. Full raw output:---{text}---")
            return {key: "no_json_found" for key in expected_keys}
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error for {image_path}: {e}. Raw JSON: '{json_str}'")
        return {key: "malformed_json_output" for key in expected_keys}


def process_image_2_stage_transformers(model, processor, device, image_path):
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    import torch
    
    try:
        start_time = time.time()
        image = Image.open(image_path).convert("RGB")
        
        # --- Stage 1: Get Caption ---
        messages_s1 = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": STAGE_1_SYSTEM_MESSAGE + "" + STAGE_1_USER_QUERY}]}]
        text_s1 = processor.apply_chat_template(messages_s1, tokenize=False, add_generation_prompt=True)
        image_inputs_s1, video_inputs_s1 = process_vision_info(messages_s1)
        inputs_s1 = processor(text=[text_s1], images=image_inputs_s1, videos=video_inputs_s1, padding=True, return_tensors="pt").to(device)
        
        with torch.no_grad():
            generate_ids_s1 = model.generate(**inputs_s1, max_new_tokens=512)
        output_s1 = processor.tokenizer.batch_decode(generate_ids_s1, skip_special_tokens=True)[0].split("assistant", 1)[-1].strip()
        parsed_s1 = parse_json_from_string(output_s1, image_path, ["caption"])
        intermediate_caption = parsed_s1.get("caption", "No caption generated.")

        # --- Stage 2: Get Label from Caption ---
        messages_s2 = [{"role": "user", "content": [{"type": "text", "text": STAGE_2_SYSTEM_MESSAGE + "" + STAGE_2_USER_QUERY.format(caption=intermediate_caption)}]}]
        text_s2 = processor.apply_chat_template(messages_s2, tokenize=False, add_generation_prompt=True)
        _, video_inputs_s2 = process_vision_info(messages_s2) # We need this to keep the input structure consistent
        inputs_s2 = processor(text=[text_s2], videos=video_inputs_s2, padding=True, return_tensors="pt").to(device)
        
        with torch.no_grad():
            generate_ids_s2 = model.generate(**inputs_s2, max_new_tokens=64)
        output_s2 = processor.tokenizer.batch_decode(generate_ids_s2, skip_special_tokens=True)[0].split("assistant", 1)[-1].strip()
        parsed_s2 = parse_json_from_string(output_s2, image_path, ["label"])
        label = parsed_s2.get("label", "unknown")
        
        inference_time = time.time() - start_time
        return label, intermediate_caption, inference_time
        
    except Exception as e:
        print(f"Error during Transformers processing for {image_path}: {e}")
        return "error", "failed_inference", 0.0


def parse_args():
    parser = argparse.ArgumentParser(description="2-Stage VLM Evaluation (Image->Caption->Label) from Ablation Template")
    parser.add_argument("--model_name", type=str, required=True, help="Name for the model run")
    parser.add_argument("--input_csv", type=str, default="vlm_finetune/test_labels.csv", help="Input CSV file")
    parser.add_argument("--output_dir", type=str, default="results_2_stage_from_ablation", help="Output directory")
    parser.add_argument("--model_path", type=str, required=True, help="HuggingFace model path")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    df = pd.read_csv(args.input_csv)
    print(f"Total samples to process: {len(df)}")
    
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor, BitsAndBytesConfig
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    processor = Qwen2_5_VLProcessor.from_pretrained(args.model_path, use_fast=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True, quantization_config=bnb_config)

    predictions, intermediate_captions, inference_times, ground_truth = [], [], [], []
    
    for idx, row in df.iterrows():
        # Correctly resolve image paths for test_labels.csv
        full_img_path = row['image_path']
        if not os.path.exists(full_img_path):
             # Path in test_labels.csv is relative to the project root, so just join directly
            full_img_path = os.path.join(os.getcwd(), 'fine_tune_dataset', 'test', 'images', os.path.basename(row['image_path']))
        
        if os.path.exists(full_img_path):
            label, caption, inf_time = process_image_2_stage_transformers(model, processor, device, full_img_path)
            predictions.append(label)
            intermediate_captions.append(caption)
            inference_times.append(inf_time)
            ground_truth.append(row['label'])
            
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(df)} images...")
        else:
            print(f"Image not found: {full_img_path}")

    results_df = pd.DataFrame({
        'image_path': df['image_path'],
        'true_label': ground_truth,
        'predicted_label': predictions,
        'intermediate_caption': intermediate_captions,
        'inference_time': inference_times,
    })
    
    csv_out = os.path.join(args.output_dir, f"{args.model_name}_2_stage_ablation_results.csv")
    results_df.to_csv(csv_out, index=False)
    print(f"Results saved to {csv_out}")
    
    valid_results = results_df[~results_df['predicted_label'].isin(["error", "malformed_json_output", "no_json_found", "unknown", "parsing_key_error"])]
    if not valid_results.empty:
        accuracy = accuracy_score(valid_results['true_label'], valid_results['predicted_label'])
        precision, recall, f1, _ = precision_recall_fscore_support(valid_results['true_label'], valid_results['predicted_label'], average='weighted', zero_division=0)
        print("Model Performance Metrics (2-Stage Ablation):")
        print(f"Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1 Score: {f1:.4f}")
    else:
        print("No valid predictions were made to calculate metrics.")

if __name__ == "__main__":
    main()
