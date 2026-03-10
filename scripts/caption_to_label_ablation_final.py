
import argparse
import os
import time
import json
import re
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Import the necessary Qwen utility that we know works
from qwen_vl_utils import process_vision_info

# --- Prompts ---
SYSTEM_MESSAGE = """You are a classification model specialized in fire safety. Your task is to analyze a text caption describing a scene and classify the situation.
Based on the text, classify the fire situation as: 'no fire', 'controlled fire', or 'dangerous fire'.
- 'no fire': Scene has no fire, or only fire safety equipment.
- 'controlled fire': A fire is present but contained and not a threat.
- 'dangerous fire': A fire is uncontrolled and poses a threat.
Focus only on the text provided."""

USER_QUERY = 'Based on the following caption, classify the situation.\nCaption: "{caption}"\nRespond only in this json format:{ "label": "no fire" | "controlled fire" | "dangerous fire" }'


def parse_json_from_string(text, identifier):
    try:
        # Use the proven regex from the working notebook
        matches = re.findall(r'\{[^{}]+\}', text, re.DOTALL)
        if matches:
            json_str = matches[-1] # Take the last match
            data = json.loads(json_str)
            return str(data.get("label", "parsing_key_error")).strip()
        else:
            print(f"❌ No JSON block found for {identifier}. Raw output:---{text}---")
            return "no_json_found"
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error for {identifier}: {e}. Raw JSON: '{json_str}'")
        return "malformed_json_output"


def process_caption_to_label(model, processor, device, caption, identifier):
    import torch
    try:
        start_time = time.time()
        
        # Build the message structure exactly like the working ablation script
        messages = [{"role": "user", "content": [{"type": "text", "text": SYSTEM_MESSAGE + USER_QUERY.format(caption=caption)}]}]
        
        # 1. Apply chat template
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # 2. Use the proven process_vision_info utility
        image_inputs, video_inputs = process_vision_info(messages)
        
        # 3. Call the processor with the full, correct structure
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)
        
        with torch.no_grad():
            generate_ids = model.generate(**inputs, max_new_tokens=64)
            
        output = processor.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0].split("assistant", 1)[-1].strip()
        label = parse_json_from_string(output, identifier)
        
        inference_time = time.time() - start_time
        return label, inference_time
        
    except Exception as e:
        print(f"Error during caption-to-label processing for {identifier}: {e}")
        return "error", 0.0


def parse_args():
    parser = argparse.ArgumentParser(description="Caption-to-Label Ablation Study (Final Version)")
    parser.add_argument("--model_name", type=str, required=True, help="Name for the model run")
    parser.add_argument("--input_csv", type=str, default="model_survey_notebooks/qwen2.5_vl_3B_results.csv", help="Input CSV with captions")
    parser.add_argument("--output_dir", type=str, default="results_caption_to_label_final", help="Output directory")
    parser.add_argument("--model_path", type=str, required=True, help="HuggingFace model path")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    df = pd.read_csv(args.input_csv).dropna(subset=['caption', 'true_label'])
    print(f"Total samples to process: {len(df)}")
    
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor, BitsAndBytesConfig
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    processor = Qwen2_5_VLProcessor.from_pretrained(args.model_path, use_fast=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True, quantization_config=bnb_config)

    predictions, inference_times = [], []
    
    for idx, row in df.iterrows():
        caption = row['caption']
        identifier = row['image_path']

        label, inf_time = process_caption_to_label(model, processor, device, caption, identifier)
        predictions.append(label)
        inference_times.append(inf_time)
            
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(df)} captions...")

    results_df = pd.DataFrame({
        'image_path': df['image_path'],
        'original_caption': df['caption'],
        'true_label': df['true_label'],
        'predicted_label': predictions,
        'inference_time': inference_times,
    })
    
    csv_out = os.path.join(args.output_dir, f"{args.model_name}_results.csv")
    results_df.to_csv(csv_out, index=False)
    print(f"Results saved to {csv_out}")
    
    valid_results = results_df[~results_df['predicted_label'].isin(["error", "malformed_json_output", "no_json_found", "unknown", "parsing_key_error"])]
    if not valid_results.empty:
        accuracy = accuracy_score(valid_results['true_label'], valid_results['predicted_label'])
        precision, recall, f1, _ = precision_recall_fscore_support(valid_results['true_label'], valid_results['predicted_label'], average='weighted', zero_division=0)
        print("Model Performance Metrics (Caption to Label Ablation):")
        print(f"Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1 Score: {f1:.4f}")
    else:
        print("No valid predictions were made to calculate metrics.")

if __name__ == "__main__":
    main()
