
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

# --- Stage 2 Prompts: Caption -> Label (copied from previous 2-stage script) ---
STAGE_2_SYSTEM_MESSAGE = """You are a classification model specialized in fire safety. Your task is to analyze a text caption describing a scene and classify the situation.
Based on the text, classify the fire situation as: 'no fire', 'controlled fire', or 'dangerous fire'.
- 'no fire': Scene has no fire, or only fire safety equipment.
- 'controlled fire': A fire is present but contained and not a threat.
- 'dangerous fire': A fire is uncontrolled and poses a threat.
Focus only on the text provided."""
STAGE_2_USER_QUERY = f"""Based on the following caption, classify the situation.
Caption: "{caption}"

Respond only in this json format:
{ \"label\": \"no fire\" | \"controlled fire\" | \"dangerous fire\" }
""" 
def parse_json_from_string(text, input_identifier, expected_keys):
    try:
        json_str_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        #json_str_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_str_match:
            json_str = json_str_match.group(0)
            data = json.loads(json_str)
            return {key: str(data.get(key, "parsing_key_error")).strip() for key in expected_keys}
        else:
            print(f"❌ No JSON block found for {input_identifier}. Full raw output:---{text}---")
            return {key: "no_json_found" for key in expected_keys}
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error for {input_identifier}: {e}. Raw JSON: '{json_str}'")
        return {key: "malformed_json_output" for key in expected_keys}

def process_caption_to_label(model, processor, device, caption, image_path_for_debug):
    import torch
    try:
        start_time = time.time()
        
        messages_s2 = [{"role": "system", "content": STAGE_2_SYSTEM_MESSAGE}, {"role": "user", "content": STAGE_2_USER_QUERY.format(caption=caption)}]
        text_s2 = processor.apply_chat_template(messages_s2, tokenize=False, add_generation_prompt=True)
        
        # IMPORTANT: No image input for this stage
        inputs_s2 = processor(text=text_s2, return_tensors="pt").to(device)
        
        with torch.no_grad():
            generate_ids_s2 = model.generate(**inputs_s2, max_new_tokens=64)
        output_s2 = processor.batch_decode(generate_ids_s2, skip_special_tokens=True)[0].split("assistant")[-1].strip()
        parsed_s2 = parse_json_from_string(output_s2, image_path_for_debug, ["label"])
        label = parsed_s2.get("label", "unknown")
        
        inference_time = time.time() - start_time
        return label, inference_time
    except Exception as e:
        print(f"Error during caption-to-label processing for {image_path_for_debug}: {e}")
        return "error", 0.0

def parse_args():
    parser = argparse.ArgumentParser(description="Caption to Label Ablation Study")
    parser.add_argument("--model_name", type=str, required=True, help="Name for the model run")
    parser.add_argument("--input_csv", type=str, default="model_survey_notebooks/qwen2.5_vl_3B_results.csv", help="Input CSV file containing captions and true labels")
    parser.add_argument("--output_dir", type=str, default="results_caption_to_label", help="Output directory")
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

    predictions, inference_times, ground_truth = [], [], []
    
    for idx, row in df.iterrows():
        caption = row['caption']
        image_path_for_debug = row['image_path'] # Keep original image path for debug context

        label, inf_time = process_caption_to_label(model, processor, device, caption, image_path_for_debug)
            
        predictions.append(label)
        inference_times.append(inf_time)
        ground_truth.append(row['true_label'])
            
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(df)} captions... Average inference time so far: {np.mean(inference_times):.3f}s")

    results_df = pd.DataFrame({
        'image_path': df['image_path'],
        'original_caption': df['caption'],
        'true_label': ground_truth,
        'predicted_label': predictions,
        'inference_time': inference_times,
    })
    
    csv_out = os.path.join(args.output_dir, f"{args.model_name}_caption_to_label_results.csv")
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
