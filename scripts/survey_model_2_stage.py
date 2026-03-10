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
{ \"caption\": \"...\" }
"""

# --- Stage 2 Prompts: Caption -> Label ---
STAGE_2_SYSTEM_MESSAGE = """You are a classification model specialized in fire safety. Your task is to analyze a text caption describing a scene and classify the situation.
Based on the text, classify the fire situation as: 'no fire', 'controlled fire', or 'dangerous fire'.
- 'no fire': Scene has no fire, or only fire safety equipment (e.g., alarm, extinguisher).
- 'controlled fire': A fire is present but contained and not a threat (e.g., fireplace, campfire, candle).
- 'dangerous fire': A fire is uncontrolled and poses a threat (e.g., curtains on fire, building fire).
Focus only on the text provided."""

STAGE_2_USER_QUERY = """Based on the following caption, classify the situation.
Caption: \"{caption}\"

Respond only with the label: 'no fire', 'controlled fire', or 'dangerous fire'.
"""

def encode_image(image_path):
    with open(image_path, "rb") as f:
        base64_bytes = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{base64_bytes}"

def parse_json_from_string(text, image_path, expected_keys):
    try:
        json_str_match = re.search(r'\{[^{}]+\}', text, re.DOTALL)
        if json_str_match:
            json_str = json_str_match.group(0)
            data = json.loads(json_str)
            result = {key: str(data.get(key, "parsing_key_error")).strip() for key in expected_keys}
            return result
        else:
            print(f"❌ No JSON block found in output for {image_path}. Full output (truncated): '{text[:100]}'")
            return {key: "no_json_found" for key in expected_keys}
    except json.JSONDecodeError as e:
        # Ensure json_str is defined in case of a syntax error during re.search
        json_str_for_error = json_str_match.group(0) if json_str_match else "N/A"
        print(f"❌ JSON parsing error for {image_path}: {e}. Raw JSON string: '{json_str_for_error}'")
        return {key: "malformed_json_output" for key in expected_keys}

def process_image_2_stage_openai(client, model_name, image_path):
    try:
        start_time = time.time()
        data_url = encode_image(image_path)

        # Stage 1
        response_s1 = client.chat.completions.create(model=model_name, temperature=0.1, messages=[{"role": "system", "content": STAGE_1_SYSTEM_MESSAGE}, {"role": "user", "content": [{"type": "text", "text": STAGE_1_USER_QUERY}, {"type": "image_url", "image_url": {"url": data_url}}]}])
        output_s1 = response_s1.choices[0].message.content
        parsed_s1 = parse_json_from_string(output_s1, image_path, ["caption"])
        intermediate_caption = parsed_s1.get("caption", "No caption generated.")
        
        # Stage 2
        response_s2 = client.chat.completions.create(model=model_name, temperature=0.1, messages=[{"role": "system", "content": STAGE_2_SYSTEM_MESSAGE}, {"role": "user", "content": STAGE_2_USER_QUERY.format(caption=intermediate_caption)}])
        output_s2 = response_s2.choices[0].message.content
        parsed_s2 = parse_json_from_string(output_s2, image_path, ["label"])
        label = parsed_s2.get("label", "unknown")

        inference_time = time.time() - start_time
        return label, intermediate_caption, inference_time
    except Exception as e:
        print(f"Error during OpenAI processing for {image_path}: {e}")
        return "error", "failed_inference", 0.0

def process_image_2_stage_transformers(model, processor, device, image_path):
    from PIL import Image
    import torch
    try:
        start_time = time.time()
        image = Image.open(image_path).convert("RGB")
        
        # Stage 1: Image -> Caption
        messages_s1 = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "<image>" + STAGE_1_SYSTEM_MESSAGE + STAGE_1_USER_QUERY}
            ]
        }]
        text_s1 = processor.apply_chat_template(messages_s1, tokenize=False, add_generation_prompt=True)
        inputs_s1 = processor(text=text_s1, images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            generate_ids_s1 = model.generate(**inputs_s1, max_new_tokens=512, do_sample=True, top_k=50, top_p=0.95)
        
        output_s1 = processor.batch_decode(generate_ids_s1, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        output_s1 = re.sub(r'^(system|user|assistant)', '', output_s1, flags=re.MULTILINE).strip()
        
        # Correctly parse the JSON to get only the caption text
        parsed_s1 = parse_json_from_string(output_s1, image_path, ["caption"])
        intermediate_caption = parsed_s1.get("caption", "No caption generated.")

        # Stage 2: Caption -> Label
        messages_s2 = [{
            "role": "user",
            "content": [
                {"type": "text", "text": STAGE_2_SYSTEM_MESSAGE + STAGE_2_USER_QUERY.format(caption=intermediate_caption)}
            ]
        }]
        text_s2 = processor.apply_chat_template(messages_s2, tokenize=False, add_generation_prompt=True)
        inputs_s2 = processor(text=text_s2, return_tensors="pt").to(device)
        
        with torch.no_grad():
            generate_ids_s2 = model.generate(**inputs_s2, max_new_tokens=64, do_sample=True, top_k=50, top_p=0.95)
        
        # For debugging, treat the entire cleaned output as the label
        label = processor.batch_decode(generate_ids_s2, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        label = re.sub(r'^(system|user|assistant)', '', label, flags=re.MULTILINE).strip()
        
        inference_time = time.time() - start_time
        return label, intermediate_caption, inference_time
    except Exception as e:
        print(f"Error during Transformers processing for {image_path}: {e}")
        return "error", "failed_inference", 0.0

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a VLM in a 2-stage pipeline (Image->Caption->Label)")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model being evaluated")
    parser.add_argument("--input_csv", type=str, default="labels_v2.csv", help="Path to input CSV")
    parser.add_argument("--output_dir", type=str, default="results_2_stage", help="Directory to save results")
    parser.add_argument("--backend", type=str, choices=["openai", "transformers"], default="transformers", help="Backend to use")
    parser.add_argument("--api_url", type=str, default="http://127.0.0.1:8080/v1", help="API URL for openai backend")
    parser.add_argument("--model_path", type=str, required=True, help="HuggingFace model path for transformers backend")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    df = pd.read_csv(args.input_csv)
    print(f"Total samples to process: {len(df)}")
    
    # Backend Initialization
    client, model, processor, device = None, None, None, None
    if args.backend == "openai":
        from openai import OpenAI
        client = OpenAI(base_url=args.api_url, api_key="sk-test", timeout=9999)
    elif args.backend == "transformers":
        import torch
        # Updated imports for InternVL
        from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
        device = "cuda" if torch.cuda.is_available() else "cpu"
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        # Use AutoProcessor and AutoModelForImageTextToText
        processor = AutoProcessor.from_pretrained(args.model_path)
        model = AutoModelForImageTextToText.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True, quantization_config=bnb_config)

    predictions, intermediate_captions, inference_times, ground_truth = [], [], [], []
    
    for idx, row in df.iterrows():
        # Correctly resolve image paths from CSV
        # Assuming image_path in CSV is relative to the current working directory or needs specific prefix
        img_path_candidates = [
            os.path.join(os.getcwd(), row['image_path']),
            os.path.join(os.getcwd(), 'FIRENET', row['image_path']),
            os.path.join(os.getcwd(), 'fine_tune_dataset', 'test', 'images', os.path.basename(row['image_path']))
        ]
        full_img_path = None
        for p in img_path_candidates:
            if os.path.exists(p):
                full_img_path = p
                break

        if full_img_path and os.path.exists(full_img_path):
            label, caption, inf_time = "", "", 0.0
            if args.backend == "openai":
                label, caption, inf_time = process_image_2_stage_openai(client, args.model_name, full_img_path)
            elif args.backend == "transformers":
                label, caption, inf_time = process_image_2_stage_transformers(model, processor, device, full_img_path)
            
            predictions.append(label)
            intermediate_captions.append(caption)
            inference_times.append(inf_time)
            ground_truth.append(row['label'])
            
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(df)} images...")
        else:
            print(f"Image not found or resolved: {row['image_path']} (tried: {img_path_candidates})")

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
    valid_results = results_df[~results_df['predicted_label'].isin(["error", "malformed_json_output", "no_json_found", "unknown"])]
    if not valid_results.empty:
        accuracy = accuracy_score(valid_results['true_label'], valid_results['predicted_label'])
        precision, recall, f1, _ = precision_recall_fscore_support(valid_results['true_label'], valid_results['predicted_label'], average='weighted', zero_division=0)
        print("\nModel Performance Metrics (2-Stage):")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
    else:
        print("No valid predictions were made to calculate metrics.")

if __name__ == "__main__":
    main()
