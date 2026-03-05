
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

# Unified prompt for ablation study (label only)
ABLATION_SYSTEM_MESSAGE = """You are a Vision Language Model specialized in detecting clues of fire, smoke and surrounding context then classify them as no fire, dangerous fire or controlled fire.
Your task is to analyze the provided image and respond with a classification label.
Based on the visual information, classify the fire situation: 'no fire', 'controlled fire', or 'dangerous fire'.
Focus on delivering a precise label based on the visual information."""

ABLATION_USER_QUERY = """Analyze the image for signs of fire and smoke and classify the situation.
The possible classifications are:
- 'no fire' (e.g., fire alarm, fire distinguisher)
- 'controlled fire' (e.g., fireplace, campfire, cooking, candles)
- 'dangerous fire' (e.g., curtains on fire, smoke on ceiling, spreading fire)

Respond only with a JSON format containing the classification label:
{ "label": "no fire" | "controlled fire" | "dangerous fire" }
"""

def encode_image(image_path):
    with open(image_path, "rb") as f:
        base64_bytes = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{base64_bytes}"

def process_image_openai(client, model_name, image_path):
    try:
        start_time = time.time()
        data_url = encode_image(image_path)

        response = client.chat.completions.create(
            model=model_name,
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": [{"type": "text", "text": ABLATION_SYSTEM_MESSAGE}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": ABLATION_USER_QUERY},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ]
                }
            ]
        )

        full_output = response.choices[0].message.content
        
        # Extract JSON from output
        json_str = re.sub(r"^```json|```$", "", full_output.strip(), flags=re.MULTILINE).strip()
        
        matches = re.findall(r'\{[^{}]+\}', json_str, re.DOTALL)
        if matches:
            result = json.loads(matches[-1])
            label = result.get("label", "unknown").strip()
        else:
            print(f"No JSON block found in output for {image_path}")
            label = "unknown"

        inference_time = time.time() - start_time
        return label, inference_time

    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return "unknown", 0.0

def process_image_transformers(model, processor, device, image_path):
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    import torch
    
    try:
        start_time = time.time()
        image = Image.open(image_path).convert("RGB")
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": ABLATION_SYSTEM_MESSAGE + "" + ABLATION_USER_QUERY}
            ]
        }]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)
        
        with torch.no_grad():
            generate_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                num_return_sequences=1,
            )
            
        output = processor.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
        if output.startswith("system"):
            output = output.split("assistant", 1)[-1].strip()

        clean_output = re.sub(r'^(system|user|assistant)', '', output, flags=re.MULTILINE).strip()

        matches = re.findall(r'\{[^{}]+\}', clean_output, re.DOTALL)
        if matches:
            result = json.loads(matches[-1])
            label = result.get("label", "unknown").strip()
        else:
            label = "unknown"
            
        inference_time = time.time() - start_time
        return label, inference_time
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return "unknown", 0.0

def process_image_lmdeploy(pipe, image_path):
    from lmdeploy.vl import load_image
    try:
        start_time = time.time()
        image = load_image(image_path)
        prompt = "<image>" + ABLATION_SYSTEM_MESSAGE + "" + ABLATION_USER_QUERY
        
        response = pipe((prompt, image))
        response_text = response.text
        
        matches = re.findall(r'\{[^{}]+\}', response_text, re.DOTALL)
        if matches:
            result = json.loads(matches[-1])
            label = result.get("label", "unknown").strip()
        else:
            label = "unknown"

        inference_time = time.time() - start_time
        return label, inference_time
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return "unknown", 0.0


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a VLM on the fire detection dataset (Ablation Study: Label Only)")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model being evaluated (used for output files)")
    parser.add_argument("--input_csv", type=str, default="data/processed/test_labels.csv", help="Path to input CSV")
    parser.add_argument("--output_dir", type=str, default="results_ablation", help="Directory to save ablation study results")
    parser.add_argument("--backend", type=str, choices=["openai", "transformers", "lmdeploy"], default="transformers", help="Backend to use for inference")
    parser.add_argument("--api_url", type=str, default="http://127.0.0.1:8080/v1", help="API URL for openai backend")
    parser.add_argument("--model_path", type=str, help="HuggingFace model path for transformers/lmdeploy backends")
    return parser.parse_args()


def main():
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    df = pd.read_csv(args.input_csv)
    print(f"Total samples in dataset: {len(df)}")
    
    # Initialize backend
    if args.backend == "openai":
        from openai import OpenAI
        client = OpenAI(base_url=args.api_url, api_key="sk-test", timeout=9999)
    elif args.backend == "transformers":
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor, BitsAndBytesConfig
        device = "cuda" if torch.cuda.is_available() else "cpu"
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        )
        processor = Qwen2_5_VLProcessor.from_pretrained(args.model_path, use_fast=True)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
            quantization_config=bnb_config
        )
    elif args.backend == "lmdeploy":
        from lmdeploy import pipeline, TurbomindEngineConfig, ChatTemplateConfig
        pipe = pipeline(args.model_path, backend_config=TurbomindEngineConfig(session_len=16384, tp=1), chat_template_config=ChatTemplateConfig(model_name='internvl2_5'))
    
    predictions = []
    inference_times = []
    ground_truth = []
    
    for idx, row in df.iterrows():
        img_path = row['image_path']
        
        # Fallbacks for image paths if running from root
        if not os.path.exists(img_path):
            alt_path = os.path.join('data/images', os.path.basename(img_path))
            if os.path.exists(alt_path):
                img_path = alt_path
            elif os.path.exists(os.path.join('../', img_path)):
                img_path = os.path.join('../', img_path)
            else:
                # One more check for fine_tune_dataset just in case
                alt_path2 = os.path.join('fine_tune_dataset', 'test', 'images', os.path.basename(img_path))
                if os.path.exists(alt_path2):
                    img_path = alt_path2
                
        if os.path.exists(img_path):
            if args.backend == "openai":
                pred, inf_time = process_image_openai(client, args.model_name, img_path)
            elif args.backend == "transformers":
                pred, inf_time = process_image_transformers(model, processor, device, img_path)
            elif args.backend == "lmdeploy":
                pred, inf_time = process_image_lmdeploy(pipe, img_path)
                
            predictions.append(pred)
            inference_times.append(inf_time)
            ground_truth.append(row['label'])
            
            if idx % 10 == 0 and idx > 0:
                print(f"Processed {idx} images... Average inference time so far: {np.mean(inference_times):.3f}s")
        else:
            print(f"Image not found: {img_path}")
            
    print(f"Processing complete! Average inference time: {np.mean(inference_times):.3f}s")
    
    # Save results
    results_df = pd.DataFrame({
        'image_path': df['image_path'][:len(predictions)],
        'true_label': ground_truth,
        'predicted_label': predictions,
        'inference_time': inference_times,
        'correct': [t == p for t, p in zip(ground_truth, predictions)]
    })
    
    csv_out = os.path.join(args.output_dir, f"{args.model_name}_ablation_results.csv")
    results_df.to_csv(csv_out, index=False)
    print(f"Results saved to {csv_out}")
    
    # Calculate metrics
    accuracy = accuracy_score(ground_truth, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        ground_truth, predictions, average='weighted', zero_division=0
    )
    
    print("Model Performance Metrics (Ablation Study):")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    print("Inference Time Statistics:")
    if len(inference_times) > 0:
        print(f"Average: {np.mean(inference_times):.3f}s")
        print(f"Std Dev: {np.std(inference_times):.3f}s")
        print(f"Min: {np.min(inference_times):.3f}s")
        print(f"Max: {np.max(inference_times):.3f}s")
    
    # Generate plots
    plt.figure(figsize=(10, 6))
    plt.hist(inference_times, bins=30)
    plt.title('Distribution of Inference Times (Ablation Study)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(args.output_dir, f"{args.model_name}_ablation_inference_times.png"))
    plt.close()
    
    labels = sorted(list(set(ground_truth)))
    cm = confusion_matrix(ground_truth, predictions, labels=labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=labels, yticklabels=labels
    )
    plt.title(f'Confusion Matrix for {args.model_name} (Ablation Study)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"{args.model_name}_ablation_confusion_matrix.png"))
    plt.close()
    print(f"Plots saved to {args.output_dir}")

if __name__ == "__main__":
    main()
