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
import multiprocessing
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig # Updated imports
from PIL import Image

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

def worker_process(gpu_id, data_chunk, model_path):
    """
    The main function for each worker process.
    It initializes the model on a specific GPU and processes a chunk of data.
    """
    
    # CRITICAL: Set the CUDA_VISIBLE_DEVICES environment variable for this specific process
    # This ensures the process only sees and uses the assigned GPU.
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = "cuda" # This will now map to the specific GPU (e.g., cuda:0)
    
    print(f"[Worker {gpu_id}] Initializing model on GPU {gpu_id}...")
    
    # --- Model Initialization (within the worker) ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )
    # Use AutoProcessor and AutoModelForImageTextToText for InternVL3
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device, # Use the isolated device
        trust_remote_code=True,
        quantization_config=bnb_config
    )
    
    local_predictions = []
    local_inference_times = []
    local_ground_truth = []
    
    print(f"[Worker {gpu_id}] Starting inference on {len(data_chunk)} images.")
    
    for idx, row in data_chunk.iterrows():
        try:
            start_time = time.time()
            image = Image.open(row['image_path']).convert("RGB")
            
            # InternVL3 specific message formatting with <image> token
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "<image>" + ABLATION_SYSTEM_MESSAGE + ABLATION_USER_QUERY}
                ]
            }]
            
            # InternVL's apply_chat_template expects a list of dictionaries for messages
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # For InternVL, processor directly handles the image object in inputs
            inputs = processor(
                text=text,  # text should be a single string for InternVL in this context
                images=image,
                padding=True,
                return_tensors="pt",
            ).to(device)
            
            with torch.no_grad():
                generate_ids = model.generate(**inputs, max_new_tokens=128, do_sample=True, top_k=50, top_p=0.95)
            
            output = processor.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
            # InternVL output often includes system/user/assistant prefixes
            output = re.sub(r'^(system|user|assistant)', '', output, flags=re.MULTILINE).strip()
            
            matches = re.findall(r'\{[^{}]+\}', output, re.DOTALL)
            label = json.loads(matches[-1]).get("label", "unknown").strip() if matches else "unknown"
            
            inference_time = time.time() - start_time

            local_predictions.append(label)
            local_inference_times.append(inference_time)
            local_ground_truth.append(row['label'])

        except Exception as e:
            print(f"[Worker {gpu_id}] Error processing {row['image_path']}: {e}")
            local_predictions.append("error")
            local_inference_times.append(0)
            local_ground_truth.append(row['label'])
            
    print(f"[Worker {gpu_id}] Finished processing.")
    return local_predictions, local_inference_times, local_ground_truth


def main():
    parser = argparse.ArgumentParser(description="Parallel Ablation Study for InternVL3-2B-hf on Multiple GPUs")
    parser.add_argument("--model_name", type=str, required=True, help="Name for the model, used for output files")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV")
    parser.add_argument("--output_dir", type=str, default="results_ablation_parallel", help="Directory to save results")
    parser.add_argument("--model_path", type=str, default="OpenGVLab/InternVL3-2B-hf", help="HuggingFace model path") # Updated default model_path
    parser.add_argument("--num_gpus", type=int, default=4, help="Number of GPUs to use")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    df = pd.read_csv(args.input_csv)
    # Ensure image paths are valid before starting workers
    df['image_path'] = df['image_path'].apply(lambda p: os.path.join('fine_tune_dataset', 'test', 'images', os.path.basename(p)) if not os.path.exists(p) else p)
    df = df[df['image_path'].apply(os.path.exists)].reset_index(drop=True)
    print(f"Found {len(df)} valid images to process.")

    # Split dataframe into chunks for each GPU
    data_chunks = np.array_split(df, args.num_gpus)
    
    # Prepare arguments for each worker process
    worker_args = [(i, data_chunks[i], args.model_path) for i in range(args.num_gpus)]
    
    print(f"Starting parallel processing on {args.num_gpus} GPUs...")
    
    # --- Multiprocessing Pool ---
    with multiprocessing.Pool(processes=args.num_gpus) as pool:
        # starmap unpacks the tuples in worker_args for the worker_process function
        results = pool.starmap(worker_process, worker_args)
    
    # --- Aggregate results ---
    predictions = []
    inference_times = []
    ground_truth = []
    for res in results:
        predictions.extend(res[0])
        inference_times.extend(res[1])
        ground_truth.extend(res[2])
        
    print(f"Processing complete! Average inference time: {np.mean(inference_times):.3f}s")
    
    # --- Save and Analyze Results (same as original script) ---
    results_df = pd.DataFrame({
        'true_label': ground_truth,
        'predicted_label': predictions,
        'inference_time': inference_times,
        'correct': [t == p for t, p in zip(ground_truth, predictions)]
    })
    
    csv_out = os.path.join(args.output_dir, f"{args.model_name}_parallel_results.csv")
    results_df.to_csv(csv_out, index=False)
    print(f"Results saved to {csv_out}")
    
    accuracy = accuracy_score(ground_truth, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(ground_truth, predictions, average='weighted', zero_division=0)
    
    print("\nModel Performance Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}\n")
    
    labels = sorted(list(set(ground_truth)))
    cm = confusion_matrix(ground_truth, predictions, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix for {args.model_name} (Parallel)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"{args.model_name}_parallel_cm.png"))
    plt.close()
    print(f"Plots saved to {args.output_dir}")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
