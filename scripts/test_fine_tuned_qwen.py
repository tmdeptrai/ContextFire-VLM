import torch
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
import pandas as pd
import json
from PIL import Image
from qwen_vl_utils import process_vision_info
import time
import re
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Model and Processor Initialization
model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    device_map="cuda",
    torch_dtype=torch.bfloat16,
)
processor = Qwen2_5_VLProcessor.from_pretrained(model_id, use_fast=True)

# Load Adapter
adapter_path = "qwen2-7b-instruct-trl-sft-NEW"
model.load_adapter(adapter_path)

# 2. System and User Prompts
system_message = """You are a Vision Language Model specialized in detecting clues of fire, smoke and surrounding context then classify them as no fire, dangerous fire or controlled fire.
Your task is to analyze the provided image and respond to queries with concise answers, usually a json format of a caption and a label.
Summarize what you see in the image. Describe the environment, key objects, people, and any signs of fire or smoke.
Based on your summary, classify the fire situation: no fire(e.g., fire alarm, fire distinguisher,..), controlled fire (e.g., fireplace emitting, campfire, cooking, candles, match stick, lighter..) or a dangerous/uncontrolled fire (e.g., curtains on fire, smoke on ceiling, couch on fire, bed sheet on fire, spreading fire on furniture..)
Focus on delivering accurate, succinct caption and precise label based on the visual information. Add a brief explanation for your choice of label in the caption if necessary."""

user_query = """Summarize this situation in the image, look for signs of fire and smoke and classify whether the situation is: 

- **No fire**: The image may contain symbols, representations, or objects related to fire, but no actual fire or smoke is present. This includes things like warning signs, fire safety equipment (extinguishers, alarms), digital or printed representations of fire (e.g., a laptop screen showing fire, a drawing or painting), or thematic decorations.

- **Controlled fire**: There is a visible flame or heat source, but it is clearly contained, expected, and managed by people or objects in its environment. This includes fireplaces, campfires, cooking appliances, candles, lighters, or matchsticks. There should be no signs of danger, spread, panic, or damage in the surroundings.

- **Dangerous (uncontrolled) fire**: The fire appears out of control or harmful to the environment or people. Signs include flames spreading to flammable materials (e.g., curtains, furniture, bedding), thick smoke near the ceiling, charring, visible structural damage, or people reacting with fear or urgency. The context suggests potential or ongoing property damage or bodily harm.

Add a brief explanation for your choice of label in the caption if necessary.

Respond only this json format:

{ "caption": "...", "label": "no fire"|"controlled fire"|"dangerous fire" }
"""

def format_data(sample):
    return [
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
        {"role": "user", "content": [{"type": "image", "image": f"file://{sample['image_path']}"}, {"type": "text", "text": user_query}]},
    ]

# 3. Generation and Processing Functions
def generate_text_from_sample(model, processor, sample, max_new_tokens=1024, device="cuda"):
    text_input = processor.apply_chat_template(sample, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(sample)
    model_inputs = processor(text=[text_input], images=image_inputs, return_tensors="pt").to(device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)
    trimmed_generated_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(trimmed_generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return output_text[0]

def process_sample(sample):
    try:
        start_time = time.time()
        clean_output = generate_text_from_sample(model, processor, sample)
        matches = re.findall(r'\{[^{}]+\}', clean_output, re.DOTALL)
        if matches:
            try:
                result = json.loads(matches[-1])
                caption = result.get("caption", "").strip()
                label = result.get("label", "").strip()
            except Exception as e:
                caption = clean_output
                label = "unknown"
        else:
            caption = clean_output
            label = "unknown"
        inference_time = time.time() - start_time
        return label, caption, inference_time
    except Exception as e:
        return "error", "", 0.0

# 4. Load Data and Run Inference
df = pd.read_csv('vlm_finetune/test_labels.csv')
predictions, captions, inference_times, ground_truth = [], [], [], []

for idx, row in df.iterrows():
    sample = format_data(row)
    pred, caption, inf_time = process_sample(sample)
    predictions.append(pred)
    captions.append(caption)
    inference_times.append(inf_time)
    ground_truth.append(row['label'])

# 5. Save and Analyze Results
results_df = pd.DataFrame({
    'image_path': df['image_path'],
    'true_label': ground_truth,
    'predicted_label': predictions,
    'caption': captions,
    'inference_time': inference_times,
    'correct': [t == p for t, p in zip(ground_truth, predictions)]
})
results_df.to_csv('qwen2.5_vl_3B_FINE_TUNED_results.csv', index=False)

# Calculate and print metrics
accuracy = accuracy_score(ground_truth, predictions)
precision, recall, f1, _ = precision_recall_fscore_support(ground_truth, predictions, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Plot confusion matrix
labels = sorted(df['label'].unique())
cm = confusion_matrix(ground_truth, predictions, labels=labels)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix for Qwen2.5_VL_3B_FINE_TUNED')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('qwen_confusion_matrix.png')
