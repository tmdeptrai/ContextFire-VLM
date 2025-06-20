import base64
import json
import time
import re
import pandas as pd
import numpy as np
import concurrent.futures
import requests
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

# ─── 1. Load dataset and set up ────────────────────────────────────────────────
df = pd.read_csv('fire_captions_combined.csv')
URL = "http://127.0.0.1:8080/v1/chat/completions"

prompt = (
    "You are a visual analyst evaluating an image for signs of fire and the surrounding context. "
    "Do the following tasks:\n"
    "1: Summarize what you see in the image. Describe the environment, key objects, people, and any signs of fire or smoke.\n"
    "2: Based on your summary, classify the fire situation: "
    "no fire (e.g., fire alarm, fire distinguisher, fireplace with no fire..), controlled fire (e.g., fireplace contains fire, campfire, cooking, candles, match stick, lighter..) or "
    "a dangerous/uncontrolled fire (e.g., curtains on fire, smoke on ceiling, couch on fire, bed sheet on fire, spreading fire on furniture..).\n"
    "Return only this JSON format:\n"
    "{ \"caption\": \"...\", \"label\": \"no fire\"|\"controlled fire\"|\"dangerous fire\" }"
)

def call_llama(image_path):
    with open(image_path, 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode()
    data_url = f"data:image/jpeg;base64,{img_b64}"

    payload = {
        "max_tokens": 100,
        "messages": [
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]
            }
        ]
    }

    start = time.time()
    resp = requests.post(URL, json=payload).json()
    raw = resp["choices"][0]["message"]["content"]
    js = re.sub(r"^```json|```$", "", raw.strip(), flags=re.M).strip()
    out = json.loads(js)
    return out["label"], out["caption"], time.time() - start

# ─── 2. Batch inference with concurrency ───────────────────────────────────────
concurrency = 8
results = []

with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
    futures = {
        executor.submit(call_llama, row['image_path']): idx
        for idx, row in df.iterrows()
    }
    for fut in concurrent.futures.as_completed(futures):
        idx = futures[fut]
        label, caption, t = fut.result()
        print(idx,label)
        results.append((idx, label, caption, t))

# ─── 3. Assemble results into DataFrame ───────────────────────────────────────
results_df = pd.DataFrame(results, columns=['idx', 'predicted_label', 'caption', 'inference_time'])
results_df = results_df.sort_values('idx').reset_index(drop=True)
results_df['image_path'] = df['image_path']
results_df['true_label'] = df['label']
results_df['correct'] = results_df['true_label'] == results_df['predicted_label']

# Save to CSV
results_df.to_csv('gemma_batch_results.csv', index=False)
print("Results saved to gemma_batch_results.csv")

# ─── 4. Compute and print metrics ──────────────────────────────────────────────
accuracy = accuracy_score(results_df['true_label'], results_df['predicted_label'])
precision, recall, f1, _ = precision_recall_fscore_support(
    results_df['true_label'],
    results_df['predicted_label'],
    average='weighted'
)

print("\nModel Performance Metrics:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}\n")

# ─── 5. Plot confusion matrix ──────────────────────────────────────────────────
labels = sorted(results_df['true_label'].unique())
cm = confusion_matrix(results_df['true_label'], results_df['predicted_label'], labels=labels)

plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation='nearest')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.colorbar()
plt.xticks(np.arange(len(labels)), labels, rotation=45)
plt.yticks(np.arange(len(labels)), labels)
plt.tight_layout()
plt.show()

# ─── 6. Plot inference time distribution ───────────────────────────────────────
times = results_df['inference_time']
print(f"Inference time (s): avg={times.mean():.3f}, std={times.std():.3f}, "
      f"min={times.min():.3f}, max={times.max():.3f}")

plt.figure(figsize=(6, 4))
plt.hist(times, bins=30)
plt.title('Inference Time Distribution')
plt.xlabel('Time (seconds)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

