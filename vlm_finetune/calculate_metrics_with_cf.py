import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Load the CSV file
# df = pd.read_csv("vlm_finetune/qwen2.5_vl_3B_org.csv")
# df = pd.read_csv("vlm_finetune/qwen2.5_vl_3B_FINE_TUNED_results.csv")
df = pd.read_csv("results_2_stage/internvl3-2B_2_stage_results.csv")
df = df.dropna(subset=['true_label', 'predicted_label', 'inference_time'])  # Remove rows with missing values

# Extract relevant columns
ground_truth = df['true_label'].astype(str).str.strip()
predictions = df['predicted_label'].astype(str).str.strip()
inference_times = df['inference_time'].astype(float)

# Calculate classification metrics
accuracy = accuracy_score(ground_truth, predictions)
precision, recall, f1, _ = precision_recall_fscore_support(
    ground_truth, predictions, average='weighted', zero_division=0
)

# Print performance metrics
print("Model Performance Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# --- Calculate TP, FP, TN, FN for each class ---
print("Detailed Class-wise Metrics (TP, FP, TN, FN):")
labels = sorted(ground_truth.unique()) # Get all unique labels and sort them for consistent indexing
cm = confusion_matrix(ground_truth, predictions, labels=labels)

for i, label in enumerate(labels):
    tp = cm[i, i] # True Positives for this class
    fp = cm[:, i].sum() - tp # False Positives for this class
    fn = cm[i, :].sum() - tp # False Negatives for this class
    tn = cm.sum() - (tp + fp + fn) # True Negatives for this class

    print(f"  Class '{label}':")
    print(f"    True Positives (TP): {tp}")
    print(f"    False Positives (FP): {fp}")
    print(f"    False Negatives (FN): {fn}")
    print(f"    True Negatives (TN): {tn}")

# Inference time stats
print("Inference Time Statistics:")
print(f"Average: {np.mean(inference_times):.3f}s")
print(f"Std Dev: {np.std(inference_times):.3f}s")
print(f"Min: {np.min(inference_times):.3f}s")
print(f"Max: {np.max(inference_times):.3f}s")
