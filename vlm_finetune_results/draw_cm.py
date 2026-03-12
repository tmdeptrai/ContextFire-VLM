import pandas as pd
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("vlm_finetune_results/qwen2.5_vl_3B_FINE_TUNED_WEIGHTED_results.csv")
ground_truth = df["true_label"]
predictions = df["predicted_label"]

# Plot confusion matrix
labels = sorted(df['true_label'].unique())
cm = confusion_matrix(ground_truth, predictions, labels=labels)
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=labels, yticklabels=labels
)
plt.title('Confusion Matrix for Qwen2.5VL_3B_FINE_TUNED_WEIGHTED') # Changed plot title
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('vlm_finetune_results/qwen2.5VL_3B_fine_tuned_weighted_confusion_matrix.png')
