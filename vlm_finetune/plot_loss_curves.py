import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the datasets
df_baseline = pd.read_csv('./vlm_finetune/unfrozen.csv')
df_dropout = pd.read_csv('./vlm_finetune/unfrozen_dropout_0.1.csv')
df_alpha = pd.read_csv('./vlm_finetune/unfrozen_alpha_32.csv')

# Create a figure with two subplots (one for training, one for validation)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

# --- Subplot 1: Training Loss Curves ---
ax1.plot(df_baseline['global_steps'], df_baseline['train/loss'], label='Baseline (Dropout=0.05, Alpha=16)', marker='o', linestyle='-')
ax1.plot(df_dropout['global_steps'], df_dropout['train/loss'], label='Experiment 1 (Dropout=0.1)', marker='x', linestyle='--')
ax1.plot(df_alpha['global_steps'], df_alpha['train/loss'], label='Experiment 2 (Alpha=32)', marker='s', linestyle=':')
ax1.set_title('Training Loss Comparison')
ax1.set_ylabel('Train Loss')
ax1.legend()
ax1.grid(True)

# --- Subplot 2: Validation Loss Curves ---
ax2.plot(df_baseline['global_steps'], df_baseline['eval/loss'], label='Baseline (Dropout=0.05, Alpha=16)', marker='o', linestyle='-')
ax2.plot(df_dropout['global_steps'], df_dropout['eval/loss'], label='Experiment 1 (Dropout=0.1)', marker='x', linestyle='--')
ax2.plot(df_alpha['global_steps'], df_alpha['eval/loss'], label='Experiment 2 (Alpha=32)', marker='s', linestyle=':')
ax2.set_title('Validation Loss Comparison')
ax2.set_xlabel('Global Steps')
ax2.set_ylabel('Validation Loss')
ax2.legend()
ax2.grid(True)

# Find and plot the minimum validation loss for each experiment
min_loss_baseline = df_baseline['eval/loss'].min()
min_step_baseline = df_baseline['global_steps'][df_baseline['eval/loss'].idxmin()]
ax2.axhline(y=min_loss_baseline, color='blue', linestyle='--', alpha=0.5, label=f'Baseline Min Loss: {min_loss_baseline:.4f} at step {min_step_baseline}')

min_loss_dropout = df_dropout['eval/loss'].min()
min_step_dropout = df_dropout['global_steps'][df_dropout['eval/loss'].idxmin()]
ax2.axhline(y=min_loss_dropout, color='orange', linestyle='--', alpha=0.5, label=f'Dropout Min Loss: {min_loss_dropout:.4f} at step {min_step_dropout}')

min_loss_alpha = df_alpha['eval/loss'].min()
min_step_alpha = df_alpha['global_steps'][df_alpha['eval/loss'].idxmin()]
ax2.axhline(y=min_loss_alpha, color='green', linestyle='--', alpha=0.5, label=f'Alpha Min Loss: {min_loss_alpha:.4f} at step {min_step_alpha}')

ax2.legend() # Re-call legend to include axhline labels

plt.suptitle('Parameter Sweep: Impact of LoRA Dropout and Alpha on Loss', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make room for suptitle
plt.show()
