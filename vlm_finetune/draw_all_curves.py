
import pandas as pd
import matplotlib.pyplot as plt

def generate_combined_loss_plot():
    """
    Reads data from four different experiment CSVs and plots their
    training and evaluation loss curves on a single figure.
    """
    try:
        # --- Load Data ---
        df_frozen = pd.read_csv("vlm_finetune/frozen.csv")
        df_unfrozen = pd.read_csv("vlm_finetune/unfrozen.csv")
        df_frozen_weighted = pd.read_csv("vlm_finetune/frozen_weighted.csv")
        df_unfrozen_weighted = pd.read_csv("vlm_finetune/unfrozen_weighted.csv")
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure all four CSV files are in the 'vlm_finetune' directory.")
        return

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 8))

    # FT1: Frozen Vision Encoder
    plt.plot(df_frozen["global_steps"], df_frozen["train/loss"], label="FT1 Train Loss (Frozen)", color='blue', linestyle='-')
    plt.plot(df_frozen["global_steps"], df_frozen["eval/loss"], label="FT1 Eval Loss (Frozen)", color='blue', linestyle='--')

    # FT2: Unfrozen Vision Encoder
    plt.plot(df_unfrozen["global_steps"], df_unfrozen["train/loss"], label="FT2 Train Loss (Unfrozen)", color='red', linestyle='-')
    plt.plot(df_unfrozen["global_steps"], df_unfrozen["eval/loss"], label="FT2 Eval Loss (Unfrozen)", color='red', linestyle='--')
    
    # FT1-Weighted: Frozen Vision Encoder with Weighted Loss
    plt.plot(df_frozen_weighted["global_steps"], df_frozen_weighted["train/loss"], label="FT1-W Train Loss (Frozen, Weighted)", color='green', linestyle='-')
    plt.plot(df_frozen_weighted["global_steps"], df_frozen_weighted["eval/loss"], label="FT1-W Eval Loss (Frozen, Weighted)", color='green', linestyle='--')

    # FT2-Weighted: Unfrozen Vision Encoder with Weighted Loss
    plt.plot(df_unfrozen_weighted["global_steps"], df_unfrozen_weighted["train/loss"], label="FT2-W Train Loss (Unfrozen, Weighted)", color='purple', linestyle='-')
    plt.plot(df_unfrozen_weighted["global_steps"], df_unfrozen_weighted["eval/loss"], label="FT2-W Eval Loss (Unfrozen, Weighted)", color='purple', linestyle='--')

    # --- Labels, Title, and Legend ---
    plt.xlabel("Steps", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Comparison of Fine-Tuning Strategies: Training and Evaluation Loss", fontsize=14)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True)
    
    # Set y-axis limit to better visualize the convergence
    plt.ylim(0, max(df_frozen['eval/loss'].max(), df_frozen_weighted['eval/loss'].max()) * 1.1)
    
    # --- Save Figure ---
    output_path = "vlm_finetune/combined_loss_curves.png"
    plt.savefig(output_path, dpi=300)
    print(f"Successfully generated and saved the combined plot to '{output_path}'")

if __name__ == '__main__':
    generate_combined_loss_plot()
