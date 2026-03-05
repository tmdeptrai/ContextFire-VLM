
import pandas as pd
from sklearn.model_selection import train_test_split
import os

SOURCE_CSV = 'labels_v2.csv'
OUTPUT_DIR = 'vlm_finetune'
TRAIN_FILE = os.path.join(OUTPUT_DIR, 'train_labels.csv')
VAL_FILE = os.path.join(OUTPUT_DIR, 'val_labels.csv')
TEST_FILE = os.path.join(OUTPUT_DIR, 'test_labels.csv')

SEED = 42
TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15


def split_dataset():
    """
    Reads the full dataset, performs a stratified split into train, validation,
    and test sets, and saves them to new CSV files.
    """
    print(f"Reading source dataset from '{SOURCE_CSV}'...")
    try:
        df = pd.read_csv(SOURCE_CSV)
    except FileNotFoundError:
        print(f"Error: The source file '{SOURCE_CSV}' was not found.")
        print("Please ensure the file exists in the root directory.")
        return

    if 'label' not in df.columns:
        print("Error: The CSV file must contain a 'label' column for stratification.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory '{OUTPUT_DIR}' ensured.")

    train_df, temp_df = train_test_split(
        df,
        train_size=TRAIN_SIZE,
        stratify=df['label'],
        random_state=SEED
    )

    relative_test_size = TEST_SIZE / (VAL_SIZE + TEST_SIZE)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        stratify=temp_df['label'],
        random_state=SEED
    )

    # Save the splits to CSV files
    train_df.to_csv(TRAIN_FILE, index=False)
    val_df.to_csv(VAL_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)

    print("--- Split Summary ---")
    print(f"Source DataFrame shape: {df.shape}")
    print(f"Train DataFrame shape:  {train_df.shape}")
    print(f"Val DataFrame shape:    {val_df.shape}")
    print(f"Test DataFrame shape:   {test_df.shape}")
    print("--- Label Distribution ---")
    print("Source:")
    print(df['label'].value_counts(normalize=True))
    print("Train:")
    print(train_df['label'].value_counts(normalize=True))
    print("Validation:")
    print(val_df['label'].value_counts(normalize=True))
    print("Test:")
    print(test_df['label'].value_counts(normalize=True))
    print(f"Successfully created split files in '{OUTPUT_DIR}' directory.")


if __name__ == '__main__':
    split_dataset()
