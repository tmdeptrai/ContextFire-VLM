import pandas as pd
import re

def extract_final_label(text):
    """
    Extracts the very last potential label from the noisy text.
    Handles cases where the label is the last line or quoted.
    """
    if not isinstance(text, str):
        return "extraction_error"
    
    # Pre-cleaning for some common failure patterns
    text = text.replace('**', '').replace("''", "'")

    # Regex to find one of the three labels, possibly surrounded by quotes
    # It will find all matches and we will take the last one.
    matches = re.findall(r"'?(no fire|controlled fire|dangerous fire)'?", text, re.IGNORECASE)
    
    if matches:
        # Return the last match found in the string
        return matches[-1].lower()
    else:
        # Fallback for cases where the model provides a text explanation
        if "cannot determine" in text or "caption is empty" in text or "is missing" in text:
            return "unknown"
        return "no_label_found"

# Load the erroneous CSV file
input_path = 'results_2_stage/internvl3-2B_2_stage_2_stage_results.csv'
output_path = 'results_2_stage/internvl3-2B_2_stage_2_stage_results_CLEANED.csv'

try:
    df = pd.read_csv(input_path)

    # Apply the cleaning function to the 'predicted_label' column
    df['predicted_label'] = df['predicted_label'].apply(extract_final_label)

    # Recalculate the 'correct' column based on the cleaned labels
    df['correct'] = df.apply(lambda row: row['true_label'] == row['predicted_label'], axis=1)

    # Save the cleaned dataframe to a new file
    df.to_csv(output_path, index=False)

    print(f"Successfully cleaned the file and saved it to:{output_path}")

    # Optional: Display a summary of the cleaned data
    print("--- Summary of Cleaned Data ---")
    print("Value counts for 'predicted_label':")
    print(df['predicted_label'].value_counts())
    print(f"New accuracy: {df['correct'].mean():.4f}")

except FileNotFoundError:
    print(f"Error: The input file was not found at '{input_path}'")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
