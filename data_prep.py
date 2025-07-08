import pandas as pd
import json
import os

# Prompt template
prompt_text = (
    "<image>\n"
    "Summarize this situation in the image, look for signs of fire and smoke and classify whether the situation is:\n"
    "- no fire (e.g., fire alarm, fire distinguisher,...)\n"
    "- controlled fire (e.g., fireplace, cooking, campfire,...)\n"
    "- dangerous/uncontrolled fire (e.g., curtains on fire, smoke on ceiling, spreading flames...)\n\n"
    "Add a brief explanation in the caption if necessary.\n"
    "Respond only in this JSON format:\n"
    "{ \"caption\": \"...\", \"label\": \"no fire\"|\"controlled fire\"|\"dangerous fire\" }"
)

# Function to create one example
def format_for_qwen(sample):
    image_filename = os.path.basename(sample["image_path"])
    return {
        "id": os.path.splitext(image_filename)[0],
        "image": image_filename,
        "conversations": [
            {
                "from": "human",
                "value": prompt_text
            },
            {
                "from": "gpt",
                "value": f"""```json\n{json.dumps({"caption": sample["caption"], "label": sample["label"]}, ensure_ascii=False)}\n```"""
            }
        ]
    }

# Load CSVs
train_df = pd.read_csv("labels_v2.csv")
val_df = pd.read_csv("val_labels_shortened.csv")
test_df = pd.read_csv("test_labels.csv")

# Process datasets
train_data = [format_for_qwen(row) for row in train_df.to_dict("records")]
val_data = [format_for_qwen(row) for row in val_df.to_dict("records")]
test_data = [format_for_qwen(row) for row in test_df.to_dict("records")]

# Save to JSON files
with open("qwen_train.json", "w") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

with open("qwen_val.json", "w") as f:
    json.dump(val_data, f, ensure_ascii=False, indent=2)

with open("qwen_test.json", "w") as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)

print("✅ Saved qwen_train.json, qwen_val.json, qwen_test.json")
