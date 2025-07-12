import json

with open("qwen_train.json", "r") as f:
    data = json.load(f)

with open("internvl3_annotations.jsonl", "w") as f:
    for item in data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
