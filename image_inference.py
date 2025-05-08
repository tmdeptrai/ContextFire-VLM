import time
import torch
from PIL import Image
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    pipeline
)

# ── 1) Setup ────────────────────────────────────────────────────────────────
device    = "cuda" if torch.cuda.is_available() else "cpu"
cache_dir = "./models/blip"

# BLIP (faster than BLIP‑2)
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    cache_dir=cache_dir
)
blip = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    device_map="auto",
    cache_dir=cache_dir
)

# Classification pipeline (using your saved 3‑label model)
clf = pipeline(
    "text-classification",
    model="fire-risk-classifier",
    tokenizer="fire-risk-classifier",
    device=0 if device == "cuda" else -1,
    top_k=None      # return scores for all labels
)

# ── 2) Inference Functions ─────────────────────────────────────────────────

def generate_caption(image_path: str) -> str:
    img   = Image.open(image_path).convert("RGB")
    batch = processor(images=img, return_tensors="pt").to(device)
    ids   = blip.generate(**batch, max_new_tokens=30)
    return processor.decode(ids[0], skip_special_tokens=True).strip()

def classify_caption(caption: str):
    scores = clf(caption)[0]  # list of dicts
    label = max(scores, key=lambda x: x["score"])["label"]
    probabilities = {item["label"]: item["score"] for item in scores}
    return label, probabilities

# ── 3) Run on an Image ───────────────────────────────────────────────────────
if __name__ == "__main__":
    img_path = "kobe.png"  # ← your image here
    start    = time.time()

    # Generate caption
    caption = generate_caption(img_path)
    print(f"💬 Caption: {caption}")

    # Classify caption
    label, probabilities = classify_caption(caption)
    print(f"🔥 Dangerous? {label}")
    print(f"Probabilities: {probabilities}")
