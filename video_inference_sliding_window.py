import cv2
import time
import torch
from PIL import Image
from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
    pipeline
)

# ── 1) Setup ───────────────────────────────────────────────────────────────
device    = "cuda" if torch.cuda.is_available() else "cpu"
cache_dir = "./models/blip2"

# BLIP‑2 (8‑bit)
processor = Blip2Processor.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    cache_dir=cache_dir,
    use_fast=True
)
blip2 = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    load_in_8bit=True,
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

# ── 2) Helpers ─────────────────────────────────────────────────────────────

def generate_caption(img: Image.Image) -> str:
    batch   = processor(images=img, return_tensors="pt").to(device)
    out_ids = blip2.generate(**batch, max_new_tokens=30)
    return processor.decode(out_ids[0], skip_special_tokens=True).strip()

# ── 3) Video processing ────────────────────────────────────────────────────
input_path    = "video4.mp4"
output_path   = "output_video4_no_window.mp4"
sample_rate   = 15

cap    = cv2.VideoCapture(input_path)
fps    = cap.get(cv2.CAP_PROP_FPS)
W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

frame_idx          = 0
last_caption       = ""
last_blip_time     = 0.0
last_clf_label     = ""
last_clf_time      = 0.0
last_scores        = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1) For each sampled frame, generate caption & classification
    if frame_idx % sample_rate == 0:
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # BLIP caption timing
        t0 = time.time()
        last_caption = generate_caption(pil)
        last_blip_time = time.time() - t0

        # Classification timing
        t1 = time.time()
        scores = clf(last_caption)[0]  # list of dicts
        last_clf_time = time.time() - t1

        # pick top label & collect all scores
        last_clf_label = max(scores, key=lambda x: x["score"])["label"]
        last_scores = {item["label"]: item["score"] for item in scores}

    # 2) Annotate current frame
    line1 = f"Frame: {frame_idx}"
    line2 = f"Caption: {last_caption}"
    line3 = f"BLIP time: {last_blip_time:.2f}s"
    line4 = f"Label: {last_clf_label}"
    probs_str = ", ".join(f"{lbl}:{score:.2f}" for lbl, score in last_scores.items())
    line5 = f"Scores: {probs_str}"
    line6 = f"Clf time: {last_clf_time:.2f}s"

    # Calculate box size
    font, scale, thickness, margin = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1, 5
    texts = [line1, line2, line3, line4, line5, line6]
    sizes = [cv2.getTextSize(txt, font, scale, thickness)[0] for txt in texts]
    box_h = sum(h for _, h in sizes) + margin * (len(texts) + 1)
    box_w = max(w for w, _ in sizes) + margin * 2

    # Draw background and text
    cv2.rectangle(frame, (0, 0), (box_w, box_h), (255, 255, 255), -1)
    y = margin
    for txt in texts:
        y += cv2.getTextSize(txt, font, scale, thickness)[0][1]
        cv2.putText(frame, txt, (margin, y), font, scale, (0, 0, 0), thickness)
        y += margin

    # Show processed frame in a window
    cv2.imshow("Processing", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Write frame to output video
    writer.write(frame)
    frame_idx += 1

cap.release()
writer.release()
cv2.destroyAllWindows()
print(f"✅ Done! Saved annotated video to {output_path}")