import cv2
import time
import torch
from PIL import Image
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)

# ── 1) SETUP ───────────────────────────────────────────────────────────────
device    = "cuda" if torch.cuda.is_available() else "cpu"
cache_blip = "./models/blip"
cache_llm  = "./models/deepseek-7b"

# 1a) BLIP for captions
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    cache_dir=cache_blip,
    use_fast=True
)
blip = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    device_map="auto",
    cache_dir=cache_blip
)

# 1b) DeepSeek‑7B Base without quantization
tokenizer = AutoTokenizer.from_pretrained(
    "deepseek-ai/deepseek-llm-7b-base",
    cache_dir=cache_llm
)
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/deepseek-llm-7b-base",
    device_map="auto",  # Automatically offload parts of the model to CPU
    cache_dir=cache_llm,
    torch_dtype=torch.float16,  # Use float16 for reduced memory usage
    offload_folder="./offload_weights"  # Specify folder for offloaded weights
)

# 1c) Pipeline for generation
llm = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    torch_dtype=torch.float16,
    max_new_tokens=10,
    do_sample=False
)

FEW_SHOT_PROMPT = """You are a safety classification assistant. Choose one label from: No fire; Controlled fire; Dangerous fire.

Examples:
Smoke billowing from a building → Dangerous fire
A candle flickering on a table → Controlled fire
A peaceful park at sunset → No fire

Caption: "{caption}" →"""

# ── 2) HELPERS ─────────────────────────────────────────────────────────────

def generate_caption(img: Image.Image) -> str:
    inputs  = processor(images=img, return_tensors="pt").to(device)
    out_ids = blip.generate(**inputs, max_new_tokens=30)
    return processor.decode(out_ids[0], skip_special_tokens=True).strip()

def classify_with_llm(caption: str):
    prompt = FEW_SHOT_PROMPT.format(caption=caption)
    t0 = time.time()
    gen = llm(prompt)[0]["generated_text"]
    elapsed = time.time() - t0

    # parse the label after the arrow
    try:
        label = gen.split("→")[-1].strip().split("\n")[0]
    except:
        label = "No fire"
    return label, elapsed

# ── 3) VIDEO PROCESSING ────────────────────────────────────────────────────
input_path    = "video5.mp4"
output_path   = "output_video5_deepseek.mp4"
sample_rate   = 15

cap    = cv2.VideoCapture(input_path)
fps    = cap.get(cv2.CAP_PROP_FPS)
W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

frame_idx      = 0
last_caption   = ""
last_blip_time = 0.0
last_label     = ""
last_llm_time  = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % sample_rate == 0:
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # BLIP caption timing
        t0 = time.time()
        last_caption   = generate_caption(pil)
        last_blip_time = time.time() - t0

        # DeepSeek classification timing
        last_label, last_llm_time = classify_with_llm(last_caption)

    # Annotate the frame
    lines = [
        f"Frame: {frame_idx}",
        f"Caption: {last_caption}",
        f"BLIP time: {last_blip_time:.2f}s",
        f"Decision: {last_label}",
        f"LLM time: {last_llm_time:.2f}s"
    ]

    font, scale, thickness, m = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1, 5
    sizes = [cv2.getTextSize(l, font, scale, thickness)[0] for l in lines]
    box_h = sum(h for _,h in sizes) + m*(len(sizes)+1)
    box_w = max(w for w,_ in sizes) + m*2

    cv2.rectangle(frame, (0,0), (box_w, box_h), (255,255,255), -1)
    y = m
    for l in lines:
        y += cv2.getTextSize(l, font, scale, thickness)[0][1]
        cv2.putText(frame, l, (m,y), font, scale, (0,0,0), thickness)
        y += m

    cv2.imshow("Processing", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    writer.write(frame)
    frame_idx += 1

cap.release()
writer.release()
cv2.destroyAllWindows()
print("✅ Done! Saved:", output_path)
