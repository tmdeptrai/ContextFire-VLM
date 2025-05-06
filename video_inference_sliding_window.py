import cv2
import time
import torch
import torch.nn.functional as F
from collections import deque
from PIL import Image
from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
    AutoTokenizer,
    AutoModelForCausalLM,
)

# ── 1) Setup ───────────────────────────────────────────────────────────────
device    = "cuda" if torch.cuda.is_available() else "cpu"
cache_dir = "./models/blip2"

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", cache_dir=cache_dir,use_fast=True)
blip2      = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    load_in_8bit=True, device_map="auto", cache_dir=cache_dir
)

mistral_path = "tiiuae/falcon-7b-instruct"

tokenizer_mistral = AutoTokenizer.from_pretrained(mistral_path)
model_mistral     = AutoModelForCausalLM.from_pretrained(
    mistral_path, torch_dtype=torch.float16, device_map="auto"
)
# ensure pad_token
if tokenizer_mistral.pad_token is None:
    tokenizer_mistral.pad_token = tokenizer_mistral.eos_token
    model_mistral.config.pad_token_id = tokenizer_mistral.eos_token_id

# ── 2) Helpers ─────────────────────────────────────────────────────────────

def generate_caption(img_pil: Image.Image) -> str:
    batch   = processor(images=img_pil, return_tensors="pt").to(device)
    out_ids = blip2.generate(**batch, max_new_tokens=30)
    return processor.decode(out_ids[0], skip_special_tokens=True).strip()

def sequence_decision_with_probs(captions: list[str]):
    # build a prompt listing all captions
    prompt = (
        "You are a safety classification assistant. Decide if a fire is dangerous.\n\n"
        "Rule:\n"
        "If the fire is on anything other than a candle, oil lamp, or match stick, answer Yes (dangerous).\n"
        "If the fire is on a candle, oil lamp, or match stick, answer No (not dangerous).\n\n"
        "Examples:\n"
        "Fire on a wooden table → Yes\n"
        "Fire on a match stick → No\n"
        "Fire on an oil lamp → No\n"
        "Fire in a trash bin → Yes\n"
        "Fire on a candle → No\n"
        "Fire on a piece of clothing → Yes\n\n"
        "Now classify the following frames:\n"
    )
    for i, cap in enumerate(captions, 1):
        prompt += f"Frame {i}: \"{cap}\"\n"
    prompt += (
        "\nQuestion: Based on these frames, is the fire DANGEROUS? "
        "Answer with Yes or No.\nAnswer:"
    )

    toks = tokenizer_mistral(prompt, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        logits = model_mistral(input_ids=toks.input_ids, attention_mask=toks.attention_mask).logits
    next_logits = logits[0, -1, :]

    yes_id = tokenizer_mistral.encode(" Yes", add_special_tokens=False)[0]
    no_id  = tokenizer_mistral.encode(" No",  add_special_tokens=False)[0]
    probs  = F.softmax(next_logits[[yes_id, no_id]], dim=0)
    p_yes, p_no = probs.tolist()
    label = "Yes" if p_yes > p_no else "No"
    return label, p_yes, p_no

# ── 3) VIDEO PROCESSING w/ BLIP & LLM timings overlay ───────────────────
input_path    = "video4.mp4"
output_path   = "output_video4_with_times.mp4"
sample_rate   = 15
window_size   = 5   # sliding window of last 5 captions

cap    = cv2.VideoCapture(input_path)
fps    = cap.get(cv2.CAP_PROP_FPS)
W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

buffer             = deque(maxlen=window_size)
frame_idx          = 0
last_lbl, last_py, last_pn = "No", 0.0, 1.0
last_blip_time     = 0.0
last_decision_time = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1) Sample every N frames
    if frame_idx % sample_rate == 0:
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # time BLIP‑2 captioning
        t0 = time.time()
        cap_txt = generate_caption(pil)
        last_blip_time = time.time() - t0

        buffer.append(cap_txt)

        # once window is full, time LLM decision
        if len(buffer) == window_size:
            t1 = time.time()
            last_lbl, last_py, last_pn = sequence_decision_with_probs(list(buffer))
            last_decision_time = time.time() - t1

    # 2) Prepare overlay text
    last_caption = buffer[-1] if buffer else ""
    line1 = f"Caption: {last_caption}"
    line2 = f"BLIP time: {last_blip_time:.2f}s"
    line3 = f"Fire spreading? {last_lbl}  (Yes: {last_py:.2f}, No: {last_pn:.2f})"
    line4 = f"LLM time: {last_decision_time:.2f}s"

    # 3) Draw white box sized for four lines
    font, scale, thickness, margin = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1, 5
    (w1, h1), _ = cv2.getTextSize(line1, font, scale, thickness)
    (w2, h2), _ = cv2.getTextSize(line2, font, scale, thickness)
    (w3, h3), _ = cv2.getTextSize(line3, font, scale, thickness)
    (w4, h4), _ = cv2.getTextSize(line4, font, scale, thickness)
    box_h = h1 + h2 + h3 + h4 + margin * 5
    box_w = max(w1, w2, w3, w4) + margin * 2

    cv2.rectangle(frame, (0, 0), (box_w, box_h), (255, 255, 255), -1)

    # 4) Render the four lines
    y = margin + h1
    cv2.putText(frame, line1, (margin, y), font, scale, (0, 0, 0), thickness)
    y += margin + h2
    cv2.putText(frame, line2, (margin, y), font, scale, (0, 0, 0), thickness)
    y += margin + h3
    cv2.putText(frame, line3, (margin, y), font, scale, (0, 0, 0), thickness)
    y += margin + h4
    cv2.putText(frame, line4, (margin, y), font, scale, (0, 0, 0), thickness)

    # 5) Write frame and increment
    writer.write(frame)
    frame_idx += 1

cap.release()
writer.release()
print(f"✅ Done! Saved annotated video to {output_path}")
