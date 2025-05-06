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

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", cache_dir=cache_dir)
blip2      = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    load_in_8bit=True, device_map="auto", cache_dir=cache_dir
)

tokenizer_phi = AutoTokenizer.from_pretrained("microsoft/phi-1_5")
model_phi     = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-1_5", torch_dtype=torch.float16, device_map="auto"
)
# ensure pad_token
if tokenizer_phi.pad_token is None:
    tokenizer_phi.pad_token = tokenizer_phi.eos_token
    model_phi.config.pad_token_id = tokenizer_phi.eos_token_id

# ── 2) Helpers ─────────────────────────────────────────────────────────────

def generate_caption(img_pil: Image.Image) -> str:
    batch   = processor(images=img_pil, return_tensors="pt").to(device)
    out_ids = blip2.generate(**batch, max_new_tokens=30)
    return processor.decode(out_ids[0], skip_special_tokens=True).strip()

def sequence_decision_with_probs(captions: list[str]):
    # build a prompt listing all captions
    prompt = (
        "You are a fire safety advisor. Decide if the fire situation is dangerous.\n"
        "Rule: if the fire is on any object other than a candle, oil lamp, or match stick, answer Yes; otherwise answer No.\n\n"
    )
    for i, cap in enumerate(captions, 1):
        prompt += f"Frame {i}: \"{cap}\"\n"
    prompt += (
        "\nQuestion: Based on these frames, is the fire SPREADING? "
        "Answer with Yes or No.\nAnswer:"
    )

    toks = tokenizer_phi(prompt, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        logits = model_phi(input_ids=toks.input_ids, attention_mask=toks.attention_mask).logits
    next_logits = logits[0, -1, :]

    yes_id = tokenizer_phi.encode(" Yes", add_special_tokens=False)[0]
    no_id  = tokenizer_phi.encode(" No",  add_special_tokens=False)[0]
    probs  = F.softmax(next_logits[[yes_id, no_id]], dim=0)
    p_yes, p_no = probs.tolist()
    label = "Yes" if p_yes > p_no else "No"
    return label, p_yes, p_no

# ── 3) VIDEO PROCESSING w/ last sampled caption overlay ─────────────────────
input_path   = "video1.mp4"
output_path  = "output_video1.mp4"
sample_rate  = 15
window_size  = 10

cap    = cv2.VideoCapture(input_path)
fps    = cap.get(cv2.CAP_PROP_FPS)
W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

buffer      = deque(maxlen=window_size)
frame_idx   = 0
last_lbl    = "No"
last_py     = 0.0
last_pn     = 1.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # every Nth frame, update buffer and decision
    if frame_idx % sample_rate == 0:
        pil      = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap_txt  = generate_caption(pil)
        buffer.append(cap_txt)

        if len(buffer) == window_size:
            last_lbl, last_py, last_pn = sequence_decision_with_probs(list(buffer))

    # what to show
    last_caption = buffer[-1] if buffer else ""
    line1 = f"Caption: {last_caption}"
    line2 = f"Fire spreading? {last_lbl}  (Yes: {last_py:.2f}, No: {last_pn:.2f})"

    # text settings
    font      = cv2.FONT_HERSHEY_SIMPLEX
    scale     = 0.6
    thickness = 1
    margin    = 5

    # measure both lines
    (w1, h1), _ = cv2.getTextSize(line1, font, scale, thickness)
    (w2, h2), _ = cv2.getTextSize(line2, font, scale, thickness)
    box_h       = h1 + h2 + margin * 3
    box_w       = max(w1, w2) + margin * 2

    # draw white background
    cv2.rectangle(frame, (0, 0), (box_w, box_h), (255, 255, 255), -1)

    # draw line1
    y1 = margin + h1
    cv2.putText(frame, line1, (margin, y1), font, scale, (0, 0, 0), thickness)

    # draw line2
    y2 = y1 + margin + h2
    cv2.putText(frame, line2, (margin, y2), font, scale, (0, 0, 0), thickness)

    # write frame
    writer.write(frame)
    frame_idx += 1

cap.release()
writer.release()
print(f"✅ Done! Saved annotated video to {output_path}")

