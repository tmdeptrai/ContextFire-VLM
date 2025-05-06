import time
import torch
from PIL import Image
import torch.nn.functional as F
from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
    AutoTokenizer,
    AutoModelForCausalLM,
)

# ── 1) Setup ────────────────────────────────────────────────────────────────
device    = "cuda" if torch.cuda.is_available() else "cpu"
cache_dir = "./models/blip2"

# BLIP‑2 in 8‑bit
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", cache_dir=cache_dir,use_fast=True)
blip2      = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    load_in_8bit=True,
    device_map="auto",
    cache_dir=cache_dir,
)

# Phi‑1.5 from transformers
tokenizer_phi = AutoTokenizer.from_pretrained("microsoft/phi-1_5")
model_phi     = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-1_5",
    torch_dtype=torch.float16,
    device_map="auto",
)

# --- Fix missing pad token ---
if tokenizer_phi.pad_token is None:
    tokenizer_phi.pad_token = tokenizer_phi.eos_token
    model_phi.config.pad_token_id = tokenizer_phi.eos_token_id


# ── 2) Inference Functions ─────────────────────────────────────────────────

def generate_caption(image_path: str) -> str:
    img   = Image.open(image_path).convert("RGB")
    batch = processor(images=img, return_tensors="pt").to(device)
    ids   = blip2.generate(**batch, max_new_tokens=50)
    return processor.decode(ids[0], skip_special_tokens=True).strip()

def llm_decision(caption: str) -> str:
    # Few-shot examples to anchor safe vs dangerous
    prompt = (
        "You are a fire safety advisor. Answer Yes or No.\n\n"
        "Examples:\n"
        "- Caption: \"A living room with a fireplace and couches\" -> No\n"
        "- Caption: \"Smoke and flames from windows of an apartment building.\" -> Yes\n\n"
        f"Caption: \"{caption}\" ->"
    )

    toks = tokenizer_phi(
        prompt,
        return_tensors="pt",
        padding=True,         # now works because pad_token is set
    ).to(device)

    out = model_phi.generate(
        input_ids=toks.input_ids,
        attention_mask=toks.attention_mask,
        max_new_tokens=1,     # only one token: “Yes” or “No”
        num_beams=1,
        do_sample=False,
        pad_token_id=tokenizer_phi.eos_token_id,
        eos_token_id=tokenizer_phi.eos_token_id,
    )

    # Extract just that one new token
    new_token = out[0, toks.input_ids.shape[-1]:]
    ans = tokenizer_phi.decode(new_token, skip_special_tokens=True).strip()
    return ans.capitalize()

def llm_decision_with_probs(caption: str):
    # 1) few‑shot + prompt
    prompt = (
        "You are a fire safety advisor. Based on the caption, answer Yes or No.\n\n"
        "Examples:\n"
        "- Caption: \"A fireplace in a living room with no visible danger.\" -> No\n"
        "- Caption: \"Smoke and flames from windows of an apartment building.\" -> Yes\n\n"
        f"Caption: \"{caption}\" ->"
    )
    # 2) tokenize with attention mask
    toks = tokenizer_phi(
        prompt,
        return_tensors="pt",
        padding=True,
    ).to(device)

    # 3) forward pass to get logits
    with torch.no_grad():
        outputs = model_phi(
            input_ids=toks.input_ids,
            attention_mask=toks.attention_mask,
        )
    # shape: (1, seq_len, vocab_size)
    next_token_logits = outputs.logits[0, -1, :]

    # 4) find the token IDs for " Yes" and " No"
    yes_id = tokenizer_phi.encode(" Yes", add_special_tokens=False)[0]
    no_id  = tokenizer_phi.encode(" No",  add_special_tokens=False)[0]

    # 5) compute probabilities
    probs = F.softmax(next_token_logits[[yes_id, no_id]], dim=0)
    p_yes, p_no = probs.tolist()

    # pick the higher one
    label = "Yes" if p_yes > p_no else "No"
    return label, p_yes, p_no

# ── 3) Run on an Image ───────────────────────────────────────────────────────
if __name__ == "__main__":
    img_path = "dangerous_fire.jpg"  # ← your image here
    start    = time.time()

    caption = generate_caption(img_path)
    print(f"💬 Caption: {caption}")

    decision = llm_decision_with_probs(caption)
    decision, p_yes, p_no = llm_decision_with_probs(caption)
    print(f"🔥 Dangerous? {decision}  (P(Yes)={p_yes:.2f}, P(No)={p_no:.2f})")
