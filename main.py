import torch
import cv2
import numpy as np
from PIL import Image
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    AutoTokenizer, AutoModelForCausalLM
)

def analyze_fire_image(image_path):
    # ===== Step 1: Load BLIP (Image Captioning) =====
    blip_save_path = "./models/blip"
    blip_model_path = "./models/blip/models--Salesforce--blip-image-captioning-base/snapshots/82a37760796d32b1411fe092ab5d4e227313294b"
    processor = BlipProcessor.from_pretrained(blip_model_path, cache_dir=blip_save_path)
    blip_model = BlipForConditionalGeneration.from_pretrained(blip_model_path, cache_dir=blip_save_path)

    # Read and convert image
    bgr_image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)

    # Generate caption
    inputs = processor(pil_image, return_tensors="pt")
    with torch.no_grad():
        blip_output = blip_model.generate(**inputs)
    caption = processor.decode(blip_output[0], skip_special_tokens=True)

    # ===== Step 2: Load Phi-2 (Caption Classification) =====
    phi_save_path = "./models/phi-2"
    phi_model_path = "./models/phi-2/models--microsoft--phi-2/snapshots/ef382358ec9e382308935a992d908de099b64c23"
    tokenizer = AutoTokenizer.from_pretrained(phi_model_path, cache_dir=phi_save_path)
    phi_model = AutoModelForCausalLM.from_pretrained(phi_model_path, cache_dir=phi_save_path)

    # Few-shot prompt
    prompt = f"""You are an emergency advisor. Classify the fire situation and give brief advice.

        Example 1:
        Caption: A person is lighting a matchstick.
        Response: 🔵 Harmless. No action needed.

        Example 2:
        Caption: Flames are rising from a pile of trash on the sidewalk.
        Response: 🔴 Dangerous. Call emergency services immediately.

        Example 3:
        Caption: A campfire is burning at night with people around it.
        Response: 🟡 Controlled. Monitor closely.

        Caption: {caption}
        Response:"""

    # Generate response from Phi-2
    inputs = tokenizer(prompt, return_tensors="pt").to(phi_model.device)
    with torch.no_grad():
        output = phi_model.generate(**inputs, max_new_tokens=50)
        
    response_text = (
        tokenizer.decode(output[0], skip_special_tokens=True)
        .split("Response:")[-1]
        .strip()
        .split("\n")[0]
        .strip('"""')  # Remove accidental quote block
    )

    return caption, response_text

caption, advice = analyze_fire_image("test_image.jpg")
print("BLIP Caption:", caption)
print("Phi-2 Response:", advice)