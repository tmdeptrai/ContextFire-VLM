import torch
import time
import cv2
import numpy as np
from PIL import Image
import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    AutoTokenizer, AutoModelForCausalLM
)

app = Flask(__name__)
CORS(app)

# Constants
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MODEL_TIMEOUT = 60  # seconds

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ===== Load Models Once on Startup =====
print("[INFO] Loading models...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    blip_save_path = "./models/blip"
    blip_model_path = "./models/blip/models--Salesforce--blip-image-captioning-base/snapshots/82a37760796d32b1411fe092ab5d4e227313294b"
    blip_processor = BlipProcessor.from_pretrained(blip_model_path, cache_dir=blip_save_path)
    blip_model = BlipForConditionalGeneration.from_pretrained(blip_model_path, cache_dir=blip_save_path).to(device)

    phi_save_path = "./models/phi-2"
    phi_model_path = "./models/phi-2/models--microsoft--phi-2/snapshots/ef382358ec9e382308935a992d908de099b64c23"
    phi_tokenizer = AutoTokenizer.from_pretrained(phi_model_path, cache_dir=phi_save_path)
    phi_model = AutoModelForCausalLM.from_pretrained(phi_model_path, cache_dir=phi_save_path).to(device)
    print(f"[INFO] Models loaded. Using device: {device}")
except Exception as e:
    print(f"[ERROR] Failed to load models: {str(e)}")
    raise

@app.route("/analyze", methods=["POST"])
def analyze_fire_image():
    try:
        # Validate file exists in request
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        
        # Validate filename and file type
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type"}), 400
            
        # Validate file size
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        if file_size > MAX_FILE_SIZE:
            return jsonify({"error": "File too large"}), 400

        # Process image
        contents = file.read()
        np_array = np.frombuffer(contents, np.uint8)
        bgr_image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        if bgr_image is None:
            return jsonify({"error": "Invalid image data"}), 400
            
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        times = {}
        t0 = time.time()

        # Step 1: BLIP captioning with timeout
        try:
            torch.cuda.empty_cache()  # Clear GPU memory
            inputs = blip_processor(pil_image, return_tensors="pt").to(device)
            with torch.no_grad():
                blip_output = blip_model.generate(**inputs, max_length=50, num_return_sequences=1, timeout=MODEL_TIMEOUT)
            caption = blip_processor.decode(blip_output[0], skip_special_tokens=True)
            times['blip'] = round(time.time() - t0, 3)
        except Exception as e:
            torch.cuda.empty_cache()
            return jsonify({"error": f"BLIP model error: {str(e)}"}), 500

        # Step 2: Phi-2 classification with timeout
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

        try:
            t1 = time.time()
            torch.cuda.empty_cache()  # Clear GPU memory
            phi_inputs = phi_tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                output = phi_model.generate(**phi_inputs, max_new_tokens=50, timeout=MODEL_TIMEOUT)
            response = phi_tokenizer.decode(output[0], skip_special_tokens=True)
            response_text = response.split("Response:")[-1].strip().split("\n")[0].strip('"')
            times['phi'] = round(time.time() - t1, 3)
        except Exception as e:
            torch.cuda.empty_cache()
            return jsonify({"error": f"Phi model error: {str(e)}"}), 500

        # Clear GPU memory after processing
        torch.cuda.empty_cache()

        return jsonify({
            "caption": caption,
            "advice": response_text,
            "times": times,
            "device": str(device)
        })

    except Exception as e:
        torch.cuda.empty_cache()
        return jsonify({"error": str(e)}), 500

@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
