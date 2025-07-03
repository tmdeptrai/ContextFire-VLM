from peft import PeftModel
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id)
model = PeftModel.from_pretrained(base_model, "qwen2-7b-instruct-trl-sft-serious-this-time")
model = model.merge_and_unload()
model.save_pretrained("qwen2.5-vl-merged")