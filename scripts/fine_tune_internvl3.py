import gc
import time
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
import pandas as pd
import json
from PIL import Image
import wandb

# 1. Memory Clearing Function
def clear_memory():
    if "model" in globals(): del globals()["model"]
    if "processor" in globals(): del globals()["processor"]
    if "trainer" in globals(): del globals()["trainer"]
    if "peft_model" in globals(): del globals()["peft_model"]
    time.sleep(2)
    gc.collect()
    time.sleep(1)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(1)
    gc.collect()
    time.sleep(1)

# 2. System and User Prompts
system_message = """You are a Vision Language Model specialized in detecting clues of fire, smoke and surrounding context then classify them as no fire, dangerous fire or controlled fire.
Your task is to analyze the provided image and respond to queries with concise answers, usually a json format of a caption and a label.
Summarize what you see in the image. Describe the environment, key objects, people, and any signs of fire or smoke.
Based on your summary, classify the fire situation: no fire(e.g., fire alarm, fire distinguisher,..), controlled fire (e.g., fireplace emitting, campfire, cooking, candles, match stick, lighter..) or a dangerous/uncontrolled fire (e.g., curtains on fire, smoke on ceiling, couch on fire, bed sheet on fire, spreading fire on furniture..)
Focus on delivering accurate, succinct caption and precise label based on the visual information. Add a brief explanation for your choice of label in the caption if necessary."""

user_query = """Summarize this situation in the image, look for signs of fire and smoke and classify whether the situation is 
no fire(e.g., fire alarm, fire distinguisher,..), 
controlled fire (e.g., fireplace emitting, campfire, cooking, candles, match stick, lighter..) 
or a dangerous/uncontrolled fire (e.g., curtains on fire, smoke on ceiling, couch on fire, bed sheet on fire, spreading fire on furniture..)
Add a brief explanation for your choice of label in the caption if necessary.
Respond only this json format:
{ "caption": "...", "label": "no fire"|"controlled fire"|"dangerous fire" }
"""

# 3. Data Formatting Function
def format_data(sample):
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "path": f"{sample['image_path']}"},
                {"type": "text", "text": "<image>" + user_query},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": f"```json{json.dumps({'caption': sample['caption'], 'label': sample['label']}, ensure_ascii=False)}```"}
            ],
        },
    ]

# 4. Load and Prepare Datasets
train_df = pd.read_csv("vlm_finetune/train_labels.csv")
val_df = pd.read_csv("vlm_finetune/val_labels.csv")

train_dataset = [format_data(sample) for sample in train_df.to_dict('records')]
eval_dataset = [format_data(sample) for sample in val_df.to_dict('records')]

# 5. Model and Processor Initialization
clear_memory()
torch_device = "cuda"
model_checkpoint = "OpenGVLab/InternVL3-2B-hf"
processor = AutoProcessor.from_pretrained(model_checkpoint)
model = AutoModelForImageTextToText.from_pretrained(model_checkpoint, device_map=torch_device, torch_dtype=torch.bfloat16)

# 6. PEFT Configuration
for name, param in model.named_parameters():
    if ("vision_tower" in name or "multi_modal_projector" in name) and param.dtype.is_floating_point:
        param.requires_grad = True

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=8,
    bias="lora_only",
    target_modules=["q_proj", "k_proj", "v_proj", "lora", "up_proj", "down_proj", "gate_proj", "fc1", "fc2", "projection_layer", "linear_1", "linear_2"],
    task_type="CAUSAL_LM",
)
peft_model = get_peft_model(model, peft_config)
peft_model.print_trainable_parameters()

# 7. Training Configuration
training_args = SFTConfig(
    output_dir="internvl3-2B-finetune",
    num_train_epochs=2,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    logging_steps=15,
    eval_steps=15,
    eval_strategy="steps",
    save_strategy="steps",
    save_steps=15,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    load_best_model_at_end=True,
    bf16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.1,
    push_to_hub=True,
    report_to="wandb",
    seed=42,
    gradient_checkpointing_kwargs={"use_reentrant": True},
    dataset_kwargs={"skip_prepare_dataset": True},
)
training_args.remove_unused_columns = False

# 8. W&B Initialization
wandb.init(
    project="internvl3-2B-finetune",
    name="internvl3-2B-finetune",
    config=training_args,
)

# 9. Data Collator
def internvl3_collate_fn(examples):
    texts = [processor.apply_chat_template(sample, tokenize=False) for sample in examples]
    images = []
    for sample in examples:
        for msg in sample:
            for c in msg["content"]:
                if c["type"] == "image":
                    img = Image.open(c["path"]).convert("RGB")
                    images.append(img)
                    break
    
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    labels[labels == image_token_id] = -100
    batch["labels"] = labels
    return batch

# 10. Trainer Initialization and Training
trainer = SFTTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=internvl3_collate_fn,
    peft_config=peft_config,
    tokenizer=processor.tokenizer,
)

trainer.train()

# 11. Save Model
trainer.save_model(training_args.output_dir)
