
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model , prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer
import pandas as pd
import json
from PIL import Image
from qwen_vl_utils import process_vision_info
import wandb

# 1. System and User Prompts
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

# 2. Data Formatting Function
def format_data(sample):
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"file://{sample['image_path']}",
                },
                {
                    "type": "text",
                    "text": user_query,
                },
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": f"```json{json.dumps({'caption': sample['caption'], 'label': sample['label']}, ensure_ascii=False)}```"
                }
            ],
        },
    ]

# 3. Load and Prepare Datasets
train_df = pd.read_csv("vlm_finetune/train_labels.csv")
val_df = pd.read_csv("vlm_finetune/val_labels.csv")

train_dataset = [format_data(sample) for sample in train_df.to_dict('records')]
eval_dataset = [format_data(sample) for sample in val_df.to_dict('records')]

# 4. Model and Processor Initialization
model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config
)

processor = Qwen2_5_VLProcessor.from_pretrained(model_id, use_fast=True)

# 5. PEFT Configuration
for param in model.visual.parameters():
    param.requires_grad = True

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=8,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "up_proj", "down_proj", "gate_proj",
                    "patch_embed.proj",
                    "attn.proj"], 
    task_type="CAUSAL_LM",
)

peft_model = get_peft_model(model, peft_config)
peft_model.print_trainable_parameters()

# 6. Training Configuration
training_args = SFTConfig(
    output_dir="qwen2-7b-instruct-trl-sft-VISION-ENCODER",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    learning_rate=2e-4,
    lr_scheduler_type="constant",
    logging_steps=30,
    eval_steps=30,
    eval_strategy="steps",
    save_strategy="steps",
    save_steps=30,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    load_best_model_at_end=True,
    bf16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    push_to_hub=True,
    report_to="wandb",
    seed=42,
    gradient_checkpointing_kwargs={"use_reentrant": True},
    dataset_text_field="",
    dataset_kwargs={"skip_prepare_dataset": True},
)
training_args.remove_unused_columns = False

# 7. W&B Initialization
wandb.init(
    project="qwen2.5-3b-instruct-trl-sft-REPORT",
    name="qwen2.5-3b-instruct-trl-sft-UNFREEZE",
    config=training_args,
)

# 8. Data Collator
def collate_fn(examples):
    texts = [
        processor.apply_chat_template(example, tokenize=False) for example in examples
    ]
    image_inputs = [process_vision_info(example)[0] for example in examples]

    batch = processor(
        text=texts, images=image_inputs, return_tensors="pt", padding=True
    )

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    if isinstance(processor, Qwen2_5_VLProcessor):
        image_tokens = [151652, 151653, 151655]
    else:
        image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]

    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100

    batch["labels"] = labels
    return batch

# 9. Trainer Initialization and Training
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn,
    peft_config=peft_config,
    tokenizer=processor.tokenizer,
)

trainer.train()

# 10. Save Model
trainer.save_model(training_args.output_dir)
