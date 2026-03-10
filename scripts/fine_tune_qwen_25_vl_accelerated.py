
import torch
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
import pandas as pd
import json
from PIL import Image
from qwen_vl_utils import process_vision_info
import wandb
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# --- Custom Weighted Trainer ---
class WeightedSFTTrainer(SFTTrainer):
    def __init__(self, *args, class_weights=None, tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.tokenizer = tokenizer

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        class_labels_for_weighting = inputs.pop("class_labels_for_weighting")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        labels = inputs.get("labels")
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = loss_fct(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1))
        loss_per_token = loss.view(shift_labels.shape)
        mask = (shift_labels != -100).float()
        per_example_loss = (loss_per_token * mask).sum(dim=1) / mask.sum(dim=1)
        
        class_weights_on_device = self.class_weights.to(per_example_loss.device)
        weights_for_batch = class_weights_on_device[class_labels_for_weighting]
        
        weighted_loss = per_example_loss * weights_for_batch
        final_loss = weighted_loss.mean()
        return (final_loss, outputs) if return_outputs else final_loss

# 1. System and User Prompts (Unchanged)
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

# 2. Data Formatting Function (Unchanged)
def format_data(sample):
    return [
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
        {"role": "user", "content": [{"type": "image", "image": f"file://{sample['image_path']}"}, {"type": "text", "text": user_query}]},
        {"role": "assistant", "content": [{"type": "text", "text": f"```json{json.dumps({'caption': sample['caption'], 'label': sample['label']}, ensure_ascii=False)}```"}]},
    ]

# 3. Load and Prepare Datasets
train_df = pd.read_csv("vlm_finetune/train_labels.csv")
val_df = pd.read_csv("vlm_finetune/val_labels.csv")

print("Calculating class weights for weighted loss...")
class_labels = sorted(train_df['label'].unique())
label_map = {label: i for i, label in enumerate(class_labels)}
train_df['label_id'] = train_df['label'].map(label_map)
class_weights_np = compute_class_weight('balanced', classes=np.unique(train_df['label_id']), y=train_df['label_id'])
class_weights = torch.tensor(class_weights_np, dtype=torch.float)
print(f"Class weights calculated: {class_weights}")

train_dataset = [format_data(sample) for sample in train_df.to_dict('records')]
eval_dataset = [format_data(sample) for sample in val_df.to_dict('records')]

for i, sample in enumerate(train_dataset):
    sample[0]['label_id'] = train_df.iloc[i]['label_id']

# 4. Model and Processor Initialization
model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)

# MODIFIED for Accelerate: remove device_map
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, quantization_config=bnb_config)
processor = Qwen2_5_VLProcessor.from_pretrained(model_id, use_fast=True)

# 5. PEFT Configuration (Unchanged)
for param in model.model.visual.parameters():
    if param.dtype.is_floating_point:
        param.requires_grad = True

peft_config = LoraConfig(lora_alpha=16, lora_dropout=0.05, r=8, bias="none", target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj", "patch_embed.proj", "attn.proj"], task_type="CAUSAL_LM")
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 6. Training Configuration
# MODIFIED: Output directory and batch sizes
training_args = SFTConfig(
    output_dir="/dev/shm/qwen2-5-weighted-finetune",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
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
wandb.init(project="qwen2.5-3b-instruct-trl-sft-REPORT", name="qwen2.5-3b-instruct-ACCELERATED-WEIGHTED", config=training_args)

# 8. Data Collator (Unchanged)
def collate_fn(examples):
    label_ids = [example[0].get('label_id', -1) for example in examples]
    texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]
    image_inputs = [process_vision_info(example)[0] for example in examples]
    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    image_tokens = [151652, 151653, 151655] if isinstance(processor, Qwen2_5_VLProcessor) else [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100
    batch["labels"] = labels
    batch["class_labels_for_weighting"] = torch.tensor(label_ids, dtype=torch.long)
    return batch

# 9. Trainer Initialization and Training
trainer = WeightedSFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn,
    class_weights=class_weights,
    tokenizer=processor.tokenizer,
)

trainer.train()

# 10. Save Model
trainer.save_model(training_args.output_dir)
