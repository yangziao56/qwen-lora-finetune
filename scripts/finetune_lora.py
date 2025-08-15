#!/usr/bin/env python
# coding: utf-8

import os, torch, argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

# ---------- CLI ----------
parser = argparse.ArgumentParser(description="Fine-tune Qwen with/without LoRA")
parser.add_argument("--use_lora", action="store_true")
parser.add_argument("--lora_r", type=int, default=8)
parser.add_argument("--lora_alpha", type=int, default=16)
parser.add_argument("--lora_dropout", type=float, default=0.05)
parser.add_argument("--lora_target_modules", type=str,
                    default="q_proj,k_proj,v_proj,o_proj")
parser.add_argument("--model_name", type=str,
                    default="Qwen/QwQ-32B",
                    help="e.g. Qwen/Qwen2-7B-Instruct, Qwen/QwQ-32B")
args = parser.parse_args()

model_name = args.model_name
training_mode = "lora" if args.use_lora else "finetune"
output_dir = f"checkpoints/{os.path.basename(model_name)}-{training_mode}-singlegpu"
os.makedirs(output_dir, exist_ok=True)

data_files = {"train": "data/train.jsonl", "valid": "data/valid.jsonl"}
if not all(os.path.exists(p) for p in data_files.values()):
    raise FileNotFoundError("缺少 train.jsonl 或 valid.jsonl")

# ---------- Dataset ----------
all_ds   = load_dataset("json", data_files=data_files)
train_ds = all_ds["train"]
val_ds   = all_ds["valid"]
print(f"Train: {len(train_ds)}  |  Valid: {len(val_ds)}")

# ---------- Tokenizer ----------
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token_id is None:                 # 关键修补
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ---------- Model ----------
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

# ---------- LoRA ----------
if args.use_lora:
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules.split(","),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
else:
    model.gradient_checkpointing_enable()
    model.config.use_cache = False                 # 与 checkpointing 兼容
    print(f"Total params: {sum(p.numel() for p in model.parameters())}")
    print(f"Trainable   : {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# ---------- Preprocess ----------
def preprocess(ex):
    prompt = tokenizer.apply_chat_template(
        ex["messages"], tokenize=False, add_generation_prompt=False
    )
    toks = tokenizer(prompt, truncation=True, max_length=1024,
                     padding="max_length")
    ex["input_ids"]     = toks["input_ids"]
    ex["attention_mask"]= toks["attention_mask"]
    ex["labels"]        = toks["input_ids"].copy()
    return ex

train_ds = train_ds.map(preprocess, remove_columns=["messages"], num_proc=8)
val_ds   = val_ds.map(preprocess,   remove_columns=["messages"], num_proc=8)

# ---------- Hyper-params ----------
if args.use_lora:
    lr, bs, ga = 1e-4, 1, 2
else:
    lr, bs, ga = 5e-5, 8, 4

# ---------- TrainingArguments ----------
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=bs,
    per_device_eval_batch_size=bs,
    gradient_accumulation_steps=ga,
    num_train_epochs=3,
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    learning_rate=lr,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    fp16=False,
    report_to="none",
    eval_strategy="steps",        # 按要求保留原写法
    eval_steps=500,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=data_collator,
)

if __name__ == "__main__":
    trainer.train()
    final_path = os.path.join(output_dir, "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Model + tokenizer saved to {final_path}")
