#!/usr/bin/env python
# coding: utf-8

import os
import torch
import argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

# Command line arguments for controlling LoRA options
parser = argparse.ArgumentParser(description="Fine-tune Qwen model with or without LoRA")
parser.add_argument("--use_lora", action="store_true", help="Use LoRA for fine-tuning")
parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank parameter")
parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha parameter")
parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate")
parser.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj", 
                   help="Comma-separated list of target modules for LoRA")
parser.add_argument("--model_name", type=str, default="Qwen/QwQ-32B",
                    help="HuggingFace, Qwen/Qwen2-7B-Instruct, Qwen/QwQ-32B, Qwen/Qwen2-0.5B-Instruct")

args = parser.parse_args()

# Keep the model name unchanged
# model_name = "Qwen/Qwen2-0.5B-Instruct" 
# model_name = "Qwen/QwQ-32B"  # Use a larger model for full-parameter fine-tuning
# model_name = "Qwen/Qwen2-7B-Instruct"  # Use a larger model for full-parameter fine-tuning
model_name = args.model_name


# Set output directory, distinguish by whether LoRA is used
training_mode = "lora" if args.use_lora else "finetune"
output_dir = f"checkpoints/{os.path.basename(model_name)}-{training_mode}-singlegpu"

# Ensure checkpoint directory exists
os.makedirs(output_dir, exist_ok=True)

data_files = {"train": "data/train.jsonl", "valid": "data/valid.jsonl"}

# 1. Load train & validation sets
# Check if data files exist
if not os.path.exists(data_files["train"]) or not os.path.exists(data_files["valid"]):
    raise FileNotFoundError(
        f"Training or validation file not found. "
        f"Please make sure {data_files['train']} and {data_files['valid']} exist."
    )

all_datasets = load_dataset("json", data_files=data_files)
train_ds = all_datasets["train"]
val_ds   = all_datasets["valid"]

print(f"Number of training samples: {len(train_ds)}")
print(f"Number of validation samples: {len(val_ds)}")

# 2. Directly load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# Specify pad_token to support dynamic padding
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # For causal LM, pad on the right

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

# If using LoRA, configure LoRA model
if args.use_lora:
    # Convert target modules string to list
    target_modules = args.lora_target_modules.split(",")
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
else:
    # In non-LoRA mode, enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# Data formatting function remains unchanged
def preprocess(ex):
    prompt = tokenizer.apply_chat_template(
        ex["messages"], tokenize=False, add_generation_prompt=False
    )
    tokenized_output = tokenizer(
        prompt,
        truncation=True,
        max_length=1024,
        padding="max_length"  # Ensure all sequences are padded to the same length
    )
    ex["input_ids"] = tokenized_output["input_ids"]
    ex["attention_mask"] = tokenized_output["attention_mask"]
    ex["labels"] = tokenized_output["input_ids"].copy()
    return ex

# Data processing
train_ds = train_ds.map(preprocess, remove_columns=["messages"], num_proc=8) 
val_ds   = val_ds.map(preprocess,   remove_columns=["messages"], num_proc=8)

# Adjust training parameters based on LoRA mode
if args.use_lora:
    # LoRA allows higher learning rate and larger batch size
    learning_rate = 1e-4
    batch_size = 1
    gradient_accumulation_steps = 2
else:
    # Full-parameter fine-tuning uses smaller learning rate
    learning_rate = 5e-5
    batch_size = 8
    gradient_accumulation_steps = 4

# Training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_train_epochs=3,
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    learning_rate=learning_rate,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    fp16=False,
    report_to="none",
    eval_strategy="steps",
    eval_steps=500,
)

# Use standard data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False
)

# Start Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=data_collator,
)

if __name__ == "__main__":
    # Start training
    trainer.train()
    
    # Save final model and tokenizer
    final_save_path = os.path.join(output_dir, "final")
    model.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    print(f"Final model saved to {final_save_path}")
