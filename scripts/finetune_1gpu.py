#!/usr/bin/env python
# coding: utf-8

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

# 保持模型名称不变
model_name = "Qwen/Qwen2-0.5B-Instruct" 
# model_name = "Qwen/QwQ-32B"  # 使用更大的模型进行全参数微调
# model_name = "Qwen/Qwen2-7B-Instruct"  # 使用更大的模型进行全参数微调

output_dir = f"checkpoints/{os.path.basename(model_name)}-finetune-singlegpu"

# 确保检查点目录存在
os.makedirs(output_dir, exist_ok=True)

data_files = {"train": "data/train.jsonl", "valid": "data/valid.jsonl"}

# 1. 加载训练 & 验证集
# 检查数据文件是否存在
if not os.path.exists(data_files["train"]) or not os.path.exists(data_files["valid"]):
    raise FileNotFoundError(
        f"Training or validation file not found. "
        f"Please make sure {data_files['train']} and {data_files['valid']} exist."
    )

all_datasets = load_dataset("json", data_files=data_files)
train_ds = all_datasets["train"]
val_ds   = all_datasets["valid"]

# 2. 直接加载模型和 tokenizer（移除量化配置）
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# 指定 pad_token 以支持动态 padding
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # 对于因果语言模型，填充在右侧

# 直接加载模型进行全参数微调
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
model.gradient_checkpointing_enable()
print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# 数据格式化函数保持不变
def preprocess(ex):
    prompt = tokenizer.apply_chat_template(
        ex["messages"], tokenize=False, add_generation_prompt=False
    )
    tokenized_output = tokenizer(
        prompt,
        truncation=True,
        max_length=1024,
        padding="max_length"  # 确保所有序列填充到统一长度
    )
    ex["input_ids"] = tokenized_output["input_ids"]
    ex["attention_mask"] = tokenized_output["attention_mask"]
    ex["labels"] = tokenized_output["input_ids"].copy()
    return ex

# 数据处理
train_ds = train_ds.map(preprocess, remove_columns=["messages"], num_proc=8) 
val_ds   = val_ds.map(preprocess,   remove_columns=["messages"], num_proc=8)

# 训练参数
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=8,  # 可以适当增大，因为不需要存储LoRA参数
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,  # 可以减小，因为批次大小增加了
    num_train_epochs=3,
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    learning_rate=5e-5,  # 全参数微调通常使用更小的学习率
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,  # 使用 bf16 精度可以加速训练
    fp16=False,
    report_to="none"
)

# 使用标准数据收集器
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False  # 这是因果语言模型，不是掩码语言模型
)

# 启动 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=data_collator,
)

if __name__ == "__main__":
    # 开始训练
    trainer.train()
    
    # 保存最终的模型和 tokenizer
    final_save_path = os.path.join(output_dir, "final")
    model.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    print(f"Final model saved to {final_save_path}")
