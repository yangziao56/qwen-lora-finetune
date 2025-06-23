#!/usr/bin/env python
# coding: utf-8
import os
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)

# —— 可修改这两行以切换模型 —— 
#model_name = "Qwen/QwQ-32B"           # 或 "Qwen/Qwen2.5-7B-Instruct"
#model_name = "Qwen/Qwen2.5-7B-Instruct"
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
output_dir = f"checkpoints/{os.path.basename(model_name)}-qlora"

data_files = {"train": "data/train.jsonl", "valid": "data/valid.jsonl"}

# 1. 加载训练 & 验证集
train_ds = load_dataset("json", data_files=data_files["train"])["train"]
val_ds   = load_dataset("json", data_files=data_files["valid"])["train"]

# 2. Tokenizer + 8-bit 模型
bnb_cfg = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    low_cpu_mem_usage=True,
    quantization_config=bnb_cfg,
    trust_remote_code=True
)
model.gradient_checkpointing_enable()

# 3. 应用 LoRA
lora_cfg = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# 4. 数据格式化函数
def preprocess(ex):
    prompt = tokenizer.apply_chat_template(
        ex["messages"], tokenize=False, add_generation_prompt=False
    )
    ids = tokenizer(prompt).input_ids
    ex["input_ids"] = ids
    ex["labels"]    = ids.copy()
    return ex

train_ds = train_ds.map(preprocess, remove_columns=["messages"])
val_ds   = val_ds.map(preprocess,   remove_columns=["messages"])

# 5. 训练参数
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    logging_steps=50,
    #evaluation_strategy="steps",
    eval_steps=500,
    save_steps=500,
    save_total_limit=2,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    deepspeed="ds_zero3_cpuoffload.json",
    report_to="none"
)

# 6. 启动 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds
)

if __name__ == "__main__":
    trainer.train()
    model.save_pretrained(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")
