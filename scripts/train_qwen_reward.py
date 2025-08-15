#!/usr/bin/env python
# coding: utf-8
"""
Fine-tune Qwen-7B-Chat as a reward (regression) model.
Dataset: headlines_with_scores-2.csv (columns: headline, score)
"""

import os, argparse, torch
torch.backends.cudnn.benchmark = True

from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer
)
from transformers.modeling_outputs import SequenceClassifierOutput
from peft import LoraConfig, get_peft_model
import evaluate
from sklearn.metrics import mean_squared_error

# -------- CLI ----------
parser = argparse.ArgumentParser()
parser.add_argument("--csv_path", type=str,
                    default="output3/headlines_with_scores_1-5.csv")
parser.add_argument("--use_lora", action="store_true")
parser.add_argument("--lora_r", type=int, default=8)
parser.add_argument("--lora_alpha", type=int, default=16)
parser.add_argument("--lora_dropout", type=float, default=0.05)
parser.add_argument("--model_name",
                    default="Qwen/Qwen-7B-Chat")
parser.add_argument("--output_dir",
                    default="qwen7b-reward-lora")
args = parser.parse_args()

# -------- Dataset ----------
ds = load_dataset("csv", data_files=args.csv_path, split="train")
ds = ds.filter(lambda ex: ex["score"] is not None)

# -------- Tokenizer ----------
tokenizer = AutoTokenizer.from_pretrained(
    args.model_name, trust_remote_code=True)

# Fix: Qwen tokenizer doesn't support adding new special tokens
# Explicitly set the pad_token to the eos_token
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = "<|endoftext|>"
tokenizer.padding_side = "right"

def preprocess(example):
    text  = example["headline"]
    score = float(example["score"]) / 5.0
    tok   = tokenizer(text, truncation=True)
    tok["labels"] = score
    return tok

ds = ds.map(preprocess, remove_columns=ds.column_names, num_proc=4)

split_ds = ds.train_test_split(test_size=0.1, shuffle=True, seed=42)
train_dataset, eval_dataset = split_ds["train"], split_ds["test"]

# -------- Backbone ----------
base_model = AutoModelForCausalLM.from_pretrained(
    args.model_name, trust_remote_code=True)

# 若新增了 token，需要扩展词表
base_model.resize_token_embeddings(len(tokenizer))

# 与 gradient checkpointing 兼容
base_model.config.use_cache = False

# -------- Reward wrapper ----------
class RewardModel(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.head     = torch.nn.Linear(backbone.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None, labels=None):
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,    # 关键：拿 hidden_states 而非 last_hidden_state
            return_dict=True
        )
        last_hidden = out.hidden_states[-1][:, 0, :]        # CLS
        logits = self.head(last_hidden).squeeze(-1)
        loss = None if labels is None else \
               torch.nn.functional.mse_loss(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)

model = RewardModel(base_model)

# -------- LoRA (可选) ----------
if args.use_lora:
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="SEQ_CLS",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
else:
    base_model.gradient_checkpointing_enable()

# -------- Metrics ----------
rho_metric = evaluate.load("spearmanr")
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = preds.squeeze()
    mse = mean_squared_error(y_true=labels, y_pred=preds)
    return {
        "mse":      mse,
        "spearman": rho_metric.compute(predictions=preds, references=labels)["spearmanr"],
    }

# -------- Trainer ----------
bs = 2 if args.use_lora else 8
ga = 8 if args.use_lora else 2
lr = 1e-4 if args.use_lora else 5e-5

training_args = TrainingArguments(
    output_dir=args.output_dir,
    learning_rate=lr,
    per_device_train_batch_size=bs,
    per_device_eval_batch_size=bs,
    gradient_accumulation_steps=ga,
    num_train_epochs=3,
    bf16=True,
    dataloader_num_workers=0,
    dataloader_pin_memory=True,
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=500,
    save_steps=500,
    save_total_limit=2,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    report_to="none",
)

# -------- Collate_fn ----------
# Ensure PAD_ID is set correctly even if pad_token_id is None
PAD_ID = tokenizer.pad_token_id

def collate_fn(batch):
    input_ids = [b["input_ids"] for b in batch]
    attention = [b.get("attention_mask", [1]*len(ids))
                 for b, ids in zip(batch, input_ids)]
    labels    = [b["labels"] for b in batch]
    max_len   = max(len(ids) for ids in input_ids)

    padded_ids  = [ids + [PAD_ID]*(max_len-len(ids)) for ids in input_ids]
    padded_attn = [att + [0]*(max_len-len(att))      for att in attention]

    return {
        "input_ids":      torch.tensor(padded_ids,  dtype=torch.long),
        "attention_mask": torch.tensor(padded_attn, dtype=torch.long),
        "labels":         torch.tensor(labels,      dtype=torch.float),
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
)

if __name__ == "__main__":
    trainer.train()
    final_path = os.path.join(args.output_dir, "final")
    
    # Manually save the backbone and the custom head
    if hasattr(model, "backbone"): # For non-LoRA fine-tuning
        model.backbone.save_pretrained(final_path)
        torch.save(model.head.state_dict(), os.path.join(final_path, "reward_head.pth"))
    else: # For LoRA or other PEFT methods
        model.save_pretrained(final_path)

    tokenizer.save_pretrained(final_path)
    print(f"Model + tokenizer saved to {final_path}")
