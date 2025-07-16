#!/usr/bin/env python
# coding: utf-8
"""
deepspeed_lora_finetune.py  ·  scripts/
======================================
- Compatible with full parameter fine-tuning and LoRA fine-tuning
- Enables DeepSpeed (ZeRO-3 + CPU offload) via a command-line argument
- Automatically finds the data/ directory in the project root
- Assumes by default that scripts, .json configuration files, and run scripts are located in the scripts/ directory
"""

import os
import argparse
from typing import List
import json
import datetime
import math                        # ← 新增
import torch
import numpy as np  # 确保已导入
from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from transformers.trainer_utils import EvalPrediction      # 推荐：与源码路径一致
from peft import LoraConfig, get_peft_model



# --------------------------------------------------
# CLI
# --------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser("LoRA/Full fine-tune with DeepSpeed")
    # core
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-7B-Instruct")
    # LoRA opts
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj",
    )
    # deepspeed cfg
    default_ds_cfg = os.path.join(os.path.dirname(__file__), "ds_zero3_offload.json")
    parser.add_argument("--deepspeed_config", type=str, default=default_ds_cfg)
    # misc
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None,
                        help="override default lr: 1e-4 for LoRA, 5e-5 for full")
    parser.add_argument("--grad_acc", type=int, default=None,
                        help="override default gradient_accumulation_steps")
    return parser.parse_args()


# --------------------------------------------------
# helpers
# --------------------------------------------------

def build_tokenizer(model_name):
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def preprocess(tok):
    def _inner(ex):
        prompt = tok.apply_chat_template(ex["messages"], tokenize=False)
        out = tok(prompt, truncation=True, max_length=1024, padding="max_length")
        ex.update({
            "input_ids": out["input_ids"],
            "attention_mask": out["attention_mask"],
            "labels": out["input_ids"].copy(),
        })
        return ex
    return _inner


# --------------------------------------------------
# main
# --------------------------------------------------

def main():
    args = parse_args()

    # Calculate paths
    script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    # Output directory
    mode = "lora" if args.use_lora else "finetune"
    out_dir = os.path.join(project_root, "checkpoints",
                            f"{os.path.basename(args.model_name)}-{mode}-ds")
    os.makedirs(out_dir, exist_ok=True)

    # Data paths (project root data/)
    # data_files = {
    #     "train": os.path.join(project_root, "data", "train.jsonl"),
    #     "valid": os.path.join(project_root, "data", "valid.jsonl"),
    # }

    data_files = {
        "train": os.path.join(project_root, "data", "jsonl", "merged.jsonl"),
        "valid": os.path.join(project_root, "data", "jsonl", "poems_valid.jsonl"),
        # 这里使用同一个文件作为 train 和 valid，实际应用中应分开
    }

    if not all(os.path.exists(p) for p in data_files.values()):
        raise FileNotFoundError(
            f"Missing data files: {data_files['train']} or {data_files['valid']}"
        )

    # tokenizer & model
    tokenizer = build_tokenizer(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, trust_remote_code=True, low_cpu_mem_usage=True
    )

    if args.use_lora:
        # 1. Ensure the model is float32 type to prevent precision issues
        for param in model.parameters():
            param.data = param.data.to(torch.float32)
        
        # 2. Create LoRA configuration
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules.split(","),
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # 3. Apply LoRA
        model = get_peft_model(model, lora_cfg)
        
        # 4. Set parameter trainability: LoRA layers are trainable, others are frozen.
        for name, param in model.named_parameters():
            if 'lora' in name or 'adapter' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # 5. Print trainable parameters
        model.print_trainable_parameters()
        
        # 6. Ensure input embeddings require gradients (needed for gradient checkpointing with PEFT)
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        
        # Enable gradient checkpointing for the model
        model.gradient_checkpointing_enable()
    else:
        model.gradient_checkpointing_enable()

    # Load and preprocess datasets
    ds = load_dataset("json", data_files=data_files)
    train_ds = ds["train"].map(
        preprocess(tokenizer), remove_columns=["messages"], num_proc=8
    )
    val_ds = ds["valid"].map(
        preprocess(tokenizer), remove_columns=["messages"], num_proc=8
    )

    # Hyperparameters
    batch_size = args.batch_size or (1 if args.use_lora else 8)
    grad_acc   = args.grad_acc   or (2 if args.use_lora else 4)
    lr         = args.learning_rate or (1e-4 if args.use_lora else 5e-5)

    # 定义评估函数
    def compute_metrics(eval_pred: EvalPrediction):
        logits, labels = eval_pred.predictions, eval_pred.label_ids

        if isinstance(logits, tuple):
            logits = logits[0]

        shift_logits = logits[..., :-1, :].reshape(-1, logits.shape[-1])
        shift_labels = labels[..., 1:].reshape(-1)

        # 兼容 numpy 和 torch
        if isinstance(shift_logits, np.ndarray):
            shift_logits = torch.from_numpy(shift_logits)
        if isinstance(shift_labels, np.ndarray):
            shift_labels = torch.from_numpy(shift_labels)

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits, shift_labels)
        ppl = math.exp(loss.item())
        return {"eval_loss": loss.item(), "perplexity": ppl}

    # 更新 TrainingArguments：去掉不认识的 evaluation_* 参数
    training_args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_acc,
        num_train_epochs=args.epochs,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        fp16=True,
        deepspeed=args.deepspeed_config,
        report_to="none",
        # --- 使用旧版 transformers API ---
        # 'evaluation_strategy' -> 'evaluate_during_training'
        # evaluate_during_training=True,
        # eval_steps=500,
        # load_best_model_at_end=True,
        # 以下参数在旧版中不存在，需要移除。
        # 移除后，load_best_model_at_end 会默认使用 eval_loss 作为指标。
        # 警告：移除 eval_accumulation_steps 可能会导致评估时再次显存溢出。
        # metric_for_best_model="perplexity",
        # greater_is_better=False,
        # eval_accumulation_steps=4,

        # evaluation_strategy="steps",   # 每 500 step 评估一次
        # eval_steps=500,
        # load_best_model_at_end=True,   # 仍用 eval_loss 选最优
        # metric_for_best_model="eval_loss",
        # greater_is_better=False,
        eval_accumulation_steps=1,  # <--- 新增

    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        compute_metrics=compute_metrics,
    )

    # 开始训练
    trainer.train()

    # 等待所有进程 train 完
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    # 1) 所有进程都跑 evaluate()
    #eval_metrics = trainer.evaluate()

    # 2) 只有 rank0 打印 & 保存
    if trainer.is_world_process_zero():
        #print(f"✔ Final evaluation on validation set: {eval_metrics}")

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        final_path = os.path.join(out_dir, f"final-{timestamp}")
        trainer.save_model(final_path)
        tokenizer.save_pretrained(final_path)

        config_path = os.path.join(final_path, "training_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, ensure_ascii=False)

        print(f"✔ Saved model + tokenizer + config to {final_path}")


if __name__ == "__main__":
    main()
