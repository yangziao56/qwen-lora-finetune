#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import random

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import DPOTrainer
from typing import Dict, List, Optional


def build_preference_dataset(
    csv_path: str,
    min_score_gap: float,
    max_samples: Optional[int],
    seed: int,
) -> Dataset:
    frame = pd.read_csv(csv_path)
    needed_cols = {"brand", "persona", "headline", "overall_score"}
    missing = needed_cols.difference(frame.columns)
    if missing:
        raise KeyError(f"CSV 缺少列: {', '.join(sorted(missing))}")

    frame = frame.dropna(subset=["brand", "persona", "headline", "overall_score"])
    records: List[Dict[str, str]] = []
    rng = random.Random(seed)

    for (brand, persona), group in frame.groupby(["brand", "persona"]):
        group = group.sort_values("overall_score", ascending=False)
        if len(group) < 2:
            continue

        best_row = group.iloc[0]
        for idx in range(1, len(group)):
            cand = group.iloc[idx]
            gap = best_row["overall_score"] - cand["overall_score"]
            if gap < min_score_gap:
                continue
            prompt = f"Persona: {persona}\nBrand: {brand}\nSlogan:"
            records.append(
                {
                    "prompt": prompt,
                    "chosen": str(best_row["headline"]).strip(),
                    "rejected": str(cand["headline"]).strip(),
                }
            )

    if not records:
        raise ValueError(
            "未构建到任何偏好对。尝试降低 --min_score_gap 或检查 CSV 内容。"
        )

    rng.shuffle(records)
    if max_samples is not None and max_samples > 0:
        records = records[:max_samples]
    return Dataset.from_list(records)


def build_model_and_tokenizer(
    model_name: str,
    use_lora: bool,
    lora_args: argparse.Namespace,
    dtype: torch.dtype,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
    )

    if use_lora:
        lora_cfg = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            lora_dropout=lora_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=lora_args.lora_target_modules.split(","),
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
    else:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    ref_model.requires_grad_(False)
    ref_model.eval()
    return model, ref_model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline DPO training for slogans.")
    parser.add_argument(
        "--csv_path",
        type=str,
        default="RL/batch_evaluation_results.csv",
        help="包含 overall_score 的打分 CSV。",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/QwQ-32B",
        help="Qwen 模型 ID 或本地路径。",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj",
    )
    parser.add_argument("--min_score_gap", type=float, default=0.25)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--eval_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max_prompt_length", type=int, default=128)
    parser.add_argument("--max_response_length", type=int, default=64)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = build_preference_dataset(
        csv_path=args.csv_path,
        min_score_gap=args.min_score_gap,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    if args.eval_split > 0.0:
        split = dataset.train_test_split(test_size=args.eval_split, seed=args.seed)
        train_ds = split["train"]
        eval_ds = split["test"]
    else:
        train_ds = dataset
        eval_ds = None

    model, ref_model, tokenizer = build_model_and_tokenizer(
        args.model_name,
        args.use_lora,
        args,
        torch.bfloat16
        if args.bf16
        else torch.float16
        if args.fp16
        else torch.float32,
    )

    output_dir = (
        args.output_dir
        if args.output_dir
        else os.path.join(
            "checkpoints",
            f"{os.path.basename(args.model_name)}-dpo{'-lora' if args.use_lora else ''}",
        )
    )
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        evaluation_strategy="steps" if eval_ds else "no",
        eval_steps=args.eval_steps,
        report_to="none",
        bf16=args.bf16,
        fp16=args.fp16 and not args.bf16,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        beta=args.beta,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        max_prompt_length=args.max_prompt_length,
        max_length=args.max_response_length + args.max_prompt_length,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"DPO model saved to {output_dir}")


if __name__ == "__main__":
    main()
