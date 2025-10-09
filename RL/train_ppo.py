#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp

    _XLA_AVAILABLE = True
except ImportError:
    xm = None  # type: ignore[assignment]
    xmp = None  # type: ignore[assignment]
    _XLA_AVAILABLE = False


@dataclass
class SloganSample:
    prompt_ids: List[int]
    response_ids: List[int]
    reward: float


class TokenizedSloganDataset(Dataset):
    def __init__(self, items: List[SloganSample]):
        self._items = items

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Dict[str, List[int] | float]:
        sample = self._items[idx]
        return {
            "prompt_ids": sample.prompt_ids,
            "response_ids": sample.response_ids,
            "reward": sample.reward,
        }


def build_collate_fn(pad_token_id: int):
    def _collate(batch: List[Dict[str, List[int] | float]]) -> Dict[str, torch.Tensor]:
        prompt_lengths = torch.tensor(
            [len(item["prompt_ids"]) for item in batch],
            dtype=torch.long,
        )
        response_lengths = torch.tensor(
            [len(item["response_ids"]) for item in batch],
            dtype=torch.long,
        )
        max_prompt = prompt_lengths.max().item()
        max_response = response_lengths.max().item()
        batch_size = len(batch)

        prompt_tensor = torch.full(
            (batch_size, max_prompt),
            pad_token_id,
            dtype=torch.long,
        )
        response_tensor = torch.full(
            (batch_size, max_response),
            pad_token_id,
            dtype=torch.long,
        )

        for i, item in enumerate(batch):
            prompt_tensor[i, : prompt_lengths[i]] = torch.tensor(
                item["prompt_ids"], dtype=torch.long
            )
            response_tensor[i, : response_lengths[i]] = torch.tensor(
                item["response_ids"], dtype=torch.long
            )

        rewards = torch.tensor([float(item["reward"]) for item in batch], dtype=torch.float32)
        return {
            "prompt_ids": prompt_tensor,
            "response_ids": response_tensor,
            "prompt_lengths": prompt_lengths,
            "response_lengths": response_lengths,
            "rewards": rewards,
        }

    return _collate


def load_and_filter_records(
    csv_path: str,
    min_score: Optional[float],
    max_samples: Optional[int],
    seed: int,
) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    needed_cols = {"brand", "persona", "headline", "overall_score"}
    missing = needed_cols.difference(frame.columns)
    if missing:
        raise KeyError(f"CSV 缺少列: {', '.join(sorted(missing))}")

    frame = frame.dropna(subset=list(needed_cols))
    if min_score is not None:
        frame = frame[frame["overall_score"] >= min_score]
    if frame.empty:
        raise ValueError("筛选后的样本为空，检查 --min_score 或 CSV 数据。")

    frame = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if max_samples is not None and max_samples > 0:
        frame = frame.head(max_samples)
    return frame


def tokenize_records(
    frame: pd.DataFrame,
    tokenizer: AutoTokenizer,
    max_prompt_length: int,
    max_response_length: int,
    reward_normalizer: float,
) -> List[SloganSample]:
    items: List[SloganSample] = []
    for row in frame.itertuples(index=False):
        prompt_text = f"Persona: {row.persona}\nBrand: {row.brand}\nSlogan:"
        prompt_ids = tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_prompt_length,
        )["input_ids"]

        response_ids = tokenizer(
            str(row.headline).strip(),
            add_special_tokens=False,
            truncation=True,
            max_length=max_response_length,
        )["input_ids"]

        if not prompt_ids or not response_ids:
            continue

        reward = float(row.overall_score) / reward_normalizer
        items.append(
            SloganSample(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                reward=reward,
            )
        )
    if not items:
        raise ValueError("没有有效样本：检查 tokenization 长度或 CSV 内容。")
    return items


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
        ref_model = None
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


def compute_sequence_logprobs(
    model: AutoModelForCausalLM,
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
    prompt_lengths: torch.Tensor,
    response_lengths: torch.Tensor,
    pad_token_id: int,
) -> torch.Tensor:
    device = prompt_ids.device
    batch_size = prompt_ids.size(0)
    total_length = prompt_ids.size(1) + response_ids.size(1)

    input_ids = torch.full(
        (batch_size, total_length),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids, dtype=torch.long)

    for i in range(batch_size):
        p_len = prompt_lengths[i].item()
        r_len = response_lengths[i].item()
        total = p_len + r_len
        input_ids[i, :p_len] = prompt_ids[i, :p_len]
        input_ids[i, p_len:total] = response_ids[i, :r_len]
        attention_mask[i, :total] = 1

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    logits = outputs.logits.float()
    log_probs = torch.log_softmax(logits, dim=-1)

    seq_log_probs: List[torch.Tensor] = []
    for i in range(batch_size):
        p_len = max(1, prompt_lengths[i].item())
        r_len = response_lengths[i].item()
        if r_len == 0:
            seq_log_probs.append(torch.tensor(0.0, device=device))
            continue
        positions = torch.arange(r_len, device=device) + (p_len - 1)
        tokens = response_ids[i, :r_len]
        seq_log_probs.append(log_probs[i, positions, tokens].sum())

    return torch.stack(seq_log_probs)


def _tpu_env_available() -> bool:
    if not _XLA_AVAILABLE:
        return False
    pjrt = os.environ.get("PJRT_DEVICE", "").upper()
    if pjrt == "TPU":
        return True
    return any(
        key in os.environ
        for key in (
            "TPU_NAME",
            "TPU_WORKER_HOSTS",
            "CLOUD_TPU_TASK_ID",
            "TPU_VISIBLE_CHIPS",
        )
    )


def _resolve_device_choice(choice: str) -> str:
    choice = choice.lower()
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用，请改用 --device cpu 或 --device tpu。")
        return "cuda"
    if choice == "tpu":
        if not _tpu_env_available():
            raise RuntimeError("未检测到 TPU 环境，无法使用 --device tpu。")
        return "tpu"
    if choice == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if _tpu_env_available():
            return "tpu"
        return "cpu"
    return "cpu"


def _resolve_tpu_cores(raw_value: str) -> int:
    value = raw_value.strip().lower()
    default_cores = int(os.environ.get("TPU_NUM_DEVICES", "8"))
    if value in {"all", "-1"}:
        return default_cores
    try:
        cores = int(value)
    except ValueError as exc:
        raise ValueError("--tpu_cores 需为整数或 'all'") from exc
    if cores < 1:
        raise ValueError("--tpu_cores 需 >= 1")
    return cores


def _move_batch_to_device(
    batch: Dict[str, torch.Tensor],
    device,
    device_type: str,
) -> Dict[str, torch.Tensor]:
    moved: Dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if not isinstance(value, torch.Tensor):
            if key == "rewards":
                value = torch.tensor(value, dtype=torch.float32)
            else:
                moved[key] = value
                continue
        if device_type == "tpu":
            moved[key] = value.to(device)
        else:
            moved[key] = value.to(device, non_blocking=True)
    return moved


def _train_one_process(
    args: argparse.Namespace,
    device,
    rank: int,
    world_size: int,
) -> None:
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    dtype = (
        torch.bfloat16
        if args.bf16
        else torch.float16
        if args.fp16
        else torch.float32
    )

    frame = load_and_filter_records(
        csv_path=args.csv_path,
        min_score=args.min_score,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    model, ref_model, tokenizer = build_model_and_tokenizer(
        args.model_name,
        args.use_lora,
        args,
        dtype,
    )

    items = tokenize_records(
        frame,
        tokenizer,
        args.max_prompt_length,
        args.max_response_length,
        reward_normalizer=args.reward_normalizer,
    )

    reward_tensor = torch.tensor([item.reward for item in items])
    reward_mean = reward_tensor.mean().item()
    reward_std = reward_tensor.std(unbiased=False).item()
    reward_std = reward_std if reward_std > 1e-6 else 1.0

    dataset = TokenizedSloganDataset(items)
    collate_fn = build_collate_fn(tokenizer.pad_token_id)
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
        if world_size > 1
        else None
    )
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        collate_fn=collate_fn,
    )

    model.to(device)
    if ref_model is not None:
        ref_model.to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
    )
    optimizer.zero_grad(set_to_none=True)

    total_updates = 0
    accum_counter = 0
    for epoch in range(args.num_train_epochs):
        model.train()
        if sampler is not None:
            sampler.set_epoch(epoch)

        progress = (
            tqdm(
                data_loader,
                desc=f"Epoch {epoch + 1}/{args.num_train_epochs}",
                leave=False,
            )
            if rank == 0
            else None
        )
        iterator = progress if progress is not None else data_loader

        for step, batch in enumerate(iterator, 1):
            batch = _move_batch_to_device(batch, device, args.device_type)
            prompt_ids = batch["prompt_ids"]
            response_ids = batch["response_ids"]
            prompt_lengths = batch["prompt_lengths"]
            response_lengths = batch["response_lengths"]
            rewards = batch["rewards"] * args.reward_scale

            model.eval()
            with torch.no_grad():
                old_logprob = compute_sequence_logprobs(
                    model,
                    prompt_ids,
                    response_ids,
                    prompt_lengths,
                    response_lengths,
                    tokenizer.pad_token_id,
                )
                if ref_model is not None:
                    ref_logprob = compute_sequence_logprobs(
                        ref_model,
                        prompt_ids,
                        response_ids,
                        prompt_lengths,
                        response_lengths,
                        tokenizer.pad_token_id,
                    )
                elif isinstance(model, PeftModel):
                    with model.disable_adapter():
                        ref_logprob = compute_sequence_logprobs(
                            model,
                            prompt_ids,
                            response_ids,
                            prompt_lengths,
                            response_lengths,
                            tokenizer.pad_token_id,
                        )
                else:
                    ref_logprob = old_logprob.detach().clone()
            model.train()
            logprob = compute_sequence_logprobs(
                model,
                prompt_ids,
                response_ids,
                prompt_lengths,
                response_lengths,
                tokenizer.pad_token_id,
            )

            advantages = (rewards - reward_mean) / reward_std
            ratio = torch.exp(logprob - old_logprob)
            clipped_ratio = torch.clamp(ratio, 1.0 - args.clip_range, 1.0 + args.clip_range)
            policy_loss_unclipped = ratio * advantages
            policy_loss_clipped = clipped_ratio * advantages
            policy_loss = -torch.mean(torch.min(policy_loss_unclipped, policy_loss_clipped))

            kl_term = torch.mean(logprob - ref_logprob)
            loss = policy_loss + args.kl_coef * kl_term

            loss.backward()
            accum_counter += 1

            if accum_counter % args.gradient_accumulation_steps == 0:
                if args.grad_clip is not None and args.grad_clip > 0:
                    clip_grad_norm_(model.parameters(), args.grad_clip)
                if args.device_type == "tpu":
                    xm.optimizer_step(optimizer, barrier=True)
                    xm.mark_step()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                total_updates += 1
                accum_counter = 0

            if progress is not None and step % max(1, args.log_interval) == 0:
                progress.set_postfix(
                    loss=float(loss.detach().cpu()),
                    kl=float(kl_term.detach().cpu()),
                    reward=float(rewards.mean().detach().cpu()),
                )

        if accum_counter != 0:
            if args.grad_clip is not None and args.grad_clip > 0:
                clip_grad_norm_(model.parameters(), args.grad_clip)
            if args.device_type == "tpu":
                xm.optimizer_step(optimizer, barrier=True)
                xm.mark_step()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            total_updates += 1
            accum_counter = 0

        if rank == 0:
            print(
                f"Epoch {epoch+1}/{args.num_train_epochs} finished. "
                f"Total updates: {total_updates}"
            )

    output_dir = (
        args.output_dir
        if args.output_dir
        else os.path.join(
            "checkpoints",
            f"{os.path.basename(args.model_name)}-ppo{'-lora' if args.use_lora else ''}",
        )
    )

    if args.device_type == "tpu":
        xm.rendezvous("save_model")
        if rank == 0:
            os.makedirs(output_dir, exist_ok=True)
            model.to("cpu")
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"PPO adapter/model saved to {output_dir}")
        xm.rendezvous("save_done")
    elif rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"PPO adapter/model saved to {output_dir}")


def _tpu_worker(rank: int, args: argparse.Namespace) -> None:
    args.device_type = "tpu"
    args.world_size = len(xm.get_xla_supported_devices())
    device = xm.xla_device()
    _train_one_process(args, device, rank, args.world_size)


def main() -> None:
    parser = argparse.ArgumentParser(description="基于 overall_score 的离线 PPO 训练脚本。")
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
    parser.add_argument("--min_score", type=float, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--reward_normalizer", type=float, default=5.0)
    parser.add_argument("--reward_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--kl_coef", type=float, default=0.1)
    parser.add_argument("--max_prompt_length", type=int, default=128)
    parser.add_argument("--max_response_length", type=int, default=64)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["auto", "cpu", "cuda", "tpu"],
        help="默认 cpu，可设为 auto/cuda/tpu。",
    )
    parser.add_argument(
        "--tpu_cores",
        type=str,
        default="1",
        help="--device tpu 时使用的核心数，填具体数字或 all。",
    )
    args = parser.parse_args()

    args.device_type = _resolve_device_choice(args.device)

    if args.device_type == "tpu":
        if not _XLA_AVAILABLE:
            raise RuntimeError("未安装 torch_xla，无法在 TPU 上训练。")
        num_cores = _resolve_tpu_cores(args.tpu_cores)
        if num_cores > 1:
            os.environ["TPU_NUM_DEVICES"] = str(num_cores)
            args.world_size = num_cores
            xmp.spawn(_tpu_worker, args=(args,), start_method="fork")
        else:
            args.world_size = 1
            device = xm.xla_device()
            _train_one_process(args, device, rank=0, world_size=1)
    else:
        args.world_size = 1
        device = torch.device(args.device_type)
        _train_one_process(args, device, rank=0, world_size=1)


if __name__ == "__main__":
    main()
