#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

# Configurable parameters
MODEL_NAME="Qwen/QwQ-32B"
USE_LORA="--use_lora"
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.05
LORA_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj"
DEEPSPEED_CONFIG="ds_zero2_offload.json"

EPOCHS=3
BATCH_SIZE=1
LEARNING_RATE=1e-5
GRAD_ACC=2

# Launch training
torchrun --nproc_per_node=8 deepspeed_finetune_lora.py \
  --model_name "${MODEL_NAME}" \
  ${USE_LORA} \
  --lora_r "${LORA_R}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${LORA_DROPOUT}" \
  --lora_target_modules "${LORA_TARGET_MODULES}" \
  --deepspeed_config "${DEEPSPEED_CONFIG}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --learning_rate "${LEARNING_RATE}" 