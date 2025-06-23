torchrun --nproc_per_node=8 /sensei-fs-3/users/ziaoy/llm/qwen/scripts/finetune_lora.py \
  --model_name Qwen/QwQ-32B \
  --use_lora \
  --lora_r 1 \
  --lora_alpha 2 \
  --lora_dropout 0.1 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj
