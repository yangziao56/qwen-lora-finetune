torchrun --nproc_per_node=8 /sensei-fs-3/users/ziaoy/llm/qwen/scripts/finetune_lora.py \
  --model_name Qwen/Qwen2-7B-Instruct \
  --use_lora \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj
