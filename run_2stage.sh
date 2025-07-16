# #!/bin/bash
# # 用法: bash run_2stage.sh <persona>
# # persona 可选: Pride, Anticipation, Fear, Joy, Trust

# python 2stage.py \
#   --model_base /sensei-fs-3/users/ziaoy/llm/qwen/checkpoints/QwQ-32B-lora-ds/final-20250627-151621 \
#   --model_quote /sensei-fs-3/users/ziaoy/llm/qwen/checkpoints/QwQ-32B-lora-ds/final-20250627-151621 \
#   --persona "${1:-Joy}"
# #!/bin/bash
# # 用法: bash run_2stage.sh <persona>
# # persona 可选: Pride, Anticipation, Fear, Joy, Trust


#!/usr/bin/env bash
set -euo pipefail

# 模型路径，请根据实际修改
MODEL_BASE="/sensei-fs-3/users/ziaoy/llm/qwen/checkpoints/QwQ-32B-lora-ds/final-20250708-205411"
MODEL_QUOTE="/sensei-fs-3/users/ziaoy/llm/qwen/checkpoints/Qwen2-7B-Instruct-lora-ds/final-20250708-213018"

# 预定义 personas
personas=(Pride Anticipation Fear Joy Trust)

for p in "${personas[@]}"; do
  echo "======================================"
  echo " Running 2stage.py with persona: $p"
  echo "======================================"
  python 2stage.py \
    --model_base "$MODEL_BASE" \
    --model_quote "$MODEL_QUOTE" \
    --persona "$p"
done

echo "All personas done."