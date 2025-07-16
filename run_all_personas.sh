#!/usr/bin/env bash
set -uo pipefail

# List of predefined personas
personas=(Pride Anticipation Fear Joy Trust)

# Optional: override python command or add --model_id_or_path if needed
#PYTHON_CMD="python test.py"
PYTHON_CMD="python 1stage_multi_pass.py"

for p in "${personas[@]}"; do
  echo "============================"
  echo "Running with persona: $p"
  echo "============================"

  # Record start time
  start=$(date +%s)

  $PYTHON_CMD --persona "$p"

  # Record end time and compute duration
  end=$(date +%s)
  echo "Run time for persona '$p': $((end - start)) seconds"
done

echo "All runs complete."