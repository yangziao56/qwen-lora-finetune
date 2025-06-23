from datasets import load_dataset
import json, os, random

random.seed(42)
os.makedirs('data/jsonl', exist_ok=True)

# 加载 CSV，列名请根据 sloganlist.csv 实际字段调整
ds = load_dataset('csv', data_files='data/raw/sloganlist.csv')['train']

with open('data/jsonl/slogans.jsonl', 'w', encoding='utf-8') as f:
    for item in ds:
        brand  = item.get('Company') or item.get('Company') or 'Unknown Brand'
        slogan = item.get('Slogan') or item.get('Slogan') or ''
        if not slogan or len(slogan) > 120:
            continue
        example = {
            "messages": [
                {"role": "system",    "content": "Generate slogan for certain product."},
                {"role": "user",      "content": f"Please generate a slogan for {brand}."},
                {"role": "assistant", "content": slogan.strip()}
            ]
        }
        f.write(json.dumps(example, ensure_ascii=False) + "\n")
