from datasets import load_dataset
import json, os, random

random.seed(42)
os.makedirs('../data/jsonl', exist_ok=True)

# 1) Load local Poetry CSV file
ds = load_dataset(
    'csv',
    data_files='../data/raw/PoetryFoundationData.csv',
    split='train'
)

# 2) Write JSONL output
with open('../data/jsonl/poems.jsonl', 'w', encoding='utf-8') as f:
    for item in ds:
        title = item.get('Title', '').strip()
        poet  = item.get('Poet', '').strip()
        poem  = item.get('Poem', '').strip()
        # Skip empty poems or those longer than 2048 characters
        if not poem or len(poem) > 2048:
            continue
        example = {
            "messages": [
                {"role": "system",    "content": "You are a poetry expert."},
                {"role": "user",      "content": f"Please write a poem similar to \"{title}\" by {poet}."},
                {"role": "assistant", "content": poem}
            ]
        }
        f.write(json.dumps(example, ensure_ascii=False) + "\n")