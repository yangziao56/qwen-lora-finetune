import json

input_path  = "/sensei-fs-3/users/ziaoy/llm/qwen/data/raw/quotes.jsonl"
output_path = "/sensei-fs-3/users/ziaoy/llm/qwen/data/jsonl/quotes_converted.jsonl"

with open(input_path, "r", encoding="utf-8") as fin, \
     open(output_path, "w", encoding="utf-8") as fout:
    for line in fin:
        line = line.strip()
        # skip empty lines or comments
        if not line or line.startswith("//"):
            continue

        record = json.loads(line)
        quote  = record.get("quote", "").strip()
        author = record.get("author", "").strip().rstrip(",")

        dialogue = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a quote expert."
                },
                {
                    "role": "user",
                    "content": f"Please give me a quote from {author}."
                },
                {
                    "role": "assistant",
                    "content": quote
                }
            ]
        }

        fout.write(json.dumps(dialogue, ensure_ascii=False) + "\n")