#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
two_stage_to_three_stage_slogan.py
可直接 python two_stage_to_three_stage_slogan.py 运行
"""
import os, sys, re, argparse, torch
from datetime import datetime
from collections import Counter

import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


# ========= 工具函数 ========= #
def load_model(model_id_or_path: str, quant_config: BitsAndBytesConfig, device_id: int = None):
    """优先从本地加载；单卡放到 0，多卡 auto，也可以自行修改"""
    use_local = os.path.exists(model_id_or_path)
    ngpu = torch.cuda.device_count()
    if device_id is not None:
        # 强制把整个模型 load 到指定卡
        device_map = {"": device_id}
    else:
        device_map = {"": 0} if ngpu == 1 else "auto"
    tokenizer = AutoTokenizer.from_pretrained(
        model_id_or_path, trust_remote_code=True, local_files_only=use_local
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        device_map=device_map,
        quantization_config=quant_config,
        trust_remote_code=True,
        local_files_only=use_local,
    )
    return tokenizer, model


def extract_slogan(sample: str) -> str:
    # 匹配 Remix Slogan: "..." 部分，忽略后面的任何内容
    m = re.search(r'(Remix Slogan:\s*"[^"]+")', sample, flags=re.I)
    if m:
        return m.group(1).strip()

    # 如果上面没匹配到，作为后备方案，只提取双引号内的内容
    m = re.search(r'"([^"]+)"', sample, flags=re.I)
    return m.group(1).strip() if m else sample.strip()


# ========= Stage 1：名言抽取 ========= #
def get_famous_quotes(topic: str, model, tokenizer, device):
    system = (
        "You are an expert famous-quote curator.\n"
        "Rules:\n"
        "1. Output exactly 10 lines.\n"
        '2. Each line format: n. "Quote" — Author: one-sentence reason reflecting the mood.\n'
        "3. Use 10 distinct authors.\n"
        "4. Do NOT mention the persona name; convey the mood implicitly.\n"
        "5. Do NOT output any other text."
        "6. Quotes will be used for slogan remixing, so avoid overly complex or lengthy quotes.\n"
    )
    user = (
        f'List 10 famous quotes that align with the mood of the “{topic}” persona.\n'
        'Format: n. "Quote" — Author: one-sentence reason.'
    )
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    # <<< print prompt & flush >>>
    print("=== Stage 1 Prompt ===\n", prompt, flush=True)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outs = model.generate(
        **inputs,
        max_new_tokens=500,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
    )
    text = tokenizer.decode(outs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)

    # <<< print output >>>
    print("=== Stage 1 Output ===\n", text, flush=True)

    return [q.strip() for q in text.splitlines() if q.strip()]


# ========= Stage 2a：结构拆解 ========= #
def structure_breakdown(quote: str, persona: str, brand: str, keywords: str,
                        guideline: str, model, tokenizer, device):
    system = (
        "You are an award-winning slogan copywriter.\n"
        "Task: **ONLY** perform Step 2【Structure Breakdown】 for the given quote.\n"
        "- Copy the quote and use | to separate editable / fixed parts.\n"
        "- Indicate which words can be replaced with brand keywords and why.\n"
        "Output format:\nStructure: …\n"
    )
    user = (
        f"Persona tone: {persona}\nBrand: {brand}\nBrand keywords: {keywords}\n"
        f"Persona goal: {guideline}\n\nQuote:\n{quote}\n\nPlease output Step 2 only."
    )
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    # <<< print prompt >>>
    print("=== Stage 2a Prompt ===\n", prompt, flush=True)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outs = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.7)
    result = tokenizer.decode(outs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()

    # <<< print output >>>
    print("=== Stage 2a Output ===\n", result, flush=True)

    return result


# ========= Stage 2b：词汇替换 ========= #
def vocabulary_replacement(quote: str, breakdown: str, persona: str, brand: str,
                           keywords: str, guideline: str, model, tokenizer, device):
    system = (
        "You are an award-winning slogan copywriter.\n"
        "Task: **ONLY** perform Step 3【Vocabulary Replacement】.\n"
        "- Propose 1-2 replacement word/phrase sets with brief reasons.\n"
        "- Limit yourself to changing at most two words from the original quote; "
        "adding the brand name counts toward the two-word limit.\n"
        "Output format:\nReplacements: …\n"
    )
    user = (
        f"Persona tone: {persona}\nBrand: {brand}\nBrand keywords: {keywords}\n"
        f"Persona goal: {guideline}\n\nQuote:\n{quote}\n\nPrevious breakdown:\n{breakdown}\n"
        "Please output Step 3 only."
    )
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    # <<< print prompt >>>
    print("=== Stage 2b Prompt ===\n", prompt, flush=True)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outs = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.7)
    result = tokenizer.decode(outs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()

    # <<< print output >>>
    print("=== Stage 2b Output ===\n", result, flush=True)

    return result


# ========= Stage 2c：最终 Remix ========= #
def remix_generation(quote: str, breakdown: str, replacements: str, persona: str, brand: str,
                     keywords: str, guideline: str, model, tokenizer, device, num_passes=3):
    system = (
        "You are an award-winning slogan copywriter.\n"
        "Task: **ONLY** perform Step 4【Remix Generation】.\n"
        "- Using the best replacement, output ONE final slogan.\n"
        "- The slogan must be enclosed in double quotes.\n"
        "- Do NOT output any text after the closing double quote of the slogan.\n"
        "  * keep original syntax and meter\n"
        "  * include brand name / keyword\n"
        "  * end with punctuation\n"
        "- NO first‑person words (I, we, our…). Speak neutrally or in 2nd/3rd person.\n"
        "Output format:\nExplanation: …\nStructure: …\nReplacements: …\nRemix Slogan: \"…\"\n"
    )
    user = (
        f"Persona tone: {persona}\nBrand: {brand}\nBrand keywords: {keywords}\n"
        f"Persona goal: {guideline}\n\nQuote:\n{quote}\n\nBreakdown:\n{breakdown}\n\n"
        f"Candidate replacements:\n{replacements}\n\nPlease output Step 4 only."
    )
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    # <<< print prompt >>>
    print("=== Stage 2c Prompt ===\n", prompt, flush=True)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    gen = model.generate(
        **inputs,
        max_new_tokens=5012,
        num_return_sequences=num_passes,
        do_sample=True,
        temperature=1.0,
        top_k=40,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    seqs, scores = gen.sequences, gen.scores
    prompt_len = inputs.input_ids.shape[-1]

    slogans, logps, full_texts = [], [], []
    for i in range(num_passes):
        txt = tokenizer.decode(seqs[i][prompt_len:], skip_special_tokens=True)
        full_texts.append(txt)
        slogans.append(extract_slogan(txt) or f"SEQ-{i}")
        lp = sum(
            F.log_softmax(sc[i], dim=-1)[tok].item()
            for tok, sc in zip(seqs[i][prompt_len:], scores)
        ) / len(scores)
        logps.append(lp)

    vote = Counter(slogans)
    best_slog, freq = vote.most_common(1)[0]
    # best_slog 现在就是我们想要的干净输出
    best = best_slog

    # <<< print output >>>
    print("=== Stage 2c Output ===\n", best, flush=True)

    return best


# ========= 主程序 ========= #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_quote", type=str, default="Qwen/Qwen2-7B-Instruct",
                        help="Stage 1（名言抽取）模型")
    parser.add_argument("--model_breakdown", type=str, default="Qwen/Qwen1.5-7B-Chat",
                        help="Stage 2a（结构拆解）模型")
    parser.add_argument("--model_replace", type=str, default="Qwen/Qwen1.5-14B-Chat",
                        help="Stage 2b（词汇替换）模型")
    parser.add_argument("--model_remix", type=str, default="Qwen/QwQ-32B",
                        help="Stage 2c（Remix 生成）模型")
    parser.add_argument("--persona", choices=["Pride", "Anticipation", "Fear", "Joy", "Trust"],
                        default="Joy")
    parser.add_argument("--num_passes", type=int, default=5, help="Remix 生成采样次数")
    args = parser.parse_args()

    # 输出文件重定向
    os.makedirs("output", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = os.path.join("output", f"{os.path.basename(args.model_remix)}_4stage_{timestamp}.txt")
    sys.stdout = open(outfile, "a", encoding="utf-8")

    quant = BitsAndBytesConfig(load_in_8bit=True)

    # 加载四个模型
    tok_q,  m_q  = load_model(args.model_quote,     quant, device_id=2)
    tok_b,  m_b  = load_model(args.model_breakdown, quant, device_id=3)
    tok_rp, m_rp = load_model(args.model_replace,   quant, device_id=4)
    tok_rx, m_rx = load_model(args.model_remix,     quant, device_id=5)

    guideline_map = {
        "Pride": "celebrate achievement or exclusivity",
        "Anticipation": "spark curiosity or suspense",
        "Fear": "neutralize fear of missing out or add urgency",
        "Joy": "evoke delight or excitement",
        "Trust": "emphasize safety or reliability",
    }
    brand = "Coca‑Cola"
    keywords = "refreshment, happiness, moments"

    # === Stage 1 === #
    print("=== Stage 1: Getting famous quotes ===")
    quotes = get_famous_quotes(args.persona, m_q, tok_q, m_q.device)

    # === Stage 2 === #
    print("=== Stage 2: 3‑step Remix ===")
    remixes = []
    for idx, quote in enumerate(quotes, start=1):
        print(f"\n--- Quote {idx}/{len(quotes)} ---")
        # Step 2a
        breakdown = structure_breakdown(
            quote, args.persona, brand, keywords,
            guideline_map[args.persona], m_b, tok_b, m_b.device
        )
        # Step 2b
        replacement = vocabulary_replacement(
            quote, breakdown, args.persona, brand, keywords,
            guideline_map[args.persona], m_rp, tok_rp, m_rp.device
        )
        # Step 2c
        remix_text = remix_generation(
            quote, breakdown, replacement, args.persona, brand, keywords,
            guideline_map[args.persona], m_rx, tok_rx, m_rx.device, args.num_passes
        )
        remixes.append(remix_text)
        # print(remix_text) # 这一行可以删除，因为 remix_generation 内部已经打印

    print("\n=== All Remixes ===\n")
    print("\n\n".join(remixes))


if __name__ == "__main__":
    main()
