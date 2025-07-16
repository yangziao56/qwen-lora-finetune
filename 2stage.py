import os
import sys
import torch
import argparse
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from collections import Counter
import torch.nn.functional as F
from datetime import datetime  # Add this import for timestamp

def load_model(model_id_or_path, quant_config):
    # 本地路径优先，否则当成 HF Hub repo
    use_local = os.path.exists(model_id_or_path)
    # GPU 策略：多卡 auto，单卡放在 0
    ngpu = torch.cuda.device_count()
    #device_map = "auto" if ngpu > 1 else {"": 0}
    device_map = {"": 1}
    # 加载 tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(
        model_id_or_path,
        trust_remote_code=True,
        local_files_only=use_local
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        device_map=device_map,
        quantization_config=quant_config,
        trust_remote_code=True,
        local_files_only=use_local
    )
    return tokenizer, model

def get_famous_quotes(topic, model, tokenizer, device):
    # ---------- Stage 1 Prompt（最终版，可直接覆盖）----------
    system = (
        "You are an expert famous-quote curator.\n"
        "Rules:\n"
        "1. Output exactly 10 lines.\n"
        "2. Each line format: n. \"Quote\" — Author: one-sentence reason reflecting the mood.\n"
        "3. Use 10 distinct authors (no duplicates).\n"
        "4. Do NOT mention the persona name (e.g. Joy/Fear/etc.); convey the mood implicitly.\n"
        "5. Do NOT output any other text, reasoning, or apologies.\n"
        "6. Quotes will be used for slogan remixing, so avoid overly complex or lengthy quotes.\n"
    )

    user = (
        f"List 10 famous quotes that align with the mood of the “{topic}” persona.\n"
        'Format: n. "Quote" — Author: one-sentence reason.'
    )

    msgs = [{"role":"system","content":system}, {"role":"user","content":user}]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # 打印 Stage1 输入
    print("=== Stage 1 Prompt ===")
    print(prompt)
    # print("=== Stage 1 Input IDs ===")
    # print(inputs.input_ids)
    # 生成
    outs = model.generate(
        **inputs,
        max_new_tokens=500,
        do_sample=True,
        temperature=0.7,          # 控制随机度
        top_p=0.9,                # nucleus sampling
        repetition_penalty=1.2    # 惩罚重复
    )
    text = tokenizer.decode(outs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    # 打印 Stage1 输出
    print("=== Stage 1 Output ===")
    print(text)
    return [q.strip() for q in text.split("\n") if q.strip()]

def extract_slogan(sample: str) -> str:
    m = re.search(r'Remix Slogan:\s*"([^"]+)"', sample, flags=re.I)
    return m.group(1).strip() if m else sample.strip()

def remix_with_poetry(quotes, persona, model, tokenizer, device, num_passes=1):
    # Stage 2 prompt 构造（去掉原本的“Step 1【Quote Matching】”）
    system = (
        "You are an award-winning slogan copywriter. Follow ALL steps and output in English.\n\n"
        # "Step 1【Structure Breakdown】\n"
        # "- Copy the ★ quote and use | to separate editable/fixed parts.\n\n"
        # "Step 2【Vocabulary Replacement】\n"
        # "- Propose 1–2 replacement word/phrase sets with brief reasons.\n\n"
        # "Step 3【Remix Generation】\n"
        # "- Using the best replacement, output ONE final slogan: keep original syntax and meter, include brand name/keyword, end with punctuation.\n\n"
        # "Output format must be:\n"
        # "Explanation: …\n"
        # "Structure: …\n"
        # "Replacements: …\n"
        # "Remix Slogan: \"…\""
        "Step 2【Structure Breakdown】\n"
        "- Copy the quote and use | to separate editable and fixed parts.\n"
        "- Point out which words can be replaced with brand keywords and why.\n\n"
        "Step 3【Vocabulary Replacement】\n"
        "- Propose 1‑2 replacement word/phrase sets with short reasons.\n"
        "- Limit yourself to changing at most two words from the original quote to preserve its rhythm and cadence. Adding the brand name also counts towards this limit.\n\n"
        "Step 4【Remix Generation】\n"
        "- Using the best replacement, output ONE final slogan:\n"
        "  * keep original syntax and meter\n"
        "  * include brand name / keyword\n"
        "  * end with punctuation.\n\n"
        "Output format must be:\n"
        "Explanation: …\n"
        "Structure: …\n"
        "Replacements: …\n"
        "Remix Slogan: \"…\"\n"
        "\n"
        "Important constraints for the final Remix Slogan:\n"
        "- Do **NOT** use first‑person words such as I, we, me, my, our.\n"
        "- Speak from a neutral or second‑/third‑person perspective so the brand is **mentioned**, not the narrator.\n"

    )
    guidelines = {
        "Pride": "celebrate achievement or exclusivity",
        "Anticipation": "spark curiosity or suspense",
        "Fear": "neutralize fear of missing out or add urgency",
        "Joy": "evoke delight or excitement",
        "Trust": "emphasize safety or reliability",
    }
    user = (
        f"Persona (tone only): {persona}\n"
        f"Brand: Coca-Cola\n"
        f"Brand keywords: refreshment, happiness, moments\n"
        f"Persona goal: {guidelines[persona]}\n\n"
        "Here is the quote:\n" +
        "\n".join(f"- {q}" for q in quotes) +
        "\n\nPlease begin Step 1."
    )
    msgs = [{"role":"system","content":system}, {"role":"user","content":user}]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    print("=== Stage 2 Prompt ===")
    print(prompt)

    if num_passes > 1:
        gen = model.generate(
            **inputs,
            max_new_tokens=5120,
            num_return_sequences=num_passes,
            do_sample=True,
            temperature=1.0,
            top_k=40,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        seqs = gen.sequences  # shape [k, prompt+gen]
        scores = gen.scores    # list of [k, vocab] per token
        prompt_len = inputs.input_ids.shape[-1]

        slogans, logps, full_texts = [], [], []
        for i in range(num_passes):
            txt = tokenizer.decode(seqs[i][prompt_len:], skip_special_tokens=True)
            full_texts.append(txt)
            slog = extract_slogan(txt)
            slogans.append(slog)
            # avg log-prob
            lp = 0.0
            for tok, sc in zip(seqs[i][prompt_len:], scores):
                lp += F.log_softmax(sc[i], dim=-1)[tok].item()
            logps.append(lp / len(scores))

            print(f"\n--- Pass {i+1} ---")
            print(txt)

        # voting / fallback
        vote = Counter(slogans)
        best_slog, freq = vote.most_common(1)[0]
        if freq == 1:
            idx = int(torch.tensor(logps).argmax())
            decision = "highest confidence"
        else:
            idx = slogans.index(best_slog)
            decision = f"hard vote (freq={freq})"

        best_full = full_texts[idx]
        print(f"\n=== Best candidate ({decision}) ===\n{best_full}")
        return best_full

    else:
        outs = model.generate(**inputs,
                              max_new_tokens=5120,
                              do_sample=True,
                              temperature=1.0,
                              top_k=40,
                              pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(outs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        print("=== Stage 2 Output ===")
        print(text)
        return text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_quote", type=str, required=False, default="Qwen/Qwen2-7B-Instruct", help="轻量模型路径（用于提取名言）")
    parser.add_argument("--model_base",  type=str, required=False, default="Qwen/QwQ-32B", help="基础模型路径（用于 Remix）")
    parser.add_argument("--persona",     type=str, choices=["Pride","Anticipation","Fear","Joy","Trust"], default="Joy")
    parser.add_argument("--num_passes", type=int, default=5, help="Stage2 multi-pass sampling")
    args = parser.parse_args()

    # --- 修改：创建输出文件，并把 stdout 重定向过去 ---
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    model_name = os.path.basename(args.model_base)
    # 添加时间戳，格式 YYYYMMDD_HHMMSS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"{model_name}_2stage_{timestamp}.txt")
    sys.stdout = open(output_path, "a", encoding="utf-8")
    # ----------------------------------------------------------------

    # 量化配置
    quant = BitsAndBytesConfig(load_in_8bit=True)

    # 1) 加载提取名言模型
    tok_q, m_q = load_model(args.model_quote, quant)
    # 2) 加载 Remix 模型
    tok_r, m_r = load_model(args.model_base,  quant)

    print("=== Stage 1: Getting famous quotes ===")
    quotes = get_famous_quotes(args.persona, m_q, tok_q, m_q.device)

    print("=== Stage 2: Remix slogans (one quote at a time) ===")
    remixes = []
    for idx, quote in enumerate(quotes, start=1):
        print(f"--- Remixing quote {idx}/{len(quotes)} ---")
        # 只传入单条 quote
        remix = remix_with_poetry([quote], args.persona, m_r, tok_r, m_r.device, args.num_passes)
        remixes.append(remix)

    # 如果想把所有 remix 结果合并输出：
    final_output = "\n\n".join(remixes)
    print("=== All Remixes ===")
    print(final_output)

if __name__ == "__main__":
    main()