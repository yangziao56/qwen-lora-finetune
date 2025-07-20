#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi‑pass slogan generation with hard‑vote / confidence fallback
----------------------------------------------------------------
• Runs the chosen model *k* times (independent stochastic decoding).
• Extracts the “Remix Slogan” line from every sample.
• If duplicates exist → majority vote; otherwise → pick the answer with
  the highest average token log‑probability.
• Works on single‑ or multi‑GPU, supports 8‑bit loading via BitsAndBytes.
"""

import os
import re
import sys
import ast
import torch
import argparse
from datetime import datetime
import torch.nn.functional as F
from collections import Counter
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

# ----------------------------- helper --------------------------------- #
def extract_slogan(sample: str) -> str:
    """
    Grab the slogan from one generation. If the expected tag is missing,
    fall back to the whole text (rare but keeps script robust).
    """
    m = re.search(r'Remix Slogan:\s*"([^"]+)"', sample, flags=re.I)
    return m.group(1).strip() if m else sample.strip()


def average_logp(token_ids, logits_list) -> float:
    """
    Compute average log‑probability of generated tokens.
    `token_ids`  : IntTensor, shape [new_tokens]
    `logits_list`: list of FloatTensor, len = new_tokens
    """
    lp_sum = 0.0
    for tok, logits in zip(token_ids, logits_list):
        lp_sum += F.log_softmax(logits, dim=-1)[tok].item()
    return lp_sum / len(logits_list)
# ---------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi‑pass slogan generation with voting / confidence"
    )
    parser.add_argument(
        "--model_id_or_path",
        type=str,
        #default="Qwen/QwQ-32B",
        default="/sensei-fs-3/users/ziaoy/llm/qwen/checkpoints/QwQ-32B-lora-ds/final-20250708-205411",
        help="Hugging Face model ID or local path",
    )
    personas = ["Pride", "Anticipation", "Fear", "Joy", "Trust"]
    parser.add_argument(
        "--persona",
        type=str,
        choices=personas,
        default="Joy",
        help="Persona (tone only) for slogan generation",
    )
    parser.add_argument(
        "--num_passes",
        type=int,
        default=1,
        help="Number of independent samples (k)",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="{'': 0}",
        help="Device map: 'auto' or dict literal, e.g. \"{'':0}\" (default: single‐GPU on card 0)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus-sampling threshold",
    )
    parser.add_argument(
        "--use_voting",
        action="store_true",
        default=False,
        help="Enable hard-vote among candidates; if false, always pick highest confidence",
    )
    args = parser.parse_args()

    # -------- logging to file (optional) --------------------------------
    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)
    model_tag = os.path.basename(args.model_id_or_path.rstrip("/"))
    # add current timestamp, format YYYYMMDD_HHMMSS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(out_dir, f"{model_tag}_multipass_{timestamp}.txt")
    sys.stdout = open(log_path, "a", encoding="utf-8")
    # --------------------------------------------------------------------

    print(f"Loading model: {args.model_id_or_path}")
    print(f"Persona: {args.persona} | Passes (k): {args.num_passes}")

    # ---- load tokenizer & model (half-precision + Flash Attention) -----
    # bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)  # Disable 8-bit configuration
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id_or_path, trust_remote_code=True
    )
    ngpu = torch.cuda.device_count()

    # ---- device_map controlled by CLI, default single GPU 0 --------
    if args.device_map.strip().lower() == "auto":
        device_map = "auto"
    else:
        device_map = ast.literal_eval(args.device_map)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id_or_path,
        device_map=device_map,
        trust_remote_code=True,
        # quantization_config=bnb_cfg,  # Disable 8-bit configuration
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",  # Enable Flash Attention 2
    )
    model.eval()

    # ---- build prompt --------------------------------------------------
    system_msg = (
        "You are an award‑winning slogan copywriter. "
        "Your task is to remix a classic quote, slang, lyric, or popular expression into a brand slogan. "
        "Follow ALL steps and output in English.\n\n"
        "Step 1【Quote Matching】\n"
        "- List 3 classic sentences (quote, slang, lyric, or popular expression) that best fit the given persona and brand, each with author.\n"
        "- Each sentence must be 5–10 English words to ensure both clarity and brevity.\n"
        "- Do not include the persona name in your output; only reflect its mood through wording.\n"
        "- Explain in one sentence why each quote matches.\n"
        "- Mark the best‑matching quote with ★.\n\n"
        "Step 2【Structure Breakdown】\n"
        "- Copy the ★ quote and use | to separate editable and fixed parts.\n"
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
    
        
        # "## OUTPUT RULES (STRICT)\n"
        # "- Think step-by-step **internally**; do *not* print any chain-of-thought.\n"
        # "- After printing the line starting with `Remix Slogan:` **stop immediately**; do not add extra text.\n"
        # "- Total output ≤ 200 English words.\n"
        
    )

    brand = "Coca‑Cola"
    guidelines = {
        "Pride": "celebrate achievement or exclusivity",
        "Anticipation": "spark curiosity or suspense",
        "Fear": "neutralize fear of missing out or add urgency",
        "Joy": "evoke delight or excitement",
        "Trust": "emphasize safety or reliability",
    }
    user_msg = (
        f"Persona (tone only): {args.persona}\n"
        f"Brand: {brand}\n"
        #f"Brand keywords: refreshment, happiness, moments\n"
        f"Persona goal: {guidelines[args.persona]}\n\n"
        "Please begin Step 1.\n"
        "Do not include the persona name in your output; only reflect its mood through wording.\n"
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[-1]

    # ---- multi-pass generation in one batch ---------------------------
    print("Generating…")
    with torch.no_grad():
        gen_dict = model.generate(
            **inputs,
            max_new_tokens=8000,
            num_return_sequences=args.num_passes,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=40,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    seqs = gen_dict.sequences    # shape: [k, prompt+gen]
    scores = gen_dict.scores     # list(len=new_tokens) of [k, vocab]

    MAX_NEW = args.max_new_tokens
    candidates = []  # store valid passes

    for idx in range(args.num_passes):
        # slice out newly generated tokens
        new_tokens = seqs[idx][prompt_len:]
        # skip if reached max_new_tokens, likely truncated
        if new_tokens.shape[0] >= MAX_NEW:
            print(f"\n--- Pass {idx+1} truncated (≥{MAX_NEW} tokens), skipping ---")
            continue

        gen_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        slogan = extract_slogan(gen_text)
        lp = average_logp(new_tokens, [s[idx] for s in scores])

        print(f"\n--- Pass {idx+1} (valid) ---")
        print(gen_text)

        # preserve original idx for later mapping
        candidates.append((idx, slogan, lp, gen_text))

    if not candidates:
        raise RuntimeError("All passes were considered truncated. No valid candidates were generated. Please increase max_new_tokens or simplify the prompt.")

    # extract slogans and log probabilities from valid candidates
    slogans = [c[1] for c in candidates]
    logps   = [c[2] for c in candidates]

    # ---- voting / confidence fallback (optional) ------------------------------
    vote = Counter(slogans)
    best_slogan_vote, freq = vote.most_common(1)[0]

    if args.use_voting:
        if freq > 1:
            best_slogan = best_slogan_vote
            best_in_candidates = slogans.index(best_slogan)
            decision = f"hard vote (frequency = {freq})"
        else:
            best_in_candidates = int(torch.tensor(logps).argmax().item())
            best_slogan = slogans[best_in_candidates]
            decision = "highest confidence (no majority)"
    else:
        best_in_candidates = int(torch.tensor(logps).argmax().item())
        best_slogan = slogans[best_in_candidates]
        decision = "highest confidence (voting disabled)"

    # retrieve the original full generation text
    best_idx, best_slogan, _, best_full = candidates[best_in_candidates]

    # ---- final report -----------------------------------------------
    print("\n================ BEST CANDIDATE ================")
    print(f"Decision: {decision}")
    print(f"Slogan  : {best_slogan}")
    print("-----------------------------------------------")
    print(best_full)
    print("===============================================\n")

    # ---- Judger Scoring with QWQ-32B --------------------------------
    # === 在加载 Judger 前卸载主模型，避免显存溢出 ===
    del model, tokenizer
    torch.cuda.empty_cache()

    print("\n=============== JUDGER FEEDBACK ===============")
    judger_id = "Qwen/QwQ-32B"
    judger_tokenizer = AutoTokenizer.from_pretrained(
        judger_id, trust_remote_code=True
    )
    judger_model = AutoModelForCausalLM.from_pretrained(
        judger_id,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    judger_model.eval()

    judger_prompt = (
        "Please assign a score from 0 to 10 to the following brand slogan and provide a brief rationale:\n"
        f"Slogan: \"{best_slogan}\"\n"
        "Score and rationale:"
    )
    judger_inputs = judger_tokenizer(
        judger_prompt, return_tensors="pt"
    ).to(judger_model.device)

    with torch.no_grad():
        judger_out = judger_model.generate(
            **judger_inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=judger_tokenizer.eos_token_id,
        )

    seq = judger_out[0]
    feedback = judger_tokenizer.decode(
        seq[judger_inputs.input_ids.shape[-1]:],
        skip_special_tokens=True
    )
    print(feedback)
    print("===============================================\n")


if __name__ == "__main__":
    main()
