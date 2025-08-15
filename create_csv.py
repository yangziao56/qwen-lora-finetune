#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi‑brand / multi‑persona headline‑ or slogan‑generation script
----------------------------------------------------------------
启动示例：
torchrun --standalone --nproc_per_node=8 multi_pass_dataset.py \
  --model_id_or_path /path/to/model \
  --brands Photoshop,Lightroom \
  --personas Pride,Joy \
  --generation_type slogan \
  --num_generations 10
"""

import os
import re
import ast
import csv
import argparse
from datetime import datetime
from typing import List, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------- helper functions ---------------------- #
def extract_output(sample: str, generation_type: str) -> str:
    # 如果生成里有 </think>，只取其后部分再做匹配
    if '</think>' in sample:
        sample = sample.split('</think>', 1)[1]
    
    # 更新正则表达式以匹配 "Step 4 Remix Headline/Slogan:" 的格式
    pattern = rf'Step\s*4\s*Remix\s*{generation_type}:\s*"(.*?)"'
    m = re.search(pattern, sample, flags=re.DOTALL | re.I)
    return m.group(1).strip() if m else ""


def extract_star_quote(sample: str) -> str:
    # 将 </think> 之前丢弃，仅在思路标记后查 ★
    if '</think>' in sample:
        sample = sample.split('</think>', 1)[1]
    # 正则表达式查找以 ★ 开头，后跟 "引言" 的行
    pattern = r'^\s*★\s*.*?\"([^\"]+)\"'
    m = re.search(pattern, sample, flags=re.MULTILINE)
    return m.group(1).strip() if m else ""


def extract_explanation_for_best_quote(sample: str) -> str:
    # 丢弃 </think> 之前内容
    if '</think>' in sample:
        sample = sample.split('</think>', 1)[1]
    # 正则表达式查找 explanation_for_best_quote: 之后的内容
    pattern = r'explanation_for_best_quote:\s*(.*)'
    m = re.search(pattern, sample, flags=re.DOTALL | re.I)
    if m:
        explanation_text = m.group(1).strip()
        # 解释在 "Step 2" 之前结束
        if 'Step 2' in explanation_text:
            explanation_text = explanation_text.split('Step 2', 1)[0].strip()
        return explanation_text
    return ""


def extract_explanation(sample: str) -> str:
    # 丢弃 </think> 之前内容，提取“Explanation: …”行
    if '</think>' in sample:
        sample = sample.split('</think>', 1)[1]
    pattern = r'Explanation:\s*(.*)'
    m = re.search(pattern, sample, flags=re.DOTALL)
    return m.group(1).strip() if m else ""


def build_prompt(brand: str, persona: str, generation_type: str) -> Tuple[str, str]:
    guidelines = {
        "Pride": "celebrate achievement or exclusivity",
        "Anticipation": "spark curiosity or suspense",
        "Fear": "neutralize fear of missing out or add urgency",
        "Joy": "evoke delight or excitement",
        "Trust": "emphasize safety or reliability",
    }
    system_msg = (
        f"You are an award‑winning {generation_type} copywriter. "
        f"Your task is to remix a classic quote, slang, lyric, meme, or popular expression into a brand {generation_type}. "
        "Follow ALL steps and output in English.\n\n"
        "Step 1【Quote Matching】\n"
        "- Come up with 3 classic sentences (quote, slang, lyric, meme, or popular expression) that best fit the given persona and brand, each with author.\n"
        "- Each sentence must be 5–10 English words to ensure both clarity and brevity.\n"
        "- Do not include the persona name in your output; only reflect its mood through wording.\n"
        "- Explain how to come up with each quote based on the persona and the brand.\n"
        "- Mark the best-matching quote with ★.\n\n"
        "Step 2【Structure Breakdown】\n"
        "- Copy the ★ quote and use | to separate editable and fixed parts.\n"
        "- Point out which words can be replaced with brand keywords and why.\n\n"
        "Step 3【Vocabulary Replacement】\n"
        "- Propose 1‑2 replacement word/phrase sets with short reasons.\n"
        "- Limit yourself to changing at most two words from the original quote to preserve its rhythm and cadence. Adding the brand name also counts towards this limit.\n\n"
        "Step 4【Remix Generation】\n"
        f"- Using the best replacement, output ONE final {generation_type}:\n"
        "  * keep original syntax and meter\n"
        "  * include brand name / keyword\n"
        "  * end with punctuation.\n\n"
        # "Output format must be:\n"
        # "Explanation: …\n"
        # "Structure: …\n"
        # "Replacements: …\n"
        # f"Remix {generation_type.capitalize()}: \"…\"\n\n"
        "Output format must be:\n"
        "Step 1 Quote Matching: …\n"
        "Step 2 Structure Breakdown: …\n"
        "Step 3 Vocabulary Replacement: …\n"
        f"Step 4 Remix {generation_type.capitalize()}:\"…\"\n\n"
        "In step1, include:\n"
        "- quotes: a list of three numbered items, each with quote, author, and reason.\n"
        "- After the list, on a **new, separate line**, repeat the best quote. This line **must** start with the ★ symbol, followed by the quote in quotation marks. For example:\n"
        "  ★ \"The best quote goes here.\"\n"
        "- Immediately after the ★ line, add `explanation_for_best_quote`: how the best quote was chosen based on the persona and the brand.\n"
        "Important constraints for the final output:\n"
        "- Do **NOT** use first‑person words such as I, we, me, my, our.\n"
        "- Speak from a neutral or second‑/third‑person perspective so the brand is **mentioned**, not the narrator.\n"
    )
    user_msg = (
        f"Persona (tone only): {persona}\n"
        f"Brand: {brand}\n"
        f"Persona goal: {guidelines[persona]}\n\n"
        "Please begin Step 1.\n"
        "Do not include the persona name in your output; only reflect its mood through wording.\n"
    )
    return system_msg, user_msg
# -------------------------------------------------------------- #


def generate_for_one_pair(
    tokenizer,
    model,
    brand: str,
    persona: str,
    generation_type: str,
    num_passes: int,
    temperature: float,
    top_p: float,
    top_k: int = 40,
    log_file_handle=None,
) -> List[Tuple[str, str]]:  # 移除logp返回类型
    system_msg, user_msg = build_prompt(brand, persona, generation_type)
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[-1]

    with torch.no_grad():
        gen_dict = model.generate(
            **inputs,
            max_new_tokens=8000,
            num_return_sequences=num_passes,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            return_dict_in_generate=True,
            output_scores=False,  # 关闭scores计算来加速
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    seqs = gen_dict.sequences
    # scores = gen_dict.scores  # 注释掉scores获取
    MAX_NEW = 5120
    results: List[Tuple[str, str, str]] = []  # 移除logp类型

    for idx in range(num_passes):
        new_tokens = seqs[idx][prompt_len:]
        gen_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

        if log_file_handle:
            log_file_handle.write(f"--- Generation for {brand} | {persona} ---\n")
            log_file_handle.write(f"Token count: {new_tokens.shape[0]}\n\n")
            log_file_handle.write(gen_text)
            log_file_handle.write("\n------------------------------------------\n\n")
            log_file_handle.flush()

        if new_tokens.shape[0] >= MAX_NEW:
            continue
        slogan = extract_output(gen_text, generation_type)
        quote = extract_star_quote(gen_text)
        explanation = extract_explanation_for_best_quote(gen_text)
        # lp = average_logp(new_tokens, [s[idx] for s in scores])  # 注释掉logp计算
        # lp = 0.0  # 临时设为0以加速
        results.append((slogan, quote, explanation))  # 移除logp

    return results


def main():
    # ---------------- 分布式初始化 ---------------- #
    if dist.is_available():
        dist.init_process_group("nccl", init_method="env://")
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    torch.cuda.set_device(rank)

    # ---------------- 参数解析 ---------------- #
    parser = argparse.ArgumentParser(
        description="Batch‑generate brand × persona headlines or slogans and save to CSV"
    )
    parser.add_argument("--model_id_or_path", type=str,
                        default="/sensei-fs-3/users/ziaoy/llm/qwen/checkpoints/QwQ-32B-lora-ds/final-20250708-205411")
    parser.add_argument("--brands", type=str,
                        #default="Photoshop, Creative Cloud, Coca-Cola, Nike, Iphone, Google Cloud, Starbucks Coffee, Adidas Sneakers, Dyson Vacuum, Peloton Bike, Lego Sets, Nintendo Switch, Kindle Paperwhite, Fitbit Tracker, Nespresso Machine, KitchenAid Mixer, Ray-Ban Sunglasses, Patagonia Jackets, Timberland Boots, JBL Speakers, GoPro Camera, Brita Water Filter, OXO Kitchen Tools, Tupperware Containers, Samsonite Luggage"
                        #default="Photoshop, Creative Cloud, Coca-Cola, Nike, Iphone, Google Cloud, Starbucks Coffee, Adidas Sneakers, Dyson Vacuum, Peloton Bike, Lego Sets, Nintendo Switch, Kindle Paperwhite, Fitbit Tracker, Nespresso Machine, KitchenAid Mixer, Ray-Ban Sunglasses, Patagonia Jackets, Timberland Boots, JBL Speakers, GoPro Camera, Brita Water Filter, OXO Kitchen Tools, Tupperware Containers, Samsonite Luggage,Amazon Prime, Netflix, Disney+, Apple Music, Spotify Premium, YouTube Premium, Audible, HBO Max, DoorDash DashPass, Uber One, Lyft Pink, Marriott Bonvoy, Hilton Honors, IHG One Rewards, Delta SkyMiles, United MileagePlus, American Airlines AAdvantage, Sephora Beauty Insider, Ulta Ultamate Rewards, Nike Membership, Adidas Creators Club, Peloton All-Access, Fitbit Premium, Grammarly Premium, Canva Pro, Dropbox Plus, Squarespace, Adobe Creative Cloud Subscription, Microsoft 365, NordVPN, HelloFresh, Blue Apron, Dollar Shave Club, Casper, Warby Parker"
                        #default="Hulu, Paramount+, Peacock Premium, Discovery+, ESPN+, Crunchyroll Premium, Showtime OTT, SiriusXM Streaming, Kindle Unlimited, Google One, iCloud+ Storage, Calm Premium, Headspace Plus, Coursera Plus, MasterClass Membership, Duolingo Super, Skillshare Premium, Shopify Plus, BarkBox, Stitch Fix"
                        #default="Walmart+ Membership, Instacart+, Grubhub+, Shipt Membership, Xbox Game Pass Ultimate, PlayStation Plus, Sling TV, FuboTV, Tidal HiFi Plus, Pandora Plus, Starz Streaming Service, BritBox, AMC Stubs A-List, REI Co-op Membership, Panera Unlimited Sip Club, Planet Fitness Black Card, Bloomberg Digital Subscription, Shutterstock Subscription, AncestryDNA Membership, 23andMe Plus Membership, Samsung, Sony, LG, Panasonic, Lenovo, Dell, HP, Acer, Asus, Intel, Nvidia, AMD, IBM, Cisco, Oracle, Salesforce, Zoom, Slack, Pepsi, Budweiser"
                        #default="Toyota, Ford, BMW, Mercedes-Benz, Tesla, McDonald's, Burger King, KFC, Subway, Domino's, Visa, Mastercard, American Express, PayPal, eBay, Etsy, Booking.com, Expedia, TikTok, Facebook"
                        #default="Porsche, Audi, Hyundai, Kia, Jeep, Gucci, Louis Vuitton, Chanel, Prada, Hermes, Zara, H&M, Uniqlo, Target, Best Buy, Home Depot, Lowe's, Verizon, AT&T, T-Mobile"
                        #default="Chipotle, Dunkin', 7-Eleven, IKEA, FedEx, UPS, DHL, Under Armour, Reebok, Lululemon"
                        #default="Volkswagen, Honda, Nissan, Subaru, Southwest Airlines, JetBlue, Emirates, Qatar Airways, Geico, Progressive Insurance, State Farm, Wells Fargo, Capital One, Walgreens, CVS Pharmacy, Levi's, Puma, New Balance, Gap, Old Navy"
                        #default="Land Rover, Jaguar, Ferrari, Lamborghini, Maserati, Aston Martin, Rolls-Royce, Bentley, Cadillac, Chevrolet, GMC, Buick, Dodge, Chrysler, Ram, Alfa Romeo, Fiat, Renault, Peugeot, Citroen, Skoda, Seat, Tata Motors, Mahindra, BYD, Huawei, Xiaomi, Oppo, Vivo, OnePlus, Nokia, BlackBerry, HTC, Philips, Bosch, Siemens, GE, Whirlpool, Electrolux, Miele, Haier, Hisense, Sharp, Toshiba, Sanyo, Canon, Nikon, Fujifilm, Olympus, Goodyear, Michelin, Bridgestone, Pirelli, Red Bull, Monster Energy, Dr Pepper, Gatorade, Snapple, Lipton, Ben & Jerry's, Haagen-Dazs, Baskin-Robbins, Krispy Kreme, Taco Bell, Wendy's, Chick-fil-A, Papa John's, Pizza Hut, Little Caesars, Panda Express, Jack in the Box, Shake Shack, Raising Cane's, Dairy Queen, Arby's, Jersey Mike's, Jimmy John's, Victoria's Secret, Bath & Body Works, The Body Shop, LOreal, Estee Lauder, Clinique, Maybelline, Gillette, Colgate, Crest, Dove, Axe, Pampers, Huggies, Kleenex, Scott, Lysol, Clorox, Scotch, 3M, Caterpillar, John Deere, Royal Caribbean"
                        #default="Acrobat, Lightroom, Costco, Sam's Club, Trader Joe's, Whole Foods Market, Kroger, Aldi, Publix, Macy's, Nordstrom, Kohl's, HomeGoods, Petco, PetSmart, Rite Aid, Fenty Beauty, Aveda, Morphe, ColourPop, Columbia Sportswear, The North Face, Calvin Klein, Tommy Hilfiger, Skechers, Hershey's, Nestle, Kraft Heinz, Stripe, Square, Airbnb, TripAdvisor"
                        default="Photoshop, Lightroom, Creative Cloud, Acrobat"
                        )
    parser.add_argument("--personas", type=str,
                        default="Pride,Anticipation,Fear,Joy,Trust")
    parser.add_argument("--generation_type", choices=["headline", "slogan"], default="headline")
    parser.add_argument("--num_generations", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--out_file", type=str, default="dataset_slogans.csv")
    parser.add_argument("--output_dir", type=str, default="output5", help="Directory to save output files")
    args = parser.parse_args()

    # ---------------- 确保输出目录存在 ---------------- #
    os.makedirs(args.output_dir, exist_ok=True)

    # ---------------- 模型加载 ---------------- #
    device_map = {"": rank}   # 每个进程把整模放到自己的 GPU
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [rank {rank}] Loading model …")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id_or_path,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_cache=True,  # 确保启用KV缓存
    )
    model.eval()

    if hasattr(torch, 'compile'):
        model = torch.compile(model, mode="reduce-overhead")

    # ---------------- 任务切片 ---------------- #
    all_brands   = [b.strip() for b in args.brands.split(",") if b.strip()]
    all_personas = [p.strip() for p in args.personas.split(",") if p.strip()]
    tasks = [(b, p) for b in all_brands for p in all_personas]     # 笛卡儿积
    my_tasks = tasks[rank::world_size]                             # 轮询切片

    # ---------------- CSV & 日志 ---------------- #
    stem, ext = os.path.splitext(args.out_file)
    out_file = os.path.join(args.output_dir, f"{stem}_rank{rank}{ext}")
    log_file_path = os.path.join(args.output_dir, f"generation_log_rank{rank}.txt")
    header_needed = not os.path.exists(out_file)

    with open(out_file, "a", encoding="utf-8", newline="") as f_csv, \
         open(log_file_path, "a", encoding="utf-8") as f_log:
        writer = csv.writer(f_csv)
        if header_needed:
            writer.writerow(["brand", "persona", args.generation_type, "quote", "explanation_for_best_quote"])  # 移除logp列
            f_csv.flush()

        total = 0
        for brand, persona in my_tasks:
            print(f"[{datetime.now():%H:%M:%S}][rank {rank}] {brand} | {persona} → {args.num_generations}×{args.generation_type}")
            for _ in range(args.num_generations):
                results = generate_for_one_pair(
                    tokenizer, model,
                    brand, persona, args.generation_type,
                    1, args.temperature, args.top_p,
                    log_file_handle=f_log,
                )
                if results and results[0][0]:
                    text, quote, explanation = results[0]  # 移除logp解包
                    writer.writerow([brand, persona, text, quote, explanation])  # 移除logp写入
                    f_csv.flush()
                    total += 1

    print(f"[rank {rank}] Done. {total} rows written to {out_file}")
    print(f"[rank {rank}] Raw generations in {log_file_path}")


if __name__ == "__main__":
    main()