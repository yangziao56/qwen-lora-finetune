import os
import sys
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def main():
    parser = argparse.ArgumentParser(description="Perform inference with a specified model")
    parser.add_argument(
        "--model_id_or_path",
        type=str,
        default="/sensei-fs-3/users/ziaoy/llm/qwen/checkpoints/QwQ-32B-lora-ds/final-20250627-151621",
        help="Hugging Face model ID or local path to load"
    )
    # 1) Add a persona argument; choose one of the predefined personas
    personas = ["Pride", "Anticipation", "Fear", "Joy", "Trust"]
    parser.add_argument(
        "--persona",
        type=str,
        choices=personas,
        default="Joy",
        help="Persona for slogan generation"
    )
    args = parser.parse_args()

    model_id = args.model_id_or_path
    persona = args.persona

    # --- Create output directory, build filename from model name, redirect stdout ---
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    model_name = os.path.basename(model_id)
    output_path = os.path.join(output_dir, f"{model_name}_output.txt")
    sys.stdout = open(output_path, "a", encoding="utf-8")
    # ---------------------------------------------------------------------------

    print(f"Loading model: {model_id}, target persona: {persona}")

    quant_config = BitsAndBytesConfig(load_in_8bit=True)

    # 1. Check how many GPUs are available
    ngpu = torch.cuda.device_count()

    # 2. Select device_map depending on the number of GPUs
    if ngpu > 1:
        # Multi-GPU: allow HF to auto-split (tensor parallelism)
        device_map = "auto"
    else:
        # Single GPU: place the entire model on GPU 0
        device_map = {"": 0}

    # 3. Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # 4. Load the model from pretrained weights
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
        trust_remote_code=True,
        quantization_config=quant_config
    )

    # 5. Construct a sample conversation for the selected persona
    system = "You are an award-winning slogan copywriter. Create slogan remixes using classic poetry. Don't change the sentence's structure and pattern, just replace words with synonyms or similar phrases. "
    guidelines = {
        "Pride": "celebrate achievement or exclusivity",
        "Anticipation": "spark curiosity or suspense",
        "Fear": "neutralize fear of missing out or add urgency",
        "Joy": "evoke delight or excitement",
        "Trust": "emphasize safety or reliability",
    }
    # Clarify this is a persona type, not a person’s name
    user = (
        f"Generate Coca-Cola slogans targeting the \"{persona}\" persona, "
        f"conveying: {guidelines[persona]}.\n"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user}
    ]

    # 6. Build the prompt and convert it to tensors
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print("Prompt:", prompt)                    # Print the raw prompt text
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    print("Input IDs:", inputs.input_ids)       # Print the tokenized input IDs

    # 7. Generate the model’s response
    outputs = model.generate(
        **inputs,
        max_new_tokens=10000,
        do_sample=True,         # enable sampling
        temperature=1.0,        # sampling temperature
        top_k=50,               # top-k sampling
        top_p=1.0,              # top-p sampling
        repetition_penalty=1.0, # prevent repetition
        pad_token_id=tokenizer.eos_token_id  # avoid warnings
    )
    generated = outputs[0][inputs.input_ids.shape[-1]:]
    reply = tokenizer.decode(generated, skip_special_tokens=True)

    print("Output:", reply)

if __name__ == "__main__":
    main()
