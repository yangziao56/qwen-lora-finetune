import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def main():
    parser = argparse.ArgumentParser(description="使用指定模型进行推理")
    parser.add_argument(
        "--model_id_or_path",
        type=str,
        default="/sensei-fs-3/users/ziaoy/llm/qwen/checkpoints/QwQ-32B-lora-ds/final-20250620-140034",
        #default="Qwen/QwQ-32B",
        #default="Qwen/Qwen2-7B-Instruct",
        #default="/sensei-fs-3/users/ziaoy/llm/qwen/checkpoints/Qwen2-7B-Instruct-lora-singlegpu/final",
        help="要加载的Hugging Face模型ID或本地模型路径, for example: Qwen/Qwen2-7B-Instruct, Qwen/QwQ-32B-lora-ds/final, ",
    )
    args = parser.parse_args()

    model_id = args.model_id_or_path
    print(f"正在加载模型: {model_id}")

    quant_config = BitsAndBytesConfig(load_in_8bit=True)

    # 1. 检查有几张可用 GPU
    ngpu = torch.cuda.device_count()

    # 2. 根据 GPU 数量选择 device_map
    if ngpu > 1:
        # 多卡：让 HF 自动拆分（启用张量并行）
        device_map = "auto"
    else:
        # 单卡：把模型全部放到 GPU:0
        device_map = {"": 0}

    # 3. 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # 4. 从预训练权重加载模型
    #    ─ from_pretrained 方法既能接受Hugging Face模型ID，也能接受本地路径
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        #torch_dtype=torch.float16,
        device_map=device_map,
        trust_remote_code=True,
        quantization_config=quant_config
    )

    #5. 构造一次示例对话
    system = "Generate slogans for certain product."
    user   = "Please generate slogan for coca cola. using different styles. promote happiness and refreshment."
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user}
    ]

    # # 1) define personas in the required order
    # personas = ["Pride", "Anticipation", "Fear", "Joy", "Trust"]

    # # 2) system & user prompts
    # system = "You are an award-winning advertising copywriter."

    # user = (
    #     "Generate five distinct Coca-Cola slogans—one for each persona listed below—"
    #     "that all convey happiness and refreshment.\n"
    #     "Personas (use this order): Pride, Anticipation, Fear, Joy, Trust.\n"
    #     "Return each slogan in the format “<Persona>: <Slogan>”.\n"
    #     "Persona guidelines:\n"
    #     "• Pride: celebrate achievement or exclusivity\n"
    #     "• Anticipation: spark curiosity or suspense\n"
    #     "• Fear: neutralize fear of missing out or add urgency\n"
    #     "• Joy: evoke delight or excitement\n"
    #     "• Trust: emphasize safety or reliability\n"
    #     "Keep every slogan under eight words, and vary the linguistic style."
    # )

    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user}
    ]



    # 6. 构造 prompt 并转张量
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 7. 生成回答
    outputs = model.generate(
        **inputs,
        max_new_tokens=3000,
        do_sample=True,         # 启用随机采样
        temperature=0.6,        # 温度设置
        top_k=50,               # top-k 采样
        top_p=0.6,              # top-p 采样
        repetition_penalty=1.1, # 防止重复
        pad_token_id=tokenizer.eos_token_id  # 视情况而定，加上这行避免警告
    )
    generated = outputs[0][ inputs.input_ids.shape[-1] : ]
    reply = tokenizer.decode(generated, skip_special_tokens=True)

    print("Output：", reply)

if __name__ == "__main__":
    main()
