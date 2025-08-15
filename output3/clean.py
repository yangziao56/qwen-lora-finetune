import re
import pandas as pd
from pathlib import Path

# 绝对路径修改：
src = Path("/sensei-fs-3/users/ziaoy/llm/qwen/output3/merged_headlines.csv")
dst = Path("/sensei-fs-3/users/ziaoy/llm/qwen/output3/merged_headlines_cleaned.csv")

df = pd.read_csv(src)

# 预编译正则
RE_BOLD   = re.compile(r"\*\*(.*?)\*\*")
RE_ITALIC = re.compile(r"\*(?!\s)(.*?)\*(?!\s)")
RE_CODE   = re.compile(r"`([^`]+)`")
RE_LINK   = re.compile(r"\[([^\]]+)\]\([^)]+\)")
RE_HASH   = re.compile(r"^\s*#+\s*")
DROP_PAT  = re.compile(r"(Step\s*\d|Quote\s*Matching)", re.I)

def clean(text: str):
    if pd.isna(text):
        return None

    # 1) 行内含换行符或关键词 => 整行丢弃
    if "\n" in text or DROP_PAT.search(text):
        return None

    # 1.5) 行内含管道符号 '|' => 整行丢弃
    if "|" in text:
        return None

    # 2) 去 Markdown 标记
    text = RE_BOLD.sub(r"\1", text)
    text = RE_ITALIC.sub(r"\1", text)
    text = RE_CODE.sub(r"\1", text)
    text = RE_LINK.sub(r"\1", text)
    text = RE_HASH.sub("", text)

    # 3) 删除所有引号
    text = re.sub(r"[\"'“”‘’]", "", text)

    # 4) 压缩多余空格
    text = re.sub(r"\s+", " ", text).strip()

    return text or None

# 执行清洗
df["headline"] = df["headline"].apply(clean)
df = df.dropna(subset=["headline"]).reset_index(drop=True)

# 保存结果
df.to_csv(dst, index=False, encoding="utf-8")
print(f"✓ 已清洗完成：{len(df)} 条标题保存至 {dst}")
