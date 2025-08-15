import pandas as pd
import glob

# 1. 找到所有满足模式的 CSV 文件（绝对路径）
csv_files = sorted(glob.glob("/sensei-fs-3/users/ziaoy/llm/qwen/output3/dataset_slogans_rank*.csv"))

# 2. 读取并只保留 headline 列，然后合并
merged = pd.concat(
    [pd.read_csv(f, usecols=["headline"]) for f in csv_files],
    ignore_index=True
)

# 3.（可选）去重
# merged = merged.drop_duplicates(subset="headline")

# 4. 保存结果到绝对路径
merged.to_csv("/sensei-fs-3/users/ziaoy/llm/qwen/output3/merged_headlines.csv", index=False)

print(f"Merged {len(csv_files)} files; output saved to /sensei-fs-3/users/ziaoy/llm/qwen/output3/merged_headlines.csv with {len(merged)} rows.")
