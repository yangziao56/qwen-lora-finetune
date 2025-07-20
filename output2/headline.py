import argparse
import csv
import random
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Extract one column from multiple CSVs, shuffle and write out"
    )
    parser.add_argument(
        "--input_dir", type=str, default="output2",
        help="目录，里面包含所有 rank*.csv 文件"
    )
    parser.add_argument(
        "--column_name", type=str, default="headline",
        help="要抽取的列名（脚本里使用的 header）"
    )
    parser.add_argument(
        "--output_file", type=str, default="all_headlines_shuffled.csv",
        help="输出合并并打乱后的 CSV 文件名"
    )
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    csv_files = sorted(input_path.glob("*.csv"))

    items = []
    for file in csv_files:
        with file.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header or args.column_name not in header:
                continue
            idx = header.index(args.column_name)
            for row in reader:
                if len(row) > idx and row[idx].strip():
                    items.append(row[idx].strip())

    random.shuffle(items)

    with open(args.output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([args.column_name])
        for v in items:
            writer.writerow([v])

    print(f"✔ 抽取并打乱完成，共 {len(items)} 条写入 {args.output_file}")

if __name__ == "__main__":
    main()