"""
将分词器 JSON 词表转为纯文本词表（每行一个词）。
跳过 ID 0~259 的特殊标记和单字节 token。

用法: python vocab_to_txt.py <json> [-o output.txt]
"""
import argparse, json, sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="JSON 词表 → 纯文本词表")
    ap.add_argument("json", help="词表 JSON 路径")
    ap.add_argument("-o", "--output", help="输出路径 (默认: <json名>.txt)")
    args = ap.parse_args()

    data = json.loads(Path(args.json).read_text(encoding="utf-8"))
    vocab = data.get("vocab", {})
    byte_offset = data.get("byte_offset", 4)

    out_path = args.output or str(Path(args.json).with_suffix(".txt"))
    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for tid_str, hex_str in vocab.items():
            tid = int(tid_str)
            if tid < byte_offset + 256:
                continue
            word = bytes.fromhex(hex_str).decode("utf-8", errors="replace")
            f.write(word + "\n")
            count += 1

    print(f"已写入 {count} 个词条 → {out_path}")


if __name__ == "__main__":
    main()
