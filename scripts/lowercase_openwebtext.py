"""将 OpenWebText 转为小写。"""
import os

src = "scripts/datasets/openwebtext_en_100mb.txt"
dst = "scripts/datasets/openwebtext_en_100mb_lower.txt"

print(f"读取: {src}")
with open(src, "r", encoding="utf-8") as f:
    text = f.read()

print(f"已读取 {len(text)} 字节, {text.count(chr(10))} 行")

text_lower = text.lower()

print(f"写入: {dst}")
with open(dst, "w", encoding="utf-8") as f:
    f.write(text_lower)

size = os.path.getsize(dst)
print(f"完成! 大小: {size/1024/1024:.1f} MB")