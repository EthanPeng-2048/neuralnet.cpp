"""下载 Project Gutenberg 公版书并合并为训练数据集"""
import urllib.request
import re
import os

# (URL, 文件名)
BOOKS = [
    ("https://www.gutenberg.org/cache/epub/84/pg84.txt",   "frankenstein.txt"),
    ("https://www.gutenberg.org/cache/epub/1342/pg1342.txt", "pride.txt"),
    ("https://www.gutenberg.org/cache/epub/11/pg11.txt",    "alice.txt"),
    ("https://www.gutenberg.org/cache/epub/1661/pg1661.txt", "sherlock.txt"),
    ("https://www.gutenberg.org/cache/epub/219/pg219.txt",   "heart_of_darkness.txt"),
    ("https://www.gutenberg.org/cache/epub/98/pg98.txt",     "tale_of_two_cities.txt"),
    ("https://www.gutenberg.org/cache/epub/74/pg74.txt",     "tom_sawyer.txt"),
    ("https://www.gutenberg.org/cache/epub/76/pg76.txt",     "huck_finn.txt"),
    ("https://www.gutenberg.org/cache/epub/1952/pg1952.txt", "yellow_wallpaper.txt"),
    ("https://www.gutenberg.org/cache/epub/345/pg345.txt",   "dracula.txt"),
]

RAW_DIR = os.path.join(os.path.dirname(__file__), "datasets", "books")
OUT_FILE = os.path.join(os.path.dirname(__file__), "dataset.txt")

GUTENBERG_START = re.compile(
    r"\*\*\*\s*START OF TH[IE]S? PROJECT GUTENBERG EBOOK.*?\*\*\*",
    re.IGNORECASE | re.DOTALL,
)
GUTENBERG_END = re.compile(
    r"\*\*\*\s*END OF TH[IE]S? PROJECT GUTENBERG EBOOK.*",
    re.IGNORECASE | re.DOTALL,
)

def strip_gutenberg(text: str) -> str:
    m_start = GUTENBERG_START.search(text)
    m_end = GUTENBERG_END.search(text)
    if m_start and m_end:
        text = text[m_start.end():m_end.start()]
    elif m_start:
        text = text[m_start.end():]
    # 去掉制表符/回车，去掉所有空行
    text = re.sub(r"[\t\r]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip().lower()

def download_book(url: str, filename: str) -> str | None:
    filepath = os.path.join(RAW_DIR, filename)
    if os.path.exists(filepath):
        print(f"  [跳过] {filename} 已存在")
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        text = re.sub(r"[\t\r]+", " ", text)
        text = re.sub(r"\n{2,}", "\n", text)
        return text.strip().lower()
    print(f"  [下载] {filename} ...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        cleaned = strip_gutenberg(raw)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(cleaned)
        return cleaned
    except Exception as e:
        print(f"  [失败] {filename}: {e}")
        return None

def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    all_texts = []
    for url, fname in BOOKS:
        text = download_book(url, fname)
        if text:
            all_texts.append(text)
            print(f"    → {len(text):,} 字符")

    combined = "\n\n".join(all_texts)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write(combined)
    print(f"\n合并完成: {OUT_FILE}")
    print(f"总计 {len(combined):,} 字符, {len(combined.split()):,} 词")

if __name__ == "__main__":
    main()
