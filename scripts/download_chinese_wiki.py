"""从中文维基百科 REST API 抓取随机条目摘要，构建中文训练语料。

使用 /api/rest_v1/page/random/summary（限流比 query API 宽松）。
并发请求，提取摘要并按句号分割成独立句子，每行一句流畅的中文书面语。
依赖：仅标准库。
"""
import urllib.request
import urllib.error
import json
import re
import time
import sys
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

REST_URL = "https://zh.wikipedia.org/api/rest_v1/page/random/summary"
USER_AGENT = "neuralnet.cpp-corpus-builder/1.0 (educational research; local use)"
# 基于脚本所在目录的绝对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_FILE = os.path.join(SCRIPT_DIR, "datasets", "chinese_wiki_corpus.txt")

TARGET_LINES = 5000          # 目标句子数
CONCURRENCY = 6              # 并发请求数
MAX_RETRIES = 3              # 单条目重试次数
# 句子长度范围（字符）
MIN_LEN = 12
MAX_LEN = 150

# 中文句末标点
SENT_SPLIT = re.compile(r"[。！？；]")


def fetch_one_summary() -> str | None:
    """请求一篇随机条目的摘要，返回纯文本 extract 或 None。"""
    req = urllib.request.Request(REST_URL, headers={"User-Agent": USER_AGENT})
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data.get("extract") or None
        except urllib.error.HTTPError as e:
            if e.code == 429:
                # 限流：退避
                time.sleep(5 + attempt * 10)
                continue
            return None
        except Exception:
            time.sleep(1 + attempt * 2)
            continue
    return None


def extract_sentences(text: str) -> list[str]:
    """把摘要文本按句末标点分割成独立句子，过滤长度。"""
    if not text:
        return []
    # 去掉换行，合并
    text = text.replace("\n", "").replace("\r", "").strip()
    # 按句末标点分割
    parts = SENT_SPLIT.split(text)
    sentences = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # 跳过含太多数字/英文的（保留纯中文为主）
        chinese = sum(1 for c in p if "\u4e00" <= c <= "\u9fff")
        if chinese < len(p) * 0.5:
            continue
        if len(p) < MIN_LEN or len(p) > MAX_LEN:
            continue
        sentences.append(p)
    return sentences


def worker() -> list[str]:
    """单次任务：抓一篇并返回句子列表。"""
    extract = fetch_one_summary()
    return extract_sentences(extract) if extract else []


def main():
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    seen: set[str] = set()
    seen_lock = threading.Lock()  # 保护 seen 集合的 check-then-add
    total = 0
    requests_made = 0
    start = time.time()

    with open(OUT_FILE, "w", encoding="utf-8", buffering=1) as f:
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            # 批量提交任务
            pending = set()
            batch_size = CONCURRENCY * 4
            while total < TARGET_LINES:
                # 补充待办任务
                while len(pending) < batch_size:
                    pending.add(pool.submit(worker))
                # 取完成的
                done = set()
                try:
                    completed = as_completed(pending, timeout=30)
                except TimeoutError:
                    # 30 秒内没有任何请求完成：继续循环补充任务（单个请求
                    # 含重试+退避可达 ~70s，不能因超时直接崩溃）
                    continue
                for fut in completed:
                    done.add(fut)
                    requests_made += 1
                    try:
                        sents = fut.result()
                    except Exception:
                        sents = []
                    for s in sents:
                        key = s[:40]
                        with seen_lock:
                            if key in seen:
                                continue
                            seen.add(key)
                        f.write(s + "\n")
                        total += 1
                        if total >= TARGET_LINES:
                            break
                    if total >= TARGET_LINES:
                        break
                pending -= done
                elapsed = time.time() - start
                rate = total / elapsed if elapsed > 0 else 0
                print(f"  请求 {requests_made}, 收集 {total}/{TARGET_LINES} 句, "
                      f"{rate:.1f} 句/秒, {elapsed:.0f}s", flush=True)
                if total >= TARGET_LINES:
                    break
                # 礼貌间隔
                time.sleep(0.3)

    elapsed = time.time() - start
    size = os.path.getsize(OUT_FILE)
    print(f"\n完成! 共 {total} 句, {size:,} 字节, 耗时 {elapsed:.0f}s")
    print(f"输出: {OUT_FILE}")


if __name__ == "__main__":
    main()
