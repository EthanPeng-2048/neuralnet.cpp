# ── _diag_tok.py — 诊断 bytebpe.json (由 scripts/train_bytebpe.py 转换生成) ──
#
# 目的：验证 train_bytebpe.py 产出的 char_bpe 格式 tokenizer 是否正确，
#       以及它是否可能是"loss 卡 8.5"的根源。
import json
import sys
import os
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tokenizer import load_charbpe_from_file


def main():
    t = load_charbpe_from_file("bytebpe.json")
    assert t is not None, "bytebpe.json load failed"
    print("vocab_size =", t.vocab_size())
    print("bos/eos/pad =", t.bos_id(), t.eos_id(), t.PAD_ID)
    print("n_merges =", len(t._merges))

    # ── 1) 英文编码：检查 UNK、round-trip ──
    s = ("Once upon a time there was a little girl named Lily who loved to read "
         "stories. She had a small dog called Max.")
    ids = t.encode(s)
    print("\n[en] tokens =", len(ids), " UNK =", ids.count(t.UNK_ID),
          " max_id =", max(ids))
    print("[en] decode_ok =", t.decode(ids) == s)

    # ── 2) 中文编码 ──
    zh = "人工智能是未来的方向，深度学习让机器更聪明。"
    ids2 = t.encode(zh)
    print("[zh] tokens =", len(ids2), " UNK =", ids2.count(t.UNK_ID))
    print("[zh] decode =", t.decode(ids2))

    # ── 3) merge 完整性：merged 的字节必须 = 两部分的字节拼接 ──
    vocab = t._vocab
    bad = 0
    for m in t._merges:
        a, b, new = m[0], m[1], m[2]
        ba = vocab.get(a, b"")
        bb = vocab.get(b, b"")
        bn = vocab.get(new, b"")
        if bn != ba + bb:
            bad += 1
    print("\n[merge] bad(不一致) =", bad, "/", len(t._merges))

    # ── 4) 英文 token 字节长度分布 ──
    c = Counter(len(vocab.get(i, b"")) for i in ids)
    print("[en] token byte-len dist =", dict(sorted(c.items())))

    # ── 5) 与 C++ .tokcache 对比：token 流最大 ID / 分布 ──
    cache_path = "datasets/tinystories.txt.tokcache"
    if os.path.exists(cache_path):
        import struct
        with open(cache_path, "rb") as f:
            raw = f.read(64)
        print("\n[cache] raw header hex:", raw[:48].hex())
        magic = raw[:4]
        ver, sz = struct.unpack("<II", raw[4:12])
        text_sz, vocab_sz, n = struct.unpack("<QQQ", raw[12:36])
        print("[cache] magic=", magic, "ver=", ver, "sizeof_size_t=", sz,
              "text_sz=", text_sz, "vocab_file_sz=", vocab_sz, "n=", n)
        with open(cache_path, "rb") as f:
            f.seek(36)
            toks = struct.unpack("<%dQ" % n, f.read(n * 8))
        print("[cache] n_tokens =", n, " max_id =", max(toks))
        cc = Counter(toks)
        print("[cache] top10 ids =", cc.most_common(10))
        print("[cache] id>=vocab_size count =",
              sum(v for k, v in cc.items() if k >= t.vocab_size()))


if __name__ == "__main__":
    main()
