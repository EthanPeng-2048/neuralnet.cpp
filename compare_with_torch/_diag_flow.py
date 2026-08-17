# ── _diag_flow.py — 验证 .tokcache token 流 ↔ bytebpe.json 的一致性 ──
# 目的：检查 token 流是否真的是用 bytebpe.json 编码的（能 round-trip 解码），
#       以及 BOS/EOS 实际 ID 是 charBPE(2/3) 还是 BBPE(256/257)。
import struct, sys, os
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tokenizer import load_charbpe_from_file

CACHE = "datasets/tinystories.txt.tokcache"


def load_toks(path, max_n=2_000_000):
    with open(path, "rb") as f:
        raw = f.read(64)
    text_sz, vocab_sz, n = struct.unpack("<QQQ", raw[16:40])
    n = min(n, max_n)
    with open(path, "rb") as f:
        f.seek(40)
        return struct.unpack("<%dQ" % n, f.read(n * 8))


def main():
    t = load_charbpe_from_file("bytebpe.json")
    assert t is not None
    print(f"bytebpe.json: vocab={t.vocab_size()} bos={t.bos_id()} eos={t.eos_id()} pad={t.PAD_ID}")
    toks = load_toks(CACHE)
    print(f"cache: n={len(toks)} max_id={max(toks)}")

    # ── 1) 统计特殊 ID 出现次数 ──
    c = Counter(toks)
    for special, name in [(t.PAD_ID, "PAD"), (t.UNK_ID, "UNK"), (t.bos_id(), "BOS"), (t.eos_id(), "EOS")]:
        print(f"  id={special} ({name}) count={c.get(special, 0)}")

    # ── 2) round-trip：取一段连续 token 解码成文本，再重新编码，看是否一致 ──
    #    只取一段不含 BOS/EOS 的纯文本区间（跳过行边界附近）
    seg = toks[1000:2000]
    text = t.decode(seg)
    reids = t.encode(text)
    print(f"\n[roundtrip] decode({len(seg)} tok) -> {len(text)} chars -> encode -> {len(reids)} toks")
    print(f"[roundtrip] 一致 = {seg[:min(len(seg), len(reids))] == reids[:min(len(seg), len(reids))]}")
    print(f"[roundtrip] 前 60 tokens 匹配?")
    m = sum(1 for a, b in zip(seg, reids) if a == b)
    print(f"  前{min(len(seg), len(reids))}个中匹配 {m} 个")
    if m < min(len(seg), len(reids)):
        # 找出第一个不一致位置
        for i, (a, b) in enumerate(zip(seg, reids)):
            if a != b:
                print(f"  首个不一致 @pos {i}: cache={a} re-encode={b}")
                print(f"    上下文 cache=[{seg[max(0,i-3):i+4]}] re=[{reids[max(0,i-3):i+4]}]")
                print(f"    decode文本片段: ...{text[max(0,i-10):i+30]!r}...")
                break

    # ── 3) 行边界：找 BOS/EOS 出现时的相邻 ID ──
    if t.eos_id() in c or t.bos_id() in c:
        print(f"\n[boundary] 前20个 BOS 出现的下一位 ID:")
        cnt = 0
        for i, x in enumerate(toks[:-1]):
            if x == t.bos_id():
                print(f"    pos{i}: bos->{toks[i+1]}")
                cnt += 1
                if cnt >= 20:
                    break


if __name__ == "__main__":
    main()
