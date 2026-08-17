# ── _diag_entropy.py — 从 .tokcache 计算 unigram/next-token 经验分布 ──
#
# 目的：判断"loss 卡 8.5"是否 = 模型只学到了 unigram 边际分布（即没学到上下文）。
#   纯 unigram 预测的 loss = H(X)（经验熵，nats）
#   如果 H ≈ 8.5，说明 8.5 恰好是"不利用上下文"的理论下限 → 指向掩码/架构/数据对齐问题
#   如果 H 明显 < 8.5，说明模型甚至没学到边际分布 → 指向优化/初始化问题
#
# 用法: .venv\\Scripts\\python.exe compare_with_torch\\_diag_entropy.py
import os
import struct
import math
from collections import Counter

CACHE = "datasets/tinystories.txt.tokcache"


def load_toks(path, max_n=60_000_000):
    with open(path, "rb") as f:
        raw = f.read(64)
    magic = raw[:4]
    ver, sz = struct.unpack("<II", raw[4:12])
    # C++ 结构体在 MSVC 下 uint64_t 按 8 字节对齐：
    #   magic(4) + version(4) + sizeof_size_t(4) + pad(4) + text_sz(8) + vocab_sz(8) + n(8)
    text_sz, vocab_sz, n = struct.unpack("<QQQ", raw[16:40])
    print(f"magic={magic} ver={ver} sizeof_size_t={sz} text_sz={text_sz} vocab_sz={vocab_sz} n={n}")
    n = min(n, max_n)
    with open(path, "rb") as f:
        f.seek(40)
        toks = struct.unpack("<%dQ" % n, f.read(n * 8))
    return toks


def main():
    toks = load_toks(CACHE)
    N = len(toks)
    print(f"loaded {N} tokens, max_id={max(toks)}")

    # ── 1) unigram 分布 → 纯 unigram loss ──
    c = Counter(toks)
    H_u = -sum((v / N) * math.log(v / N) for v in c.values())
    print(f"\n[unigram] 经验熵 H(X) = {H_u:.4f} nats  (= 纯边际预测的期望 loss)")
    print(f"[uniform ] 均匀随机 loss = {math.log(len(c)):.4f} nats")

    # ── 2) bigram 条件熵 H(X_t | X_{t-1})：上下文有一点点用的话应低于 H_u ──
    # 连续窗口内 (x_t, x_{t+1}) 对。注意窗口/行边界会造成少量伪 bigram，可接受。
    trans = {}
    n_pairs = 0
    for a, b in zip(toks[:-1], toks[1:]):
        d = trans.get(a)
        if d is None:
            trans[a] = {b: 1}
        else:
            d[b] = d.get(b, 0) + 1
        n_pairs += 1
    # 条件熵 = -Σ p(a,b) log p(b|a)
    H_c = 0.0
    for a, d in trans.items():
        pa = c[a] / N
        na = sum(d.values())
        for b, cnt in d.items():
            H_c -= pa * (cnt / na) * math.log(cnt / na)
    print(f"[bigram ] 条件熵 H(Xt|Xprev) = {H_c:.4f} nats (pairs={n_pairs})")
    print(f"[信息量 ] H_u - H_c = {H_u - H_c:.4f} nats  (每 token 上下文能提供的信息)")

    # ── 3) 前 K 高频 token 占比 ──
    top = c.most_common(20)
    print("\n[top20 tokens] id: count, freq, (字节解码若可用)")
    for tid, cnt in top:
        print(f"  {tid:6d}: {cnt:8d}  freq={cnt/N:.5f}")

    # ── 4) 直接估计"总是预测最频繁 token"的 loss ──
    top_id, top_cnt = c.most_common(1)[0]
    loss_top = -math.log(top_cnt / N)
    print(f"\n[always-top1] 预测最频繁 token {top_id} 的 loss = {loss_top:.4f}")

    # ── 5) 一个粗的"真实可实现"下界参考：条件熵基于更长上下文的粗略近似 ──
    # 用 k-gram 频率近似 H(X_t | X_{t-k:t-1})，k=3 快速估计
    from collections import defaultdict
    trigram = defaultdict(Counter)
    n_tri = 0
    for i in range(N - 3):
        key = (toks[i], toks[i + 1])
        trigram[key][toks[i + 2]] += 1
        n_tri += 1
    H_tri = 0.0
    for key, d in trigram.items():
        cnt_key = sum(d.values())
        for b, cnt in d.items():
            H_tri -= (cnt_key / n_tri) * (cnt / cnt_key) * math.log(cnt / cnt_key)
    print(f"[trigram] 条件熵 H(X_t|X_{t-2:t}) = {H_tri:.4f} nats")
    print(f"[结论参考] 正常 GPT 训练应从 ~H_u({H_u:.2f}) 快速降到远低于 H_u；"
          f"若一直停在 H_u 附近 = 没学到上下文")


if __name__ == "__main__":
    main()
