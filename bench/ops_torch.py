# ── ops_torch.py — PyTorch CUDA 算子基准（对照 neuralnet.cpp layer_bench --op）──
#
# 统一尺寸、CUDA event 计时（纯 kernel，不含 Python/框架延迟）、大矩阵稀释
# 启动开销、warmup≥20 防时钟爬坡、多轮取 best（与 layer_bench 的 best-of-iter
# 口径一致）。
#
# 尺寸契约（与 src/layer_bench.cpp 的 OpSpec 一一对应）：
#   matmul       A(m,k) @ B(k,n)          — layer_bench: a=(m,k), b=(k,n)
#   matmul_bt    A(m,k) @ B(n,k)^T        — b 存 (n,k)，transB
#   matmul_at    A(k,m)^T @ B(k,n)        — a 存 (k,m)，transA
#   batched      (B,m,k) bmm (B,k,n)      — layer_bench: a=(batch*m,k), b=(batch*k,n)
#   add          a += b                   — in-place, 3N 字节
#   exp          c = exp(a)               — 出点, 2N 字节
#   broadcast_col  a += row(1,n)          — in-place, 2N+n
#   row_reduce   sum(dim=1) → (m,)        — N+m 字节（记账与 layer_bench 一致）
#   col_reduce   sum(dim=0) → (n,)        — N+n 字节
#   transpose    c = a^T（物化）           — 2N 字节
#   scale        a *= 0.5                 — in-place, 2N 字节
# 全部 f32（layer_bench 的 Matrix=Scalar=float → f32 张量）。
#
# 用法: python ops_torch.py [--rounds 3] [--warmup 20] [--iter 50]
#        [--out results.json]
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import json
import os

import torch

DTYPE = torch.float32


def ms_best(fn, warmup: int, iters: int) -> float:
    """warmup 后 CUDA event 计时 iters 次，返回 best（与 layer_bench 口径一致）。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end))
    return best


def make(shape, seed=1234):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(*shape, generator=g, device="cuda", dtype=DTYPE)


def bench_case(name, setup, run, work, is_flops, warmup, iters):
    """setup() → ctx; run(ctx) → 输出张量; work = FLOPs 或字节。"""
    ctx = setup()
    t = ms_best(lambda: run(ctx), warmup, iters)
    unit_val = work / (t / 1e3) / 1e9
    unit = "GFLOPS" if is_flops else "GB/s"
    return {
        "op": name, "ms": t, "unit": unit, "value": unit_val,
        "ms_str": f"{t:8.3f} ms  {unit_val:10.1f} {unit}",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iter", type=int, default=50)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    assert torch.cuda.is_available()
    dev_name = torch.cuda.get_device_name(0)
    print(f"=== ops_torch (torch CUDA, {dev_name}, fp32) ===")

    results = []

    def report(r):
        print(f"  {r['op']:<16}: {r['ms_str']}")
        results.append(r)

    # ── 形状组（与 layer_bench 调用参数一致，见 bench/run_ops.ps1）────────
    # 组1: 方阵1024（matmul 系 + 小逐元素）
    # 组2: 方阵2048
    # 组3: 方阵4096（matmul 系 + 大逐元素/归约/转置/广播）
    # 组4: 8192²（仅逐元素族，大矩阵）
    # 组5: LM head 形状 (16384,64)@(64,8208)
    # 组6: batched
    cases = []

    def matmul_family(m, n, k, prefix=""):
        def s_ab():
            return {"a": make((m, k)), "b": make((k, n))}

        def s_bt():
            return {"a": make((m, k)), "b": make((n, k))}

        def s_at():
            return {"a": make((k, m)), "b": make((k, n))}

        flops = 2.0 * m * n * k
        cases.append((f"{prefix}matmul[{m}x{n}x{k}]", s_ab,
                      lambda c: c["a"] @ c["b"], flops, True))
        cases.append((f"{prefix}matmul_bt[{m}x{n}x{k}]", s_bt,
                      lambda c: c["a"] @ c["b"].transpose(0, 1), flops, True))
        cases.append((f"{prefix}matmul_at[{m}x{n}x{k}]", s_at,
                      lambda c: c["a"].transpose(0, 1) @ c["b"], flops, True))

    def elem_family(m, n, prefix=""):
        N = m * n

        def s2():
            return {"a": make((m, n)), "b": make((m, n))}

        def s1():
            return {"a": make((m, n))}

        def srow():
            return {"a": make((m, n)), "r": make((1, n))}

        cases.append((f"{prefix}add_inplace[{m}x{n}]", s2,
                      lambda c: c["a"].add_(c["b"]), 3.0 * N * 4, False))
        cases.append((f"{prefix}elementwise_exp[{m}x{n}]", s1,
                      lambda c: c["a"].exp(), 2.0 * N * 4, False))
        cases.append((f"{prefix}broadcast_col[{m}x{n}]", srow,
                      lambda c: c["a"].add_(c["r"]), (2.0 * N + n) * 4, False))
        cases.append((f"{prefix}row_reduce_sum[{m}x{n}]", s1,
                      lambda c: c["a"].sum(dim=1), (N + m) * 4, False))
        cases.append((f"{prefix}col_reduce_sum[{m}x{n}]", s1,
                      lambda c: c["a"].sum(dim=0), (N + n) * 4, False))
        cases.append((f"{prefix}transpose[{m}x{n}]", s1,
                      lambda c: c["a"].t().contiguous(), 2.0 * N * 4, False))
        cases.append((f"{prefix}scale_inplace[{m}x{n}]", s1,
                      lambda c: c["a"].mul_(0.5), 2.0 * N * 4, False))

    matmul_family(1024, 1024, 1024)
    elem_family(1024, 1024)
    matmul_family(2048, 2048, 2048)
    elem_family(2048, 2048)
    matmul_family(4096, 4096, 4096)
    elem_family(4096, 4096)
    elem_family(8192, 8192)
    # LM head: A(16384,64) @ B(64,8208)
    matmul_family(16384, 8208, 64, prefix="lm_head_")

    # batched: layer_bench a=(batch*m,k) → 语义 (B,m,k) bmm (B,k,n)
    def bat_setup(B, m, n, k):
        def s():
            return {"a": make((B, m, k)), "b": make((B, k, n))}
        return s

    for (B, m, n, k) in [(64, 256, 256, 256), (512, 64, 64, 64), (4096, 64, 64, 64)]:
        flops = 2.0 * B * m * n * k
        cases.append((f"batched_matmul[B={B},{m}x{n}x{k}]",
                      bat_setup(B, m, n, k),
                      lambda c: torch.bmm(c["a"], c["b"]), flops, True))

    # ── 多轮执行：每 case 独立多轮，报告每轮 + 取 best ──────────────────
    out_all = []
    for name, setup, run, work, is_flops in cases:
        rounds = []
        for rd in range(args.rounds):
            r = bench_case(name, setup, run, work, is_flops, args.warmup, args.iter)
            rounds.append(r["ms"])
        best = min(rounds)
        unit = "GFLOPS" if is_flops else "GB/s"
        val = work / (best / 1e3) / 1e9
        line = (f"  {name:<34}: rounds(ms)="
                + "/".join(f"{x:.3f}" for x in rounds)
                + f"  best={best:8.3f} ms  {val:10.1f} {unit}")
        print(line)
        out_all.append({"op": name, "rounds_ms": rounds, "best_ms": best,
                        "unit": unit, "value": val, "work": work,
                        "is_flops": is_flops})
        results.clear()

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"device": dev_name, "dtype": "fp32",
                       "rounds": args.rounds, "warmup": args.warmup,
                       "iter": args.iter, "cases": out_all}, f, indent=2)
        print(f"结果已写入 {args.out}")


if __name__ == "__main__":
    main()
