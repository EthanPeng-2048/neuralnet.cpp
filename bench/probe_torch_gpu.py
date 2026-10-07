# probe_torch_gpu.py — torch 训练步的 GPU(CUDA) 时间分解（对照 nn 的设备侧 kernel 表）
# 目的：回答"慢在算子还是框架"——本探针给出 torch 侧
#   · 每步 CUDA kernel 总时间（= 设备忙碌，对照 nn 的 GPU busy 98.6 ms/step）
#   · 按 kernel 名聚合的 GPU 时间 top 表（对照 nn 的 fused_mm / fused / layout 三桶）
#   · host(CPU) 时间 vs CUDA 时间 = torch 的框架开销
# 用法: python bench/probe_torch_gpu.py [--steps 60] [--profile-from 20]
import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "compare_with_torch"))
from model import GPTModel  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=60)
p.add_argument("--profile-from", type=int, default=20)
args = p.parse_args()

torch.manual_seed(0)
dev = torch.device("cuda")
# 与 nn text_train 基准同构: d64/heads4/L4/ff256/seq256/batch64, vocab 8208, fp32, Adam
V, D, SEQ, B, L, H, FF = 8208, 64, 256, 64, 4, 4, 256

model = GPTModel(vocab_size=V, d_model=D, seq_len=SEQ, num_heads=H,
                 d_ff=FF, num_layers=L).to(dev)          # fp32（默认）
opt = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-8)
model.train()

flow = np.random.randint(0, V, size=(64 * 400,), dtype=np.int64)
ar = np.arange(SEQ)


def build_batch():
    starts = np.random.randint(0, len(flow) - SEQ - 1, size=B)
    pos = starts[:, None] + ar[None, :]
    x = flow[pos]
    y = flow[pos + 1]
    m = np.ones((B, SEQ), dtype=np.float32)
    return x, y, m


def full_step():
    a, b, c = build_batch()
    _x = torch.from_numpy(a).to(dev)
    _y = torch.from_numpy(b).to(dev)
    _mm = torch.from_numpy(c).to(dev)
    opt.zero_grad(set_to_none=True)
    _, l = model(_x, targets=_y, loss_mask=_mm)
    l.backward()
    opt.step()


# 预热（分配器 + 时钟）
for _ in range(args.steps):
    full_step()
torch.cuda.synchronize()

# 基线墙钟（不带 profiler 开销）
t0 = time.perf_counter()
for _ in range(48):
    full_step()
torch.cuda.synchronize()
wall_ms = (time.perf_counter() - t0) / 48 * 1e3
print(f"full step 墙钟（无 profiler）: {wall_ms:.3f} ms/step")

# ── 设备流时间（CUDA event 包住整段，= GPU 串行执行这些步的净时间）──
# 对照 nn 的 timestamp 口径「设备 kernel 98.6 ms/step / GPU busy 89.8%」
ev0 = torch.cuda.Event(enable_timing=True)
ev1 = torch.cuda.Event(enable_timing=True)
for _ in range(10):
    full_step()
torch.cuda.synchronize()
ev0.record()
for _ in range(48):
    full_step()
ev1.record()
torch.cuda.synchronize()
stream_ms = ev0.elapsed_time(ev1) / 48
print(f"设备流时间（GPU busy 口径）  : {stream_ms:.3f} ms/step  "
      f"(设备忙碌率 ≈ {100.0*stream_ms/wall_ms:.1f}%)")


# ── 分段分解（每段独立 warmup + best-of-50，CUDA event 纯设备计时）──
def seg_time(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a = torch.cuda.Event(enable_timing=True)
    b = torch.cuda.Event(enable_timing=True)
    best = float("inf")
    for _ in range(iters):
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        best = min(best, a.elapsed_time(b))
    return best


def seg_h2d():
    a2, b2, c2 = build_batch()
    _x = torch.from_numpy(a2).to(dev)
    _y = torch.from_numpy(b2).to(dev)
    _m = torch.from_numpy(c2).to(dev)
    return _x, _y, _m


x0, y0, m0 = build_batch()
xb = torch.from_numpy(x0).to(dev)
yb = torch.from_numpy(y0).to(dev)
mb = torch.from_numpy(m0).to(dev)

t_h2d = seg_time(seg_h2d)


def fwd_only():
    with torch.no_grad():
        model(xb)

t_fwd = seg_time(fwd_only)


def fwd_loss():
    with torch.no_grad():
        _, _l = model(xb, targets=yb, loss_mask=mb)
    return _l

t_fwd_loss = seg_time(fwd_loss)


def fwd_bwd():
    opt.zero_grad(set_to_none=True)
    _, _l = model(xb, targets=yb, loss_mask=mb)
    _l.backward()

t_fb = seg_time(fwd_bwd)
t_adam = seg_time(lambda: opt.step())
t_zero = seg_time(lambda: opt.zero_grad(set_to_none=True))

print("\n分段（CUDA event，best-of-50，fp32）:")
print(f"  H2D 含构 batch        : {t_h2d:8.3f} ms")
print(f"  forward (no loss)     : {t_fwd:8.3f} ms")
print(f"  forward + CE loss     : {t_fwd_loss:8.3f} ms")
print(f"  zero+fwd+loss+bwd     : {t_fb:8.3f} ms")
print(f"  Adam step             : {t_adam:8.3f} ms")
print(f"  zero_grad             : {t_zero:8.3f} ms")
print(f"  ⇒ backward ≈ {t_fb - t_fwd_loss:.3f} ms, CE ≈ {t_fwd_loss - t_fwd:.3f} ms")

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for i in range(args.steps):
        full_step()
    torch.cuda.synchronize()

evt = prof.key_averages()
rows = []
cpu_total = 0.0
cuda_total = 0.0
for e in evt:
    ct = e.self_cpu_time_total / 1e3      # ms
    gt = getattr(e, "self_cuda_time_total", 0) / 1e3
    if e.key == "[cuda]":
        cuda_total = gt
    if e.key == "[_cpu]":
        cpu_total = ct
    if gt > 0 and not e.key.startswith("aten::"):
        rows.append((e.key, gt, ct, e.count))
rows.sort(key=lambda r: -r[1])

steps_profiled = args.steps
print(f"\nprofiled steps = {steps_profiled}, 每步 wall ≈ {wall_ms:.3f} ms")
print(f"CUDA kernel 总时间: {cuda_total:.3f} ms → {cuda_total/steps_profiled:.3f} ms/step")
print(f"CPU 总时间(profiled 窗口): {cpu_total:.3f} ms → {cpu_total/steps_profiled:.3f} ms/step")
print(f"设备忙碌率(窗口口径) ≈ {100.0*cuda_total/(wall_ms*steps_profiled):.1f}%\n")

print(f"{'kernel':60s} {'cuda_ms':>9s} {'/step':>8s} {'calls':>7s}")
tot = 0.0
for key, gt, ct, cnt in rows:
    print(f"{key[:60]:60s} {gt:9.3f} {gt/steps_profiled:8.3f} {cnt:7d}")
    tot += gt
print(f"{'合计(非 aten)':60s} {tot:9.3f} {tot/steps_profiled:8.3f}")

# 按类别聚合（对照 nn 的三桶：GEMM / 融合表达式(注意力+逐元素+CE) / Adam）
def bucket(name):
    n = name.lower()
    if any(k in n for k in ("gemm", "matmul", "cutlass", "gemv", "s1688", "sm80", "sm75", "ampere", "volta", "spgemm")):
        return "GEMM"
    if any(k in n for k in ("foreach", "ampere_foreach", "vectorized_elementwise", "elementwise")):
        return "elementwise/foreach(含Adam/zero_grad)"
    if "softmax" in n or "attention" in n or "fill" in n:
        return "softmax/fill(注意力周边)"
    if "reduce" in n or "norm" in n:
        return "reduce/norm"
    if "memcpy" in n or "copy" in n or "index" in n:
        return "copy/index"
    return "other"


agg = {}
for key, gt, ct, cnt in rows:
    agg.setdefault(bucket(key), [0.0, 0])
    agg[bucket(key)][0] += gt
    agg[bucket(key)][1] += cnt
print("\n按类别（ms/step，profiled 窗口 / 步数）:")
tot2 = 0.0
for k, (gt, cnt) in sorted(agg.items(), key=lambda x: -x[1][0]):
    print(f"  {k:34s} {gt/steps_profiled:8.3f} ms/step   calls={cnt}")
    tot2 += gt
print(f"  {'合计':34s} {tot2/steps_profiled:8.3f} ms/step")

# aten 层聚合（看 host 侧 dispatch 数量，对照 nn 的 ~602 dispatch/step）
aten_rows = [(e.key, e.self_cpu_time_total / 1e3, e.count) for e in evt
             if e.key.startswith("aten::") and e.self_cpu_time_total > 0]
aten_rows.sort(key=lambda r: -r[1])
aten_calls = sum(c for _, _, c in aten_rows)
print(f"\naten 调用总数(profiled 窗口) = {aten_calls} → {aten_calls/steps_profiled:.0f} 次/step")
for key, ct, cnt in aten_rows[:8]:
    print(f"  {key[:52]:52s} cpu {ct:8.3f} ms  calls={cnt}")

print("\n== torch profiler 原始表(前 40 行) ==")
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=40))
