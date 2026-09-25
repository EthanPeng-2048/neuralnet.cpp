# probe_torch_step.py — torch 训练单步分段计时（定位 e2e 150ms 的构成）
# 预构建一个 batch，逐段 CUDA event 计时：data/H2D、forward(+CE)、backward、Adam。
import sys, os, time
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "compare_with_torch"))
from model import GPTModel

torch.manual_seed(0)
dev = torch.device("cuda")
V, D, SEQ, B, L, H, FF = 8208, 64, 256, 64, 4, 4, 256

model = GPTModel(vocab_size=V, d_model=D, seq_len=SEQ, num_heads=H,
                 d_ff=FF, num_layers=L).to(dev).half()
opt = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-4)
model.train()

# 预构建数据（host 侧仍每步重算，模拟真实管线）
flow = np.random.randint(0, V, size=(64 * 400,), dtype=np.int64)
ar = np.arange(SEQ)

def build_batch():
    starts = np.random.randint(0, len(flow) - SEQ - 1, size=B)
    pos = starts[:, None] + ar[None, :]
    x = flow[pos]
    y = flow[pos + 1]
    m = np.ones((B, SEQ), dtype=np.float32)   # probe 只测计时，mask 语义无关
    return x, y, m

def ev_time(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    tss = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    best = float("inf")
    for _ in range(iters):
        tss[0].record(); fn(); tss[1].record(); torch.cuda.synchronize()
        best = min(best, tss[0].elapsed_time(tss[1]))
    return best

# 预热分配器
for _ in range(10):
    x, y, m = build_batch()
    xb = torch.from_numpy(x).to(dev); yb = torch.from_numpy(y).to(dev); mb = torch.from_numpy(m).to(dev)
    _, loss = model(xb, targets=yb, loss_mask=mb)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
torch.cuda.synchronize()

# 分段：各段独立计时会破坏流水，但用于量化开销构成已足够
# host 构 batch（纯 CPU，wall clock）
best_host = 1e9
for _ in range(50):
    t0 = time.perf_counter(); build_batch(); best_host = min(best_host, time.perf_counter() - t0)

x, y, m = build_batch()
xb = torch.from_numpy(x).to(dev); yb = torch.from_numpy(y).to(dev); mb = torch.from_numpy(m).to(dev)

def seg_h2d():
    a, b, c = build_batch()
    _x = torch.from_numpy(a).to(dev); _y = torch.from_numpy(b).to(dev); _mm = torch.from_numpy(c).to(dev)
    return _x, _y, _mm

t_h2d = ev_time(lambda: seg_h2d())

def seg_fwd():
    global xb, yb, mb
    return model(xb, targets=yb, loss_mask=mb)

# forward 计时（含 CE）
def fwd_only():
    with torch.no_grad():
        model(xb)
t_fwd_noloss = ev_time(fwd_only)

_, loss0 = model(xb, targets=yb, loss_mask=mb)

def seg_fwdloss():
    _, l = model(xb, targets=yb, loss_mask=mb)
    return l
t_fwd_loss = ev_time(seg_fwdloss)

def seg_bwd():
    opt.zero_grad(set_to_none=True)
    _, l = model(xb, targets=yb, loss_mask=mb)
    l.backward()
t_fb = ev_time(seg_bwd)          # forward+loss+backward

def seg_adam():
    opt.step()
t_adam = ev_time(seg_adam)

def seg_zero():
    opt.zero_grad(set_to_none=True)
t_zero = ev_time(seg_zero)

# full step（与 train_torch.py 相同结构，无逐步 sync）
def full_step():
    a, b, c = build_batch()
    _x = torch.from_numpy(a).to(dev); _y = torch.from_numpy(b).to(dev); _mm = torch.from_numpy(c).to(dev)
    opt.zero_grad(set_to_none=True)
    _, l = model(_x, targets=_y, loss_mask=_mm)
    l.backward()
    opt.step()

best_full = 1e9
for _ in range(50):
    t0 = time.perf_counter(); full_step(); best_full = min(best_full, time.perf_counter() - t0)
torch.cuda.synchronize()
best_full_sync = 1e9
for _ in range(50):
    torch.cuda.synchronize(); t0 = time.perf_counter(); full_step(); torch.cuda.synchronize()
    best_full_sync = min(best_full_sync, time.perf_counter() - t0)

print(f"host 构 batch        : {best_host*1e3:8.3f} ms (CPU wall)")
print(f"H2D 含构 batch       : {t_h2d:8.3f} ms")
print(f"forward (no loss)    : {t_fwd_noloss:8.3f} ms")
print(f"forward + CE loss    : {t_fwd_loss:8.3f} ms")
print(f"fwd+loss+bwd         : {t_fb:8.3f} ms")
print(f"Adam step            : {t_adam:8.3f} ms")
print(f"zero_grad            : {t_zero:8.3f} ms")
print(f"full step (host wall, 无逐步 sync): {best_full*1e3:8.3f} ms")
print(f"full step (双侧 sync): {best_full_sync*1e3:8.3f} ms")
