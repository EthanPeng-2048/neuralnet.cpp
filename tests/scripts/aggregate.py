#!/usr/bin/env python3
"""汇总小模型 norm A/B 各轮结果：按配置聚合（跨轮次合并），只统计跑满 6 epoch 的轮次"""
import glob, os, re, statistics as st

D = os.path.dirname(os.path.abspath(__file__))

def parse(path):
    """返回 (final_test, best_test, label) 或 None（未跑满则跳过）"""
    txt = open(path, errors='replace').read().replace('\r', '\n')
    m = re.search(r'归一化: (.+)', txt)
    label = m.group(1).strip() if m else '?'
    eps = re.findall(r'Epoch (\d+)/6\s+lr=\S+\s+loss=\S+\s+train_acc=\S+\s+test_acc=(\S+)%', txt)
    if not eps or eps[-1][0] != '6':
        return None                      # 未完成的轮次不算
    finals = [float(t) for e, t in eps]
    return (finals[-1], max(finals), label)

rows = {}
for f in sorted(glob.glob(D + '/s*_*.log')):
    base = os.path.basename(f)[:-4]
    cfg = re.sub(r'^s\d*_', '', base)    # s_base / s2_base / s3_base → base
    r = parse(f)
    if r:
        rows.setdefault(cfg, []).append(r)

if not rows:
    raise SystemExit('no completed runs')

print(f"{'config':<10} {'n':>2} {'mean':>7} {'std':>6} {'min':>7} {'max':>7} "
      f"{'best_mean':>9}  label")
stats = []
for name in sorted(rows, key=lambda n: -st.mean(x[0] for x in rows[n])):
    finals = [x[0] for x in rows[name]]
    bests  = [x[1] for x in rows[name]]
    mean, sd = st.mean(finals), (st.stdev(finals) if len(finals) > 1 else 0.0)
    stats.append((name, mean, sd))
    print(f"{name:<10} {len(finals):>2} {mean:>7.2f} {sd:>6.2f} {min(finals):>7.2f} "
          f"{max(finals):>7.2f} {st.mean(bests):>9.2f}  {rows[name][0][2]}")

base = [s for s in stats if s[0] == 'base']
if base:
    b = base[0]
    print(f"\nvs base ({b[1]:.2f} ± {b[2]:.2f}, n={len(rows['base'])}):")
    for name, mean, sd in stats:
        if name != 'base':
            print(f"  {name:<10} {mean-b[1]:+.2f} pp")
