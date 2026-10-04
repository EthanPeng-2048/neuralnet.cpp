#!/usr/bin/env python3
"""从原始日志生成汇总表：summary.csv / small_proxy_3rounds.txt / fullsize_18ep.txt
用法: python3 make_tables.py <pack_root>     (pack_root = norm_ab_pack/)
"""
import csv, glob, os, re, statistics as st, sys

pack = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else '.')
res = os.path.join(pack, 'results')

def parse(path, total_epochs):
    txt = open(path, errors='replace').read().replace('\r', '\n')
    m = re.search(r'归一化: (.+)', txt)
    label = m.group(1).strip() if m else '?'
    pat = (r'Epoch (\d+)/' + str(total_epochs) + r'\s+lr=\S+\s+loss=\S+'
           r'\s+train_acc=(\S+)\s+test_acc=([0-9.]+)')
    eps = re.findall(pat, txt)
    if not eps:
        return None
    done = eps[-1][0] == str(total_epochs)
    finals = [float(t) for _, _, t in eps]
    return dict(done=done, n_epochs=len(eps), final=finals[-1], best=max(finals),
                final_train=float(eps[-1][1].rstrip('%')), label=label)

rows = []
for sub, phase, total in (('logs_smallproxy', 'small_proxy', 6),
                          ('logs_fullsize', 'fullsize', 18)):
    for f in sorted(glob.glob(os.path.join(res, sub, '*.log'))):
        stem = os.path.basename(f)[:-4]
        cfg = re.sub(r'^s\d*_', '', stem) if phase == 'small_proxy' else stem
        rnd = 1 if not re.match(r'^s\d*_', stem) else (1 if stem.startswith('s_') else int(stem[1:stem.index('_')]))
        p = parse(f, total)
        if not p:
            continue
        model = os.path.join(res, 'models_' + ('smallproxy' if phase == 'small_proxy' else 'fullsize'),
                             stem + '.bin')
        rows.append(dict(phase=phase, config=cfg, round=rnd, completed=p['done'],
                         epochs_done=p['n_epochs'], final_test=p['final'], best_test=p['best'],
                         final_train=p['final_train'], norm=p['label'],
                         model=os.path.relpath(model, pack) if os.path.exists(model) else '',
                         log=os.path.relpath(f, pack)))

with open(os.path.join(res, 'summary.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

# ── 小模型 3 轮聚合 ──────────────────────────────────────────────
def table(phase, out, complete_only=True):
    groups = {}
    for r in rows:
        if r['phase'] != phase or (complete_only and not r['completed']):
            continue
        groups.setdefault(r['config'], []).append(r)
    lines, stats = [], []
    for cfg, rs in sorted(groups.items(), key=lambda kv: -st.mean(x['final_test'] for x in kv[1])):
        f = [x['final_test'] for x in rs]
        b = [x['best_test'] for x in rs]
        mean = st.mean(f)
        sd = st.stdev(f) if len(f) > 1 else 0.0
        stats.append((cfg, mean, sd))
        lines.append((cfg, len(f), mean, sd, min(f), max(f), st.mean(b), rs[0]['norm']))
    base = next((s for s in stats if s[0] == 'base'), None)
    hdr = (f"{'config':<10}{'n':>3}{'mean':>8}{'std':>7}{'min':>8}{'max':>8}"
           f"{'best_mean':>11}{'vs_base':>9}  norm")
    out_lines = [hdr, '-' * len(hdr)]
    for cfg, n, mean, sd, lo, hi, bm, label in lines:
        d = f"{mean - base[1]:+.2f}" if base and cfg != 'base' else ('0.00' if cfg == 'base' else 'n/a')
        out_lines.append(f"{cfg:<10}{n:>3}{mean:>8.2f}{sd:>7.2f}{lo:>8.2f}{hi:>8.2f}"
                         f"{bm:>11.2f}{d:>9}  {label}")
    if base:
        out_lines.append(f"\nbase = BatchNorm @ conv（当前默认），mean {base[1]:.2f} ± {base[2]:.2f}")
    with open(out, 'w') as fh:
        fh.write('\n'.join(out_lines) + '\n')
    print('\n'.join(out_lines))

print('── 小模型代理（6 epoch / 20k 样本 / 8,16ch / FC 64,10，3 轮）')
table('small_proxy', os.path.join(res, 'small_proxy_3rounds.txt'))
print('\n── 全尺寸（18 epoch / 60k 样本 / 16,32ch / FC 256,128,10，单轮）')
table('fullsize', os.path.join(res, 'fullsize_18ep.txt'), complete_only=False)
print(f"\nwrote {os.path.join(res, 'summary.csv')}  ({len(rows)} runs)")
