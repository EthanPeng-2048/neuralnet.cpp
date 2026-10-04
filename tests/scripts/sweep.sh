#!/bin/bash
# CNN 归一化 A/B 扫描 —— 串行（实测 GPU 并发会互相拖慢：3 路并发聚合吞吐 ≈ 单跑 55%）
# 顺序按"最有希望先跑"排列，便于中途出结论
set -u
D=/data/data/com.termux/files/home/codes/neuralnet.cpp/build/norm_sweep

LIST="
base:--norm auto --norm-place auto
bn_both:--norm batchnorm --norm-place both
bn_head:--norm batchnorm --norm-place head
ln_both:--norm layernorm --norm-place both
ln_conv:--norm layernorm --norm-place conv
rms_both:--norm rmsnorm --norm-place both
ln_head:--norm layernorm --norm-place head
rms_conv:--norm rmsnorm --norm-place conv
rms_head:--norm rmsnorm --norm-place head
none:--norm batchnorm --norm-place none
"

while IFS= read -r line; do
  [ -z "$line" ] && continue
  name=${line%%:*}
  args=${line#*:}
  if [ -f "$D/$name.done" ]; then echo "skip $name"; continue; fi
  echo "=== START $name $(date +%H:%M:%S) $args"
  bash "$D/run.sh" "$name" $args
  echo "=== DONE  $name $(date +%H:%M:%S) $(grep -o 'test_acc=[0-9.]*%' $D/$name.log | tail -1)"
  touch "$D/$name.done"
done <<< "$LIST"
echo SWEEP_DONE $(date +%H:%M:%S)
