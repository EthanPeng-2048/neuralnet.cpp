#!/bin/bash
# 小模型归一化 A/B —— 快速筛选版（省手机资源）
# 与全尺寸差异：通道 8,16（原 16,32）/ FC 64,10（原 256,128,10）/
#               6 epoch（原 18，warmup 2）/ 训练样本 20000（原 60000）
# 优化器与 lr 保持一致，norm 排序在代理配置上仍可比
set -u
D=/data/data/com.termux/files/home/codes/neuralnet.cpp/build/norm_sweep
PFX=${PFX:-s_}
cd /data/data/com.termux/files/home/codes/neuralnet.cpp || exit 1

LIST="
base:--norm auto --norm-place auto
bn_both:--norm batchnorm --norm-place both
bn_head:--norm batchnorm --norm-place head
ln_both:--norm layernorm --norm-place both
ln_conv:--norm layernorm --norm-place conv
ln_head:--norm layernorm --norm-place head
rms_both:--norm rmsnorm --norm-place both
rms_conv:--norm rmsnorm --norm-place conv
rms_head:--norm rmsnorm --norm-place head
none:--norm batchnorm --norm-place none
"

while IFS= read -r line; do
  [ -z "$line" ] && continue
  name=${line%%:*}
  args=${line#*:}
  if [ -f "$D/$PFX$name.done" ]; then echo "skip $name"; continue; fi
  echo "=== START $name $(date +%H:%M:%S) $args"
  {
    ./build/mnist_train --arch cnn --gpu --epochs 6 --batch-size 128 \
      --lr 0.005 --optimizer adamw --weight-decay 0.001 \
      --cnn-channels 8,16 --cnn-fc 64,10 \
      --lr-schedule cosine --warmup-epochs 2 --min-lr 1e-6 \
      --max-samples 20000 --save "$D/$PFX$name.bin" $args
    echo "EXIT=$?"
  } > "$D/$PFX$name.log" 2>&1
  echo "=== DONE  $name $(date +%H:%M:%S) $(grep -o 'test_acc=[0-9.]*%' $D/$PFX$name.log | tail -1)"
  touch "$D/$PFX$name.done"
done <<< "$LIST"
echo SMALL_SWEEP_DONE $(date +%H:%M:%S)
