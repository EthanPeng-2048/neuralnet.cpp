#!/bin/bash
# CNN 归一化 A/B：类型 × 挂载位置（其余超参与用户已知最优配置逐字一致）
# 用法: bash run.sh <name> [extra args...]
set -u
ROOT=/data/data/com.termux/files/home/codes/neuralnet.cpp
D=$ROOT/build/norm_sweep
NAME=$1; shift
cd "$ROOT" || exit 1
{
  echo "### $NAME  extra: $*"
  ./build/mnist_train --arch cnn --gpu --epochs 18 --batch-size 128 \
    --lr 0.005 --optimizer adamw --weight-decay 0.001 \
    --cnn-channels 16,32 --cnn-fc 256,128,10 \
    --lr-schedule cosine --warmup-epochs 3 --min-lr 1e-6 \
    --save "$D/$NAME.bin" "$@"
  echo "EXIT=$?"
} > "$D/$NAME.log" 2>&1
