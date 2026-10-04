#!/bin/bash
# 汇总：每个配置逐 epoch 的 train/test acc + 最终值
D=/data/data/com.termux/files/home/codes/neuralnet.cpp/build/norm_sweep
printf "%-10s %-6s %-12s %-22s %-10s\n" NAME NORM final_test best_test last_train
for f in $D/*.log; do
  n=$(basename "$f" .log)
  [ "$n" = sweep ] && continue
  norm=$(tr '\r' '\n' < "$f" | grep -m1 "归一化:" | sed 's/.*归一化: //')
  epochs=$(tr '\r' '\n' < "$f" | grep -o "Epoch [0-9]*/18  *lr=.*test_acc=[0-9.]*%")
  [ -z "$epochs" ] && { printf "%-10s %-6s (running/failed)\n" "$n" "$norm"; continue; }
  final=$(echo "$epochs" | tail -1 | grep -o "test_acc=[0-9.]*%" | grep -o "[0-9.]*")
  best=$(echo "$epochs" | grep -o "test_acc=[0-9.]*%" | grep -o "[0-9.]*" | sort -rn | head -1)
  tr_=$(echo "$epochs" | tail -1 | grep -o "train_acc=[0-9.]*%" | grep -o "[0-9.]*")
  printf "%-10s %-6s %-12s %-22s %-10s\n" "$n" "$norm" "$final" "$best" "$tr_"
done
