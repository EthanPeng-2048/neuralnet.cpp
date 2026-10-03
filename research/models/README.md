# 模型权重（不在仓库内）

本目录的 `*.bin` 是训练产物，仓库 `.gitignore` 全域忽略它们，**刻意不入库**：

- 权重体积约 750 KB/个，逐 run 备份会把仓库撑大，且完全可由脚本重跑得到；
- 需要分享的权重走 **GitHub Releases**（发布资产），仓库只保留产生它的代码与超参。

## 本次研究的最佳权重

| 资产名 | 文件 | 测试准确率 | 体积 | 训练命令 |
|---|---|---|---|---|
| `mnist_cnn_best_v3.bin` | 本目录 | 99.38%（该轮；5 轮均值 99.34%） | 743 KB | 见 `research/README.md` §2 |

> 该文件在本地由 `research/scripts/focus_cnn.py` 的 confirm 轮次产出，
> 也可按 README §2 的命令直接重训得到（18 epoch，GPU 约 2 分钟）。

## 复现权重

```bash
cmake -B build_rel -G Ninja -DCMAKE_BUILD_TYPE=Release -DNN_ENABLE_TESTS=OFF
cmake --build build_rel
./build_rel/mnist_train.exe --arch cnn --gpu --epochs 18 --batch-size 128 \
  --optimizer adamw --weight-decay 0.001 --lr 0.003 \
  --lr-schedule cosine --warmup-epochs 3 --min-lr 1e-6 \
  --cnn-channels 16,32 --cnn-fc 256,128,10 --save mnist_cnn_best_v3.bin
```

（同一命令每次会因 batch 打乱得到 99.26%~99.38% 的不同权重，属正常轮间波动。）
