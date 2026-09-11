# 📦 训练包：跨设备复现训练

> 用一个文件记录超参与训练集，用一条命令在任意设备上按超参训练，
> 方便在多台设备上做训练/测试/对比。

---

## 为什么用「tar + 外层压缩」

| 方案 | 优点 | 缺点 | 结论 |
|------|------|------|------|
| **tar.gz（默认）** | 仅标准库 `tarfile`；全平台通用；对文本语料压缩率高（3~5x）；任何工具可打开 | gzip 压缩略慢 | ✅ 推荐默认 |
| **tar.zst（可选）** | 压缩/解压快得多；比率与 gzip 相当；适合超大语料 | 需 `pip install zstandard` | ⭐ 大语料推荐 |
| **tar.xz** | 压缩率最高 | 压缩极慢，日常迭代不划算 | ❌ 归档专用 |

**结论**：日常训练对比推荐 **gzip**（零依赖、通用）；语料很大（GB 级）时用 **zstd**。
训练集是文件夹时用 `tar` 递归打包整个目录树，是单个超大文件时作为单个条目打入，
最后外层压缩一层，形成单个 `.nnpkg` 文件便于拷贝。

## 包格式

```
run.nnpkg                          # = tar.gz（或外层 zstd 的 tar）
├── manifest.json                   # 超参、任务、数据元信息、sha256 校验和
└── data/                           # 训练数据（文件或目录树）
    ├── train/...                   #   训练集（必填）
    ├── test/...                    #   测试集（GPT 可选）
    └── vocab/...                   #   词表（GPT 可选）
```

## 配置模板

```bash
python train_pkg.py new --task gpt -o runs/gpt.json
python train_pkg.py new --task mnist -o runs/mnist.json
```

`runs/gpt.json` 示例：

```json
{
  "format_version": 1,
  "name": "gpt-tiny",
  "task": "gpt",                // gpt | mnist
  "device": "cpu",              // cpu | gpu | cuda（train 时可用 --device 覆盖）
  "data": {                     // 训练集路径（相对本配置所在目录）
    "train": "datasets/tinystories_20k.txt",
    "test": "",
    "vocab": "bpe_vocab.json"
  },
  "hyperparameters": {          // 键名与 cli_controllers 控制器参数一致
    "epochs": 10,
    "lr": 0.001,
    "batch_size": 4,
    "d_model": 256,
    "num_heads": 4,
    "num_layers": 4,
    "d_ff": 1024,
    "optimizer": "adam",
    "weight_decay": 0.01,
    "seq_len": 256,
    "lr_schedule": "fixed"
  }
}
```

MNIST 的 `data.train` 是**数据集目录**（如 `datasets/mnist_data`），
GPT 的 `data.train` 是**文本文件**（`data.test`/`data.vocab` 可选）。

## 工作流

### 方式一：本机直接用配置训练

```bash
python train_pkg.py train runs/gpt.json
```

### 方式二：跨设备复现

**A 设备（制作包）：**

```bash
python train_pkg.py pack runs/gpt.json -o runs/gpt.nnpkg
# 大语料可换 zstd：
python train_pkg.py pack runs/gpt.json -o runs/gpt.nnpkg --compress zstd
```

**拷贝 `runs/gpt.nnpkg` 到 B 设备，然后：**

```bash
python train_pkg.py info runs/gpt.nnpkg     # 查看包内配置/数据/校验和
python train_pkg.py train runs/gpt.nnpkg    # 自动解包 + 校验 + 训练
python train_pkg.py train runs/gpt.nnpkg --device cuda   # 按设备覆盖
python train_pkg.py train runs/gpt.nnpkg --save runs/gpt.bin
```

`train` 会解包到 `<包路径>.d/` 缓存目录（manifest 未变则复用，不重复解压），
数据路径自动映射，按 `manifest.json` 里的超参调用 `cli_controllers` 训练。

### 其他命令

```bash
python train_pkg.py extract runs/gpt.nnpkg -o runs/gpt_extract   # 解包查看
python train_pkg.py info runs/gpt.nnpkg --list                    # 列出包内文件
```

## GUI：导出训练包

`gui.py` 的 **MNIST 训练** 与 **GPT 训练** 标签页底部有 **「📦 导出训练包」** 按钮：

1. 填好当前页面的所有超参与数据路径
2. 点「📦 导出训练包」，选择保存位置（`.nnpkg`）
3. 自动把配置 + 训练集（MNIST 是数据集目录、GPT 是文本/词表）打包压缩
4. 把文件拷到其他设备，用 `python train_pkg.py train xxx.nnpkg` 即可复现

MNIST 与 GPT 各自独立导出，互不影响。

## 数据校验

打包时对每个**文件**记录 `sha256`，`train` 解包后按 manifest 定位数据；
如需核对完整性，可用 `info` 查看校验和。
目录型数据（MNIST）递归打包、按文件数/总字节记录大小。

## 说明

- 模型文件（`save`/`resume`）属于产物而非输入，**不随包分发**；
  `resume` 为本地路径，跨设备时需自行拷贝或移除。
- `hyperparameters` 的键名直接对应 `cli_controllers` 各控制器参数
  （`MnistTrainController` / `GptTrainController`），未知键会被忽略。
- 依赖：仅 `cli_controllers.py`（同目录）；zstd 为可选增强。
