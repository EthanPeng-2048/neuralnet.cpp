# 快速上手：训练与推理

**本教程教你如何使用 neuralnet.cpp 训练和推理 MNIST 手写数字识别与 GPT 文本生成模型。**

---

## 前置条件

**构建项目：**

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

**数据准备：**

MNIST 数据不随仓库分发：从
[Releases](https://github.com/EthanPeng-2048/neuralnet.cpp/releases) 下载 `mnist_data`
压缩包，解压到 `datasets/mnist_data/`（`train.csv` / `test.csv`）。
GPT 训练需要一份文本语料（如 `datasets/llm_corpus.txt`，用 `scripts/download_*.py` 拉取）。

---

## MNIST 手写数字识别

### 训练

```bash
# MLP 架构（默认）
./build/mnist_train --epochs 10 --lr 0.001 --optimizer adam

# Transformer (ViT) 架构
./build/mnist_train --arch transformer --epochs 20 --lr 0.001

# 使用 Vulkan GPU 加速
./build/mnist_train --gpu --epochs 10

# 多卡机器：手动指定计算设备（默认自动按"独显优先 + 驱动 apiVersion"选卡）
#   索引形式（枚举索引，见初始化时打印的 GPU 名或错误里列出的候选）
./build/mnist_train --gpu 2 --epochs 10
#   名称子串形式（--gpu= 写法，避免与位置参数歧义）
./build/mnist_train --gpu=NVIDIA --epochs 10

# 恢复训练
./build/mnist_train --resume mnist_model.bin --epochs 5

# 快速测试（限制样本数）
./build/mnist_train --max-samples 1000 --epochs 3
```

**完整参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--arch` | `mlp` | 架构：`mlp` / `transformer` / `cnn` |
| `--resume <path>` | 无 | 从已有模型恢复训练 |
| `--save <path>` | `mnist_model.bin` | 模型保存路径 |
| `--dataset <path>` | `datasets/mnist_data` | 数据集目录 |
| `--epochs <n>` | `10` | 训练轮数 |
| `--lr <lr>` | `0.001` | 学习率 |
| `--batch-size <n>` | `64` | 批大小 |
| `--optimizer <name>` | `adam` | 优化器：`sgd`/`sgd_momentum`/`adam`/`adamw`/`muon` |
| `--gpu <索引>` / `--gpu=<名称>` | 禁用 | 启用 Vulkan GPU 加速；可选指定计算设备（索引如 `2`，名称子串如 `--gpu=40HX`/`--gpu=NVIDIA`） |
| `--max-samples <n>` | 全部 | 限制训练样本数 |
| `--norm <type>` | `auto` | 归一化类型：`auto` / `layernorm` / `rmsnorm` / `batchnorm`（三架构通用；`auto` = 按架构：MLP/ViT = layernorm、CNN = batchnorm） |
| `--norm-place <where>` | `auto` | 归一化挂载：`auto` / `none` / `conv` / `head` / `both`（CNN）、`final`（ViT）；`auto` = 按架构：CNN = `conv`、ViT = `final`；`none` = 不额外添加（旧模型文件缺该键即此值，参数布局与旧实现一致），MLP 只接受 `none`/`auto` |

### 推理

推理入口接受**单张** CSV（`784` 像素一行）。用 GUI 的**手写板**画一个数字、
或自己导出一张 CSV 即可，无需预先准备测试图。

```bash
# 使用已训练的模型推理（图片路径是位置参数）
./build/mnist_infer my_digit.csv --model mnist_model.bin
```

---

## GPT 文本生成

### 准备数据集（dataset_gen）

训练数据统一为 `.nndataset` 文件：token 流、词表、loss 掩码、test 子集都在**生成期**
定好并嵌入数据集（设计见 `docs/todo/19-unified-dataset.md`）：

```bash
# 先训词表（tokenizer_train），再打包数据集（dataset_gen）
./build/tokenizer_train datasets/llm_corpus.txt --vocab-size 10000 --output my_bpe.nnvocab
./build/dataset_gen datasets/llm_corpus.txt --vocab my_bpe.nnvocab -o llm.nndataset

# 对话 SFT：生成期写 assistant 掩码（--loss-scope assistant）；
# 可选第二输入 → test 子集（每 epoch 末自动评估 test loss）
./build/dataset_gen dialogs.txt --vocab my_bpe.nnvocab -o dialogs.nndataset \
    --loss-scope assistant --test dialogs_eval.txt
```

### 训练

```bash
# 基础训练
./build/text_train datasets/llm.nndataset --epochs 10 --lr 0.001

# 自定义模型大小
./build/text_train datasets/llm.nndataset \
    --d-model 256 --num-heads 8 --num-layers 6 --d-ff 1024 \
    --seq-len 512 --batch-size 16 --epochs 20

# 使用 Muon 优化器
./build/text_train datasets/llm.nndataset --optimizer muon --lr 0.002

# 恢复训练（.nnckpt = 断点续训：形状/配方/计数器由断点决定，禁止再给 --epochs/--lr 等；
#           .nnmodel = warm start：CLI 全权）
./build/text_train datasets/llm.nndataset --resume gpt_model.nnckpt
```

**完整参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `<dataset.nndataset>` | 必需 | 数据集文件路径（dataset_gen 生成；词表/loss 掩码/test 子集随数据集） |
| `--save <path>` | `gpt_model.nnmodel` | 推理模型保存路径（`.nnmodel`，不含训练参数） |
| `--resume <path>` | 无 | 从模型文件恢复训练：`.nnckpt` = 断点续训（形状/配方/计数器由断点决定）；`.nnmodel` = warm start |
| `--epochs <n>` | `10` | 训练轮数 |
| `--lr <lr>` | `0.001` | 学习率 |
| `--batch-size <n>` | `32` | 批大小 |
| `--seq-len <n>` | `256` | 序列长度 |
| `--optimizer <name>` | `adam` | 优化器 |
| `--weight-decay <w>` | `0.01` | AdamW 权重衰减 |
| `--beta1 <b>` / `--beta2 <b>` | `0.9` / `0.999` | Adam/AdamW 动量衰减（LLM 预训练常取 β2=0.95） |
| `--max-steps <n>` | `0` | 本次运行最多训练多少 step 后停止并保存（0=不限；吞吐实测/分段跑） |
| `--d-model <n>` | `128` | 模型维度 |
| `--num-heads <n>` | `4` | 注意力头数 |
| `--num-layers <n>` | `4` | Transformer 层数 |
| `--d-ff <n>` | `512` | FFN 中间维度 |
| `--gpu <索引>` / `--gpu=<名称>` | 禁用 | 启用 Vulkan GPU 加速；可选指定计算设备（索引如 `2`，名称子串如 `--gpu=40HX`/`--gpu=NVIDIA`） |
| `--log-interval <n>` | `50` | 日志间隔 |

> 词表、loss 范围（`all`/`assistant`）、test 子集都随数据集（生成期定好），
> 不再是 `text_train` 的选项；窗口化（`--seq-len`/`--stride`）仍属训练策略，
> P2b 起由库层 `TextSampleSet` / `TextTrainer` 承担（step = ceil(样本数/batch) 是派生量）。

### 推理

```bash
# 单次生成
./build/text_infer --model gpt_model.bin --prompt "Once upon a time" --max-tokens 200

# 交互模式
./build/text_infer --interactive

# 调节温度（0=贪心，>1=更随机）
./build/text_infer --model gpt_model.bin --prompt "Hello" --temperature 0.8

# 采样截断 + 可复现（同 seed 同 prompt ⇒ 同输出）
./build/text_infer --model gpt_model.bin --prompt "Hello" --seed 7 \
    --top-k 40 --top-p 0.9 --repeat-penalty 1.2

# 多轮对话（历史自带：GUI 的「💬 发送（多轮）」发的就是这条形状）
./build/text_infer --model gpt_model.nnmodel --seed 7 \
    --turn user:"你好" --turn assistant:"你好！有什么可以帮你的？" --turn user:"再讲一遍" \
    --reply-out reply.txt

# 使用 GPU
./build/text_infer --model gpt_model.bin --gpu --prompt "Hello"
```

**完整参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model <path>` | `gpt_model.bin` | 模型文件路径 |
| `--vocab <path>` | `bpe_vocab.nnvocab` | 词表 `.nnvocab` 路径（仅模型未嵌入时） |
| `--prompt <text>` | `Hello` | 输入提示文本 |
| `--interactive` | 禁用 | 交互式生成模式（**多轮**：上下文累积保留，输入 `quit`/`exit` 退出） |
| `--turn <role>:<text>` | 空 | 追加一轮对话历史（`role` = `user`/`assistant`，**可重复**）。给了它即进入多轮对话模式：按词表的对话标记拼 prompt（`BOS<|system|>…<|user|>…<|end_of_user|><|assistant|>`），**末条必须是 `user`**（模型回答的就是这一轮）；角色必须 user/assistant 交替 |
| `--reply-out <path>` | 空 | 多轮模式：把本轮回复（纯文本）另写一份到该文件（GUI 靠它取回复，不解析日志） |
| `--system <text>` | `你是一个有用的AI助手。` | 对话模式的系统提示（`--turn` 与 `--interactive` 共用） |
| `--max-tokens <n>` | `200` | 最大生成 token 数 |
| `--temperature <t>` | `1.0` | 温度参数（0=贪心） |
| `--top-k <n>` | `0` | top-k 截断：只保留概率最高的 n 个候选（0=关闭） |
| `--top-p <f>` | `1.0` | top-p 核采样阈值（0<f<=1；1.0=关闭） |
| `--repeat-penalty <f>` | `1.0` | 重复惩罚（>1 抑制重复；1.0=关闭） |
| `--repeat-window <n>` | `0` | 重复惩罚只看最近 n 个 token（0=全部上下文） |
| `--seed <n>` | 空 | 随机种子（给了就固定种子 ⇒ 同 prompt 输出可复现） |
| `--gpu <索引>` / `--gpu=<名称>` | 禁用 | 启用 Vulkan GPU 加速；可选指定计算设备（索引如 `2`，名称子串如 `--gpu=40HX`/`--gpu=NVIDIA`） |
| `--show-tokens` | 禁用 | 显示 token ID（调试） |

采样顺序固定为：重复惩罚 → 温度 → top-k → top-p（核采样，保留集内重新
归一化）→ softmax → 采样/贪心；全部开关关闭时与旧版（仅 temperature）
输出**逐位一致**（`sampling_test` 锁这条等价性）。`--temperature 0` 与
`--top-k 1` 都是贪心（等价）。

**多轮对话（连续 N 轮不丢上下文）**：每轮把**整段历史**编进 prompt —— 第 N 轮
的 prompt 以第 N-1 轮的 prompt 为**逐 token 前缀**（`chat_prompt_test` 锁这条
不变式），历史因此不会"每轮从零起算"。两条用法：

- `--interactive`：一个进程内跑完整场对话，历史留在进程里；助手轮用**生成的原
  token** 回灌（不经 decode → encode 往返），超 `seq_len` 时打印丢弃了哪几轮。
- `--turn … --reply-out …`：**一次一问、历史由调用方自带**（脚本 / GUI 用）。
  每次进程只回答最后一轮，回复同时打印并写进 `--reply-out`；上下文接续由调用方
  把上一轮的回复作为 `--turn assistant:<text>` 传回来。GUI 的「💬 发送（多轮）」
  就是这个形状。

两种用法共用同一条 prompt 构建（`Tokenizer::chat_prompt`）：对话词表下套标记模板，
超 `seq_len` 时**从最旧的一问一答开始整轮丢弃**（system 块与末轮恒保留，标记结构
不破）；无对话标记的词表退化为轮次文本按换行相接。多轮模式与 `--prompt`、
`--interactive` **互斥**（显式给出即报错），形状不合法（`role` 写错 / 不交替 /
末条不是 `user`）同样报错，不静默降级。

---

## 分词器

### 训练分词器

```bash
# 训练 BPE 分词器（预分词默认自动并行，可使用全部核心）
./build/tokenizer_train datasets/llm_corpus.txt --vocab-size 10000 --output my_bpe.nnvocab

# 训练 CharBPE 分词器（字符级，适合中文）
./build/tokenizer_train datasets/llm_corpus.txt --tokenizer charbpe --vocab-size 10000 --output my_charbpe.nnvocab

# 强制顺序预分词（--threads 1）或指定并行度（--threads 4）
./build/tokenizer_train datasets/llm_corpus.txt --threads 1 --output seq.nnvocab
```

`--threads`：预分词并行线程数，`0`=自动（默认，使用线程池全部核心），`1`=顺序，`>1`=指定并行度（上限为线程池大小）。并行预分词与顺序执行结果完全一致（词表逐字节相同）。

词表输出为 `.nnvocab`（kvrec v2 二进制自描述格式）。**旧 JSON 词表不直读**，
迁移用 `./build/dataset_convert vocab old.json -o old.nnvocab`。

### 分词器推理

```bash
# 编码文本
./build/tokenizer_infer --vocab gpt_bpe.nnvocab --encode "Hello world"

# 解码 token IDs
./build/tokenizer_infer --vocab gpt_bpe.nnvocab --decode "72,101,108,108,111"

# 编码整个文件（长文本自动保序并行编码，可用 --threads 指定并行度）
./build/tokenizer_infer --vocab gpt_bpe.nnvocab --encode-file corpus.txt --threads 8
```

`--threads`：单次 encode 的并行度，`0`=自动（默认，使用线程池全部核心），`1`=顺序，`>1`=指定并行度（上限为线程池大小）。**任意并行度下输出与顺序执行逐字节一致**（按空白安全切分点分段、段内并发编码、按段序拼接）；文本小于 256 KiB 或找不到切分点时自动回退顺序路径。

### 支持的分词器类型

| 类型 | `.nnvocab` 的 `"type"` 字段 | 说明 |
|------|---------------------|------|
| BPE | `"bpe"` | Byte-Pair Encoding |
| CharBPE | `"char_bpe"` | 字符级 BPE |

V4 及以上格式的模型文件会自动嵌入分词器，推理时无需单独指定 `--vocab`——
**v5 及以下模型内嵌的是 JSON 词表（照常读取），v6 起内嵌 `.nnvocab` kvrec**
（按模型版本分派，旧模型零破坏）。

---

## C++ API 编程训练

### MNIST MLP 训练

```cpp
#include <neuralnet.cpp/nn.hpp>
#include <neuralnet.cpp/model_serialization.hpp>
#include <neuralnet.cpp/domain_mnist.hpp>

int main() {
    auto engine_owner = nn::CpuEngine::create();
    nn::CpuEngine& engine = *engine_owner;

    // 1. 构建模型（使用工厂函数，自动创建同设备权重）
    auto model_result = nn::build_mnist_mlp_model(engine);
    nn::Model model = std::move(*model_result);

    // 2. 优化器
    auto optimizer = nn::create_optimizer(
        "adam", engine,
        model.parameters(), model.param_gradients(),
        0.001,  // lr
        0.01    // weight_decay
    );

    nn::CrossEntropyLoss loss_fn;

    // 3. 加载数据集（CSV 格式）
    // ... 假设已有 train_images (N×784) 和 train_labels (N×1) ...

    const std::size_t batch_size = 64;
    const std::size_t epochs = 10;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        Scalar epoch_loss = 0;

        for (std::size_t batch = 0; batch < num_batches; ++batch) {
            // 清零梯度
            optimizer->zero_grad();

            // 准备 batch 数据 (784, batch_size)
            nn::Matrix x_batch = /* ... */;
            nn::Matrix y_onehot = /* ... (10, batch_size) ... */;

            // 上传到引擎
            auto x_t = engine.from_matrix(x_batch);
            auto y_t = engine.from_matrix(y_onehot);
            nn::Tensor input = std::move(*x_t);
            nn::Tensor target = std::move(*y_t);

            // 前向传播
            auto out = model.forward(input);
            auto loss = loss_fn.forward(engine, *out, target);

            // 反向传播
            auto grad = loss_fn.backward();
            model.backward(*grad);

            // 参数更新
            optimizer->step();

            epoch_loss += *loss;
        }

        std::cout << "Epoch " << epoch + 1
                  << "  loss=" << epoch_loss / num_batches << "\n";
    }

    // 4. 保存模型（MLP 的 spec 直接构造：type + layer_dims；无 make_mlp_spec 工厂）
    nn::ModelSpec spec;
    spec.type = nn::ModelType::MLP;
    spec.layer_dims = {784, 512, 256, 128, 64, 10};
    auto save_result = nn::save_model("mnist_model.bin", model, spec);
}
```

### GPT 文本生成训练

训练入口用库层训练工厂 `nn::TextTrainer`（P2b）：**Dataset →（内部切样本）→
TrainingSession → TextTrainer**。`TextTrainConfig` 是训练配方唯一真源，epoch×step 循环、
shuffle、梯度累积/裁剪、lr 调度、测试评估、断点状态都在库里——不再需要手工切窗/填 batch。

```cpp
#include <neuralnet.cpp/nn.hpp>
#include <neuralnet.cpp/dataset.hpp>
#include <neuralnet.cpp/text_trainer.hpp>

int main() {
    auto engine_owner = nn::CpuEngine::create();
    nn::CpuEngine& engine = *engine_owner;

    // 1. 打开数据集（词表 / loss 掩码 / test 子集都在数据集内）
    auto dataset_owner = NN_UNWRAP(nn::Dataset::open("llm.nndataset"));
    const nn::Dataset& dataset = *dataset_owner;
    auto tokenizer = NN_UNWRAP(nn::load_tokenizer_from_kvrec(dataset.vocab_kvrec()));

    // 2. 构建 GPT 模型（vocab_size 来自数据集内嵌词表）
    nn::Model model = NN_UNWRAP(nn::build_gpt_model(
        engine,
        tokenizer->vocab_size(),
        128,    // d_model
        256,    // seq_len
        4,      // num_heads
        512,    // d_ff
        4       // num_layers
    ));

    // 3. 训练配方（库层唯一真源）+ Session（Model + Optimizer + TrainState）
    nn::TextTrainConfig recipe;
    recipe.seq_len      = 256;
    recipe.batch_size   = 32;
    recipe.epochs       = 10;
    recipe.lr           = 0.001f;
    recipe.lr_schedule  = "cosine";
    recipe.max_norm     = 1.0f;      // 梯度裁剪（0 = 不裁剪）

    nn::TrainingConfig scfg;
    scfg.optimizer      = "adamw";
    scfg.lr             = recipe.lr;
    scfg.weight_decay   = recipe.weight_decay;
    scfg.clip_grad_norm = recipe.max_norm;   // 必须与配方一致（Trainer 前置校验）
    nn::TrainingSession session =
        NN_UNWRAP(nn::TrainingSession::create(std::move(model), scfg));

    // 4. 数据集直连训练工厂：内部建 TextSampleSet（样本 = seq_len 窗口）并跑 epoch × step
    auto trainer = NN_UNWRAP(nn::TextTrainer::create(std::move(session), dataset, recipe));
    nn::TextTrainReport report = NN_UNWRAP(trainer.fit());
    std::printf("steps=%zu samples=%zu final_lr=%.3e\n",
                report.steps_executed, report.samples_executed,
                static_cast<double>(report.final_lr));

    // 5. 保存：用 nn::save_model(..., nn::ModelRole::Model) 写 .nnmodel；
    //    训练断点（.nnckpt）由 TextTrainerHooks::on_checkpoint 落盘。
    //    现成 CLI 见上面的 text_train（--save 默认 gpt_model.nnmodel / --save-interval 写 .nnckpt）。
}
```

> **单位约定**：`.nndataset` 的存储单位是 **doc**；训练单位是 **样本**（`seq_len` 窗口，
> 可跨 doc 边界）；`step = ceil(样本数 / batch_size)` 是**派生量**
> （`TextSampleSet::steps_per_epoch(batch_size)`）——`steps_per_epoch` **不是** CLI 旗标，
> 也不再由调用方手算。已切好样本集时用高性能入口
> `TextTrainer::create(session, train_set, test_set, recipe)`；样本集 /
> batch 装配接口见 [`../development/cpp-api-manual.md`](../development/cpp-api-manual.md) §13.3。

### GPT 推理（自回归生成）

```cpp
#include <neuralnet.cpp/nn.hpp>
#include <neuralnet.cpp/model_serialization.hpp>
#include <neuralnet.cpp/domain_gpt.hpp>

int main() {
    auto engine_owner = nn::CpuEngine::create();
    nn::CpuEngine& engine = *engine_owner;

    // 1. 加载模型（v4+ 自描述格式：文件头部含 spec + tokenizer；Model 需预先构建以提供 spec 用于校验）
    auto load_result = nn::load_model("gpt_model.bin", model);
    auto tokenizer_json = std::move(*load_result);
    auto tokenizer = nn::load_tokenizer_from_string(tokenizer_json);

    // 或使用带规格参数的工厂函数（从二进制规格直接重建模型）：
    // auto model_result = nn::build_gpt_model_from_spec(engine, spec_from_file("spec.bin"));

    // 2. 编码提示文本
    auto prompt_tokens = tokenizer->encode("Once upon a time");

    // 3. 自回归生成
    const int max_tokens = 200;
    const double temperature = 1.0;

    std::vector<std::size_t> generated(prompt_tokens.begin(), prompt_tokens.end());

    for (int i = 0; i < max_tokens; ++i) {
        // 构造输入: (seq_len, 1) — 取最后 seq_len 个 token
        nn::Matrix input(spec.seq_len, 1);
        // ... 填充 token（不足部分用 pad_id=0）...

        auto x_t = engine.from_matrix(input);
        auto logits = model.forward(*x_t);

        // 取最后一个位置的 logits
        auto logits_m = engine.to_matrix(*logits);
        // logits_m: (vocab_size, seq_len*1)
        // 最后一个位置: col = seq_len - 1

        // 温度采样
        std::vector<Scalar> probs(vocab_size);
        Scalar max_logit = *std::max_element(/* last position logits */);
        Scalar sum = 0;
        for (std::size_t v = 0; v < vocab_size; ++v) {
            probs[v] = std::exp((logits_m->at(v, seq_len-1) - max_logit) / temperature);
            sum += probs[v];
        }
        for (auto& p : probs) p /= sum;

        // 采样下一个 token
        std::size_t next_token = /* argmax 或随机采样 */;
        generated.push_back(next_token);

        if (next_token == tokenizer->eos_id()) break;
    }

    // 4. 解码输出
    std::string output = tokenizer->decode(generated);
    std::cout << output << "\n";
}
```

---

## 训练监控

### 控制台输出

训练过程中会打印：

```
[Step 50]  loss=2.345  lr=0.001000
[Step 100] loss=1.987  lr=0.001000
[Step 150] loss=1.654  lr=0.001000
...
```

### 模型保存格式

| 版本 | 格式 | 说明 |
|------|------|------|
| V5 | `[magic][version][precision][spec_len][spec: KeyValueRecord][matrices...][extra state][tokenizer]`，每个矩阵前带 1B 精度标签（f32/f16） | `save_model` 当前写入版本 |
| V4 | 同上布局，矩阵无精度标签（按 f32 读取） | 仍可读取 |
| V1/V2/V3 | 旧偏移量定长格式 | 不支持（header 校验阶段即拒绝，提示重新训练保存） |

- `save_model` 统一写入 v5；`load_model` 接受 v4 及以上自描述格式（v1/v2/v3 会提示重新训练保存）
- 规格头为长度前缀 KeyValueRecord（自描述、无偏移量假设）；
  版本默认值表保证字段缺失时回落默认（如 norm_type 缺失 → LayerNorm）；**例外：`norm_place` 缺失 → `None`（不额外加归一化）而非架构默认**，以保证旧 checkpoint 的参数布局逐位一致
- v4/v5 模型推理时无需单独指定 `--vocab` 参数（嵌入词表随模型保存）

---

## 图形化界面 (GUI) 操作指南

项目提供基于 CustomTkinter 的全功能 GUI（`gui.py`），覆盖全部训练/推理/分词器操作，无需记忆命令行参数。

```bash
# 启动 GUI
python gui.py
```

> **前提**: 需要先构建 C++ 项目（`cmake -B build -G Ninja`），GUI 会调用 `build/` 目录下的可执行文件。
> **依赖**: Python 3.10+，customtkinter（`pip install customtkinter`），Pillow（`pip install Pillow`，用于手写板功能）。

### GUI 功能一览

| Tab | 功能 |
|-----|------|
| **MNIST 训练** | MNIST 模型训练：支持 MLP / Transformer / CNN 架构切换、超参数调节、恢复训练、学习率调度（fixed / cosine）、每轮 lr 覆盖、batch 打乱开关、GPU 加速、评估样本数；实时 loss / acc 曲线 |
| **MNIST 推理** | MNIST 图片推理 + 图片查看：推理某张图片自动显示该图、Top-K 置信度条形图、目录翻页导航；内置**手写板**，鼠标书写数字即刻识别 |
| **GPT 训练** | GPT / RAPT 语言模型训练：位置编码（learned / sinusoidal / alibi / rope）、激活（GeLU / SwiGLU）、归一化（LayerNorm / RMSNorm）、梯度累积、滑动窗口 stride、学习率调度（fixed / cosine / step_cosine）、续训定位（resume-epoch / resume-step）、batch flush 粒度、梯度检查点、activation offload、tokenize 缓存开关；实时 loss 曲线 |
| **GPT 推理** | GPT 文本生成：模型 + 词表选择、温度调节、Token ID 调试输出 |
| **分词器训练** | 训练 BPE / CharBPE 分词器，配置词表大小、最小合并频率与预分词并行线程数 |
| **分词器推理** | 分词器编码/解码测试：文本编码、ID 解码、文件编码、最长行统计（top-N）、encode 并行度 |

---

### MNIST 训练 Tab

```
┌──────────────────────────────────────────────────┐
│  训练 Tab                                         │
│                                                    │
│  数据集目录:  [datasets/mnist_data    ] [浏览…]     │
│  模型保存路径: [mnist_model.bin        ] [浏览…]     │
│  ☑ 从已有模型恢复训练                                │
│                                                    │
│  ┌─ 超参数 ────────────────────────────────────┐   │
│  │ 轮数: [10]  学习率: [0.001]  批大小: [64]    │   │
│  │ 优化器: [adam ▾]  权重衰减: [0.01]           │   │
│  │ 模型类型: [mlp ▾]                            │   │
│  └─────────────────────────────────────────────┘   │
│                                                    │
│  ┌─ 模型架构 ──────────────────────────────────┐   │
│  │ MLP: 层维度 [784,512,256,128,10]             │   │
│  │ 或 Transformer: 模型维度/头数/层数/FFN/Patch  │   │
│  └─────────────────────────────────────────────┘   │
│                                                    │
│  ┌─ 学习率调度 ────────────────────────────────┐   │
│  │ 调度模式: [fixed ▾] / [cosine ▾]             │   │
│  │ cosine: 预热轮数 + 最低学习率                 │   │
│  │ 手动: 逗号分隔每轮 lr                         │   │
│  └─────────────────────────────────────────────┘   │
│                                                    │
│  ┌─ 训练选项 ──────────────────────────────────┐   │
│  │ GPU: [None ▾]  最大样本数: [0]  评估样本数: [200] │   │
│  │ ☑ 每 epoch 打乱 batch 顺序                    │   │
│  └─────────────────────────────────────────────┘   │
│                                                    │
│  [▶ 开始训练]  [⏹ 停止]                            │
│                                                    │
│  ┌─ 训练日志 ──────────────────────────────────┐   │
│  │ Epoch 1/10  loss=0.4523  train_acc=90.12%   │   │
│  │ Epoch 2/10  loss=0.2134  train_acc=94.56%   │   │
│  └─────────────────────────────────────────────┘   │
│                                                    │
│  ┌─ 训练曲线 ────────────────────────────────┐   │
│  │  ╭──╮                                        │   │
│  │ ╭╯  ╰──╮   step_loss / epoch_loss            │   │
│  │╭╯      ╰─────────   train_acc / test_acc     │   │
│  └─────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────┘
```

**操作步骤：**

1. **选择数据集** — 点击"浏览…"选择 `datasets/mnist_data` 目录（默认已填好）
2. **设置保存路径** — 模型训练完成后保存的 `.bin` 文件路径
3. **配置超参数** — 轮数、学习率、批大小、优化器（`sgd`/`adam`/`adamw`/`muon`）、权重衰减
4. **选择模型类型**：
   - `mlp`：可自定义层维度（如 `784,512,256,128,10`）
   - `transformer`：可配置模型维度、注意力头数、FFN 维度、层数、Patch 大小
5. **可选功能**：
   - ✅ 恢复训练 — 从已有模型继续训练
   - ✅ GPU 加速 — 支持 Vulkan（下拉选择设备）
   - ✅ 评估样本数 — 仅 Transformer 架构生效，限制评估样本以加快训练（0=自动）
   - ✅ 学习率调度 — 支持 cosine 衰减（含预热）和手动指定每轮 lr
6. **点击 "▶ 开始训练"** — 日志面板实时输出训练进度，下方曲线图实时绘制 step_loss、epoch_loss、train_acc、test_acc、lr

---

### MNIST 推理 Tab

**基本推理：**

1. **选择模型文件** — 点击"浏览…"选择训练好的 `.bin` 模型
2. **选择图片** — 输入单张 CSV 图片路径或包含多张图片的目录
3. **设置 Top-K** — 控制显示前 K 个预测结果
4. **点击 "▶ 开始推理"** — 左侧显示图片预览，右侧显示 Top-K 置信度条形图

**手写板识别：**

推理 Tab 内置了手写板功能，无需准备 CSV 文件即可实时识别：

1. 在左侧白色画布上用鼠标**书写一个数字**（0-9）
2. 右侧 **28×28 预览** 窗口实时显示下采样后的效果
3. 点击 **"识别手写数字"** 按钮
4. 下方显示预测结果（数字 + 置信度百分比）
5. 点击 **"清除"** 重写

```
┌──────────────────────────────────────────────────┐
│  推理 Tab                                         │
│                                                    │
│  模型文件:  [pretrained/model.bin    ] [浏览…]     │
│  图片/目录: [                        ] [浏览…]     │
│  Top-K: [3]   架构: 自动识别   ☐ GPU  ☐ 显示像素   │
│                                                    │
│  [▶ 开始推理]                                       │
│                                                    │
│  ┌─ 手写识别 ───────────────────────────────┐   │
│  │ ┌──────────┐  28×28 预览:                    │   │
│  │ │          │  ┌──────┐                       │   │
│  │ │   手写   │  │▓▓▓▓▓▓│                       │   │
│  │ │  画布    │  │▓  ▓▓▓│  预测: 3 (98.2%)     │   │
│  │ │          │  │  ▓▓▓▓│  条形图:              │   │
│  │ └──────────┘  └──────┘  ████░ 3  98.2%      │   │
│  │ [清除] [识别手写数字]    █░░░░ 7   1.2%   │   │
│  └─────────────────────────────────────────────┘   │
│                                                    │
│  ┌─ 图片预览 ────────┐  ┌─ 预测结果 ───────────┐   │
│  │                   │  │ filename.csv          │   │
│  │   28×28 灰度图    │  │ ████░░ 3  95.3%      │   │
│  │                   │  │ █░░░░░ 5   3.1%      │   │
│  └───────────────────┘  └──────────────────────┘   │
└──────────────────────────────────────────────────┘
```

---

### 图片查看（并入 MNIST 推理 Tab）

图片查看与推理合并：选择单张 CSV 或目录后，**推理某张图片会自动显示该图**。

1. 选择单张 CSV 图片文件或包含多张图片的目录
2. 使用 **上一张 / 下一张** 按钮翻页导航（已推理过的图片会缓存预测结果）
3. 左侧显示 28×28 灰度图片预览，右侧显示该图的 Top-K 置信度条形图
4. 点击 **"推理当前图片"** 推理当前图并自动刷新预览与预测；**"推理整个目录"** 可批量推理并逐个自动显示

---

### GPT 训练 Tab

```
┌──────────────────────────────────────────────────┐
│  GPT 训练 Tab                                     │
│                                                    │
│  数据集文件:  [datasets/llm.nndataset  ] [浏览…]   │
│  模型保存路径: [gpt_model.bin           ] [浏览…]   │
│  ☑ 从已有模型恢复训练                                │
│                                                    │
│  ┌─ 超参数 ────────────────────────────────────┐   │
│  │ 轮数: [10]  学习率: [0.001]  批大小: [32]    │   │
│  │ 序列长度: [256]  优化器: [adam ▾]            │   │
│  │ 权重衰减: [0.01]  日志间隔: [50]             │   │
│  └─────────────────────────────────────────────┘   │
│                                                    │
│  ┌─ 模型架构 ──────────────────────────────────┐   │
│  │ 模型维度: [128]  注意力头: [4]  层数: [4]     │   │
│  │ FFN 维度: [512]                              │   │
│  └─────────────────────────────────────────────┘   │
│                                                    │
│  位置编码 / 激活 / 归一化: [learned ▾] [gelu ▾] [layernorm ▾] │
│  梯度累积: [1]  stride: [0]   GPU: [None ▾]      │
│                                                    │
│  ▶ 高级选项（学习率调度 / TDR 防护 / 梯度日志 / 梯度裁剪） │
│    - 调度: fixed / cosine / step_cosine + 预热 + min-lr │
│    - flush 间隔（中间提交防 TDR）；梯度裁剪 max-norm       │
│                                                    │
│  ☑ 显示梯度统计                                    │
│                                                    │
│  [▶ 开始训练]  [⏹ 停止]                            │
│                                                    │
│  ┌─ 训练曲线 ────────────────────────────────┐   │
│  │  ╭──╮   step_loss / epoch_loss / lr          │   │
│  │ ╭╯  ╰──╮                                     │   │
│  │╭╯      ╰─────────                            │   │
│  └─────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────┘
```

**操作步骤：**

1. **选择数据集** — 点击"浏览…"选择 `.nndataset` 数据集文件（由 `dataset_gen` 生成；
   词表、loss 掩码、test 子集都随数据集）
2. **配置模型架构** — 模型维度、注意力头数、Transformer 层数、FFN 维度
3. **选择位置编码 / 激活 / 归一化** — `learned`/`sinusoidal`/`alibi`，`gelu`/`swiglu`，`layernorm`/`rmsnorm`
4. **配置超参数** — 轮数、学习率、批大小、序列长度、优化器；可开启梯度累积与滑动窗口 stride
5. **高级选项** — 学习率调度（cosine / step_cosine）、TDR 防护、梯度统计、梯度裁剪
6. **点击 "▶ 开始训练"** — 实时显示 loss 曲线

---

### GPT 推理 Tab

**操作步骤：**

1. **选择模型** — 指定训练好的 GPT 模型 `.bin` 文件
2. **选择词表** — 模型未嵌入词表时指定 `.nnvocab` 词表文件
3. **设置生成参数** — 最大 token 数（默认 200）、温度（0=贪心，>1=更随机，默认 1.0）
4. **输入提示文本** — 在输入框中输入起始文本（如 "Hello"）
5. **点击 "▶ 生成文本"** — 在输出区域显示生成结果
6. **多轮对话** — 在同一个输入框里输入这一轮的话 → 点 **"💬 发送（多轮）"**：
   每轮把**整段历史**（含模型自己之前的回复）交给 CLI，所以连续 N 轮上下文不丢；
   "对话记录"面板显示当前历史与轮数，**"🧹 清空对话"** 开一场新对话（下一轮从零起算）。
   系统提示可在上面的「系统提示（多轮对话）」框里改（对应 CLI `--system`）。
   采样参数（温度 / top-k / top-p / 重复惩罚 / 种子）**每轮都可以改** —— 多轮协议是
   无状态的（历史每轮随参数一起传回 CLI），不像 stdin 交互模式那样被启动参数钉死。

**可选项：**
- ✅ **显示 Token ID** — 调试用，显示每个 token 的 ID
- ✅ **GPU 加速** — 支持 Vulkan（下拉选择设备）

> **两种多轮实现的取舍**：GUI 走 `--turn` + `--reply-out` 的无状态协议（历史在
> `ChatSession` 里，每轮随命令行一起传回 CLI）——代价是每轮重新编码整段历史
> （本项目推理本就不带跨轮 KV cache，代价与一次 prefill 同量级），换来的是
> 每轮可改参数、进程崩了历史还在、GUI 无需解析 stdin 交互的输出流。
> CLI 的 `--interactive` 则把历史留在进程内（助手轮用生成的原 token 回灌），
> 适合终端里连着聊。

---

### 分词器训练 Tab

**操作步骤：**

1. **选择训练文本** — 用于训练分词器的 `.txt` 文件
2. **设置输出路径** — 分词器模型保存的 JSON 文件
3. **选择类型** — `bpe`（字节级 BPE）或 `charbpe`（字符级 BPE，中文推荐）
4. **配置参数** — 词表大小（默认 5000）、最小合并频率（默认 2）
5. **点击 "▶ 开始训练"**

---

### 分词器推理 Tab

**操作步骤：**

1. **选择词表** — 加载分词器 JSON 文件
2. **选择操作模式**：
   - **编码+解码验证** — 输入文本，查看编码再解码的完整流程
   - **仅编码** — 将文本编码为 token ID 序列
   - **仅解码** — 将 token ID 序列解码为文本
3. **输入内容** — 文本或逗号分隔的 ID 列表（按 Enter 快速执行）
4. **可选** — 勾选"显示 Token 详情"查看字节级信息
5. **点击 "▶ 执行"** — 输出区域显示结果
