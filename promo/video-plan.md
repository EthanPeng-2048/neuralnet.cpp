# neuralnet.cpp 宣传视频制作方案

> 目标：B 站 / YouTube / X 主视频一支（6.5–8 分钟），配套封面 + 8 张配图 + 可复用的文章素材。
> 分工：**我**出大纲/文案/配图内容/分镜；**你**按 §5 拍摄清单录屏、按 §4 画配图。
> 所有数字均为 2026-08-30 本机实测，拍前可再跑一遍确认。

---

## 1. 定位与标题

**一句话定位**：不靠 PyTorch、不靠 CUDA，C++26 从零实现、能真训 GPT 的神经网络库。

**标题（三选一，按平台风格）**

| 平台 | 标题 |
|------|------|
| B 站（主推） | 用 C++26 手写 GPT 训练器！不靠 PyTorch、不靠 CUDA |
| YouTube | I Built a Neural Network Framework in C++26 That Trains GPTs — No PyTorch, No CUDA |
| 备选（悬念向） | 34MB 的模型，自己生成的故事：一个从头写的神经网络框架 |

**封面大文案**：`C++26 手写 GPT` / 副标 `No PyTorch · No CUDA · 34MB`
（封面视觉见 §4 图1）

---

## 2. 数据 fact sheet（文案与字幕唯一数字来源，勿临时改口）

| 项 | 值 | 备注 |
|----|----|----|
| 语言/标准 | C++26，Clang++ 22.1+ | `-fno-exceptions -Werror` 全项目无 throw |
| 库形态 | header-only，唯一入口 `nn.hpp` | 零第三方依赖 |
| 后端 | CPU 多线程 + Vulkan GPU | CMake 自动探测，无 GPU 纯 CPU 跑 |
| GPT 模型规格 | vocab 8206 / d_model 256 / heads 4 / 4 层 / d_ff 1024 / seq 128 / learned 位置编码 | 模型内嵌 BPE tokenizer |
| 模型文件大小 | 34 MB（34,214,047 B） | float32，约 8M 参数（按字节数估算） |
| 训练数据 | TinyStories 200MB 英语小故事 | `datasets/tinystories_200mb.txt` |
| CPU 推理 | **102 tokens/s** | 80-token 生成实测 0.8s |
| GPU 推理 | 53 tokens/s（Vulkan） | 小模型下 CPU 反超，归因 staging 往返 PCIe——可当诚实细节讲，也可不拍 |
| MNIST 模型 | 1.6 MB | 4 层 MLP，可本机几分钟训出 |
| 测试 | **30/30 全绿**（ctest） | 2026-08-30 复测 |
| 文档 | docs/ 17 章 + 开发规范 | `08-pitfalls-and-lessons.md` 是推荐阅读 |
| 许可 | MIT | AI 辅助开发（DeepSeek V4 / Mi Mimo v2.5 / VS Code Copilot）——文案里如实说，是加分项不是减分项 |
| 不声称 | CUDA 后端（1.0.0 已停用）、"比 PyTorch 快/慢"（无对照 benchmark） | 视频字幕里也别出现 |

---

## 3. 视频大纲（总长 7:00–8:00）

| # | 段落 | 时长 | 素材类型 | 目的 |
|---|------|------|---------|------|
| S1 | 开场：结果先行 | 0:00–0:30 | 文字卡 + R2 录屏 3 秒闪回 | 3 秒内抓住人 |
| S2 | 项目是什么 | 0:30–1:15 | 口播 + 图2 架构层 | 定位 + 与"玩具项目"划界 |
| S3 | Demo 1：GPT 文本生成 | 1:15–2:15 | R2 + R3 录屏 | 核心震撼点 |
| S4 | Demo 2：MNIST 手写识别 | 2:15–3:15 | R6–R7 录屏 | 展示完整度（视觉+GUI） |
| S5 | 架构：引擎化 + 铁律 | 3:15–4:45 | 口播 + 图3/图4 | 技术人群留存的钩子 |
| S6 | 工程：AOT 融合管线 | 4:45–6:00 | 口播 + 图5 | 最大差异化（编译器式） |
| S7 | 工程：显存 + 两趟注意力 | 6:00–6:40 | 口播 + 图6 | 对标 FlashAttention，提可信度 |
| S8 | 原创算法：AttnZip / RAPT | 6:40–7:25 | 口播（可不配图或复用图2） | 学术向内容预告 |
| S9 | 性能与工程细节 | 7:25–7:55 | 图7 数据卡 | 硬数字收尾 |
| S10 | 结尾 CTA | 7:55–8:25 | 口播 + 图8 命令卡 | Star / 仓库 / 下期预告 |

节奏原则：前 2 分钟全是"看得见的结果"，工程内容从 3:15 才开始——留人靠 demo，不靠架构。

---

## 4. 配图内容（8 张，16:9，深色科技风，扁平矢量）

统一风格：深蓝/黑底（#0d1117 系）、青绿色主色、等宽字体展示代码、无照片。
可用 AI 生图或 PPT/矢量工具画。每张给出"画面元素 + 图上文字 + 出现时机 +（AI 生图 prompt）"。

### 图1 封面（视频封面 + 文章头图）
- **画面**：中央一个终端窗口 mockup，里面是故事文本的最后一行 + 闪烁光标；背景隐约的矩阵/网格纹理。
- **图上文字**：主标 `C++26 手写 GPT`（大号）；副标 `No PyTorch · No CUDA · 34MB 模型`；右下角 `102 tokens/s`。
- **时机**：封面 + 片头 0:05 定格 2 秒。
- **AI prompt**：`Dark tech style YouTube thumbnail, terminal window with glowing green story text, title "C++26 from-scratch GPT", badges "No PyTorch No CUDA 34MB", flat vector, 16:9, no photo`

### 图2 六层架构图
- **画面**：从上到下 6 个横条（L5→L0），每层左侧标层名、右侧列代表物：
  - L5 入口：`mnist_train / text_infer / gui.py`
  - L4 领域：`build_gpt_model / Tokenizer`
  - L3 模型：`Model / 序列化`
  - L2 计算：`Layer / Loss / Optimizer / Engine`
  - L1 代数：`Matrix / 表达式模板`
  - L0 硬件：`Tensor 存储 / 线程池`
  层间用单向下箭头，右侧竖排大字 `只许向下依赖`。
- **时机**：S2（0:35–1:15）。
- **AI prompt**：`Clean architecture diagram, 6 horizontal layers stacked L5 to L0, dark background, teal accents, one-way down arrows, flat vector, 16:9`

### 图3 引擎化："写一次，双端跑"
- **画面**：左侧一个 `Layer` 框（内写 `forward / backward 各一次`），向右分出两条箭头到两个引擎框：`CPU Engine（线程池分块）` 与 `Vulkan GPU Engine（command buffer）`；底部一行小字 `新增后端 = 只实现 Engine 接口`。
- **时机**：S5 前半（3:20–4:10）。
- **AI prompt**：`Diagram: one Layer box on left, branching arrows to two engine boxes (CPU threads / Vulkan GPU), label "write once, run anywhere", dark tech flat vector, 16:9`

### 图4 铁律三栏
- **画面**：三列卡片，每列顶部一个勾或"红线"标记：
  - `Layer：只写算法`（`ReLU = max(x,0)` 小代码块）
  - `Engine：只提供原语`（`matmul / reduce / exp` 小代码块）
  - `Shader：只是实现`（`不含任何算法名`）
  底部横条：`-fno-exceptions：全项目零 throw`。
- **时机**：S5 后半（4:10–4:45）。
- **AI prompt**：`Three vertical cards on dark background: "Layer: algorithm only" "Engine: primitives only" "Shader: implementation only", small code snippets, red line motif, flat vector, 16:9`

### 图5 AOT 融合管线（全片最重要的一张图）
- **画面**：横向 6 段流水线，左到右：
  1. C++ 代码块（`exp(x - row_max) / col_sum(...)` 这种表达式）
  2. `scan_exprs`（构建期，标注"dry-run 收集结构"）
  3. `expr_specs.bin`（小文件图标）
  4. `gen_fused` → `glslc`（生成融合 shader / SPIR-V 内联）
  5. 运行时 `Layer` 方框，虚线箭头到 4 的产物，标注 `按 key 查表`
  6. 右侧红色大字 `无运行时编译 · 无动态图 · 无 autograd`
- **时机**：S6（4:45–6:00）。
- **AI prompt**：`Horizontal pipeline diagram, 6 stages: C++ expression code block, scan tool, binary spec file, generator to shader, runtime lookup by key, red label "no runtime compilation", dark tech flat vector, 16:9`

### 图6 两趟注意力 vs 朴素注意力
- **画面**：上下两行对比。上行 `朴素`：Q·Kᵀ 生成一块大的红色 `S×S 分数矩阵`（标 `O(L²) 显存`），再 softmax 再乘 V。下行 `两趟式`：同一输入，第一趟只算行级 max/sum 两个标量，第二趟在线归一化直接出结果，中间只有 `d 维累加器`（绿色小方块），标 `不物化 S×S`。
- **时机**：S7（6:00–6:40）。
- **AI prompt**：`Comparison diagram top vs bottom: naive attention materializes large red S-by-S score matrix, two-pass attention uses only small green d-dimensional accumulators, labels O(L^2) vs no materialization, dark flat vector, 16:9`

### 图7 性能数据卡
- **画面**：2×3 大数字网格：`34MB 模型` / `102 tokens/s (CPU)` / `30/30 测试全绿` / `0 第三方依赖` / `17 章文档` / `MIT`。每格一个数字 + 一行小字。
- **时机**：S9（7:25–7:55）。
- **AI prompt**：`Stats dashboard grid, 2x3 large numbers: 34MB, 102 tok/s, 30/30 tests, 0 dependencies, 17 docs, MIT, dark background teal glow, flat vector, 16:9`

### 图8 一键运行卡（结尾）
- **画面**：终端风格卡片，三行命令 + 一句注释：
  ```
  git clone <repo>
  cmake -B build -G Ninja && cmake --build build
  ./build/text_infer --model pretrained/gpt_model_tinystories_200mb.bin --prompt "hi"
  ```
  底部小字：`有 Vulkan 自动上 GPU，没有就纯 CPU` + `MIT · AI-assisted`。
- **时机**：S10（7:55–8:25）。
- **AI prompt**：`Terminal card with three lines of commands, dark background green monospace text, small caption "auto Vulkan / CPU fallback", flat vector, 16:9`

> 图2–图8 在视频里各停留 10–40 秒，口播压在上面；同一张图不要整段静止——加缓慢缩放（Ken Burns 2–3%）防呆。

---

## 5. 拍摄清单（你拍什么，逐条可执行）

**总原则**：1080p 以上、16:9；终端字体 ≥14pt、深色主题；录前关掉无关窗口/浏览器标签；每条素材首尾留 2 秒黑场/静场；每条至少录 2 遍挑一条。手机/屏幕录制软件均可。

| 编号 | 内容 | 操作（在 `/home/ethan/codes/neuralnet.cpp` 下） | 时长 | 备注 |
|------|------|------|------|------|
| R1 | 构建过程 | `cmake -B build -G Ninja && cmake --build build` | 录 15–20s，后期 4× 加速 | 体现"一条命令"，不必录完 |
| R2 | **GPT 生成（核心素材）** | `./build/text_infer --model pretrained/gpt_model_tinystories_200mb.bin --prompt "Once upon a time, a little girl lived in a forest" --max-tokens 80` | 20s | **录 5 遍挑最好**；保留开头的"模型规格/词表大小"行——那是信息量；结尾的 `102 tokens/s` 行必须完整 |
| R3 | 交互模式 | `./build/text_infer --model pretrained/gpt_model_tinystories_200mb.bin --interactive`，输入 2–3 轮对话 | 15s | 提示词自己即兴，选自然的；退出不慌 |
| R4 | GPU 版（可选） | 同 R2 加 `--gpu` | 10s | 显示 53 tok/s；若讲"小模型 CPU 反超"就保留并配诚实旁白，否则不剪 |
| R5 | GUI 启动 | `python gui.py` | 3s | 作为 S4 转场 |
| R6 | GUI MNIST 训练页 | 打开 GUI → MNIST 训练 Tab，展示参数面板；**建议真跑 1–2 个 epoch**（MLP、小 epochs、batch 256）让 loss 曲线动起来 | 20–30s | 曲线动 = 有说服力；参数面板特写 3s |
| R7 | 手写板识别 | GUI → MNIST 推理 → 手写板，写 `7`、`3`、`0`、`2` 各一次 | 每条 5s | 至少 3 个数字；保留 Top-K 置信度条形图特写；**允许错一个**（更真实，Top-5 里对就行） |
| R8 | 测试全绿 | `ctest --test-dir build` | 8s | 结尾 `100% tests passed, 0 tests failed out of 30` 必须完整入镜 |
| R9 | GPT 训练日志（可选） | `./build/text_train datasets/everyday_conversations_train_sft.txt --vocab <词表json> --epochs 1 --seq-len 128 --batch-size 32 --save /tmp/demo_model.bin` | 录前 60s 日志 | 体现"真在训练"；时间紧就剪进 S2 当背景 5s |
| R10 | 仓库页 | GitHub 仓库页（README 滚动到 demo 段）+ `ls docs` | 8s | 结尾 CTA 用 |

**拍摄顺序建议**（一天内完成）：R8（30s 出结果）→ R2×5 → R3 → R5–R7（GUI 连着拍）→ R1 → R9/R10。
**拍前检查**：`ctest --test-dir build` 再跑一遍确认 30/30；确认 `pretrained/gpt_model_tinystories_200mb.bin` 在位。

---

## 6. 口播文案（完整稿，约 1750 字 ≈ 7.5 分钟）

> 按 §3 分段。括号内是剪辑提示，不念。语速 200–240 字/分，口语化，可微调。

**S1 开场（0:00）**
你看到的这段故事，从第一个字到最后一个字，都是一个 34MB 的模型自己生成的。（R2 闪回 3 秒）没有 PyTorch，没有 TensorFlow，没有 CUDA——训练它的框架，是从零用 C++26 写出来的。我叫它 neuralnet.cpp。（图1 定格 2 秒）今天这期，给你看它是什么、怎么跑、以及里面几个我认为真正有含金量的工程设计。

**S2 项目是什么（0:30）**（图2）
一句话说清楚：这是一个从零实现的神经网络库。header-only，一个头文件就是整个库，MLP、ViT、GPT 都能训、能推理。两个后端：CPU 多线程，和 Vulkan GPU——注意，不是 CUDA，任何支持 Vulkan 的显卡都能用，包括核显。它和 GitHub 上那些"手写神经网络"项目最大的区别是：那些大多停在 MNIST 教学层面，而这个是真的能训语言模型的。片头这个模型，用 200MB 英语小故事训出来的，4 层、256 维、34MB，大约 800 万参数。

**S3 GPT 文本生成（1:15）**（R2 全程）
先看结果。一条命令，给模型一个故事开头，它接着往下写。（停顿让它生成 2 秒）这是纯 CPU，102 个 token 每秒。（R3）也可以交互，一句一句地聊。质量你判断——它显然还是个中小模型，但请记住：它是被一个没有一行框架代码的系统训出来的。

**S4 MNIST 手写识别（2:15）**（R5→R6→R7）
除了语言，视觉也可以。这是项目自带的 GUI。MNIST 训练页，超参数全在这里，点开始，loss 曲线实时画出来——这个模型 1.6MB，本机几分钟就能训好。（切到手写板）这里有个手写板，我写个 7，识别，Top-K 置信度在这；再来个 3、0。全链路：你手写的像素，直接进模型出结果。

**S5 架构：引擎化（3:15）**（图3→图4）
那它怎么实现的？核心就一个设计：引擎化。Layer 只写一次 forward 和 backward，调用的时候传一个 ComputeEngine 进去——CPU 引擎，或者 GPU 引擎，同一份代码两端跑。（停顿）这不是技巧，是项目的铁律：引擎只提供 matmul、reduce、exp 这类底层原语，永远不认识"softmax""attention"这种名字；算法只允许写在 Layer 里。为什么较真？因为引擎只认结构、不认算法名，算子融合、IR 优化、新后端，就都有了同一张稳定的契约。以后加一个后端，只写一个 Engine 接口，Layer 一行不动。（图4）另外两条红线：`-fno-exceptions`，全项目零 throw，错误走 `std::expected`；零手动内存管理，没有一处 new/delete。

**S6 AOT 融合管线（4:45）**（图5）
第二个我最想讲的，是编译器式的表达式系统。Layer 里的 softmax、注意力，用普通 C++ 数学表达式写。构建期两个工具接管：scan_exprs 把每个 Layer 空跑一遍，收集所有表达式的结构；gen_fused 按结构生成融合的 GLSL compute shader，编译成 SPIR-V 直接内联进 C++ 头文件。运行时，表达式按 key 查预编译 shader——没有运行时编译，没有动态计算图，没有 autograd。整张图在构建期就固定死了。我们叫它闭合世界的 AOT。好处很直接：GPU 上中间结果不落显存。

**S7 显存与两趟注意力（6:00）**（图6）
显存这块，最典型的是注意力。两趟式写法，思路和 FlashAttention 同源：第一趟算每行的 max 和 sum，第二趟在线归一化直接出结果，那个 O 平方于序列长度的分数矩阵，从头到尾不物化。（停顿）再配合稀疏交叉熵——大词表下不物化 one-hot，这一项就省下 3GB 级显存——加上梯度检查点和显存池，我们把这个规模模型的训练峰值从 29GB 一路往下压。细节在仓库 docs/10，不展开了。

**S8 原创算法（6:40）**
工程之外，还有两个自研的注意力算法。AttnZip：一组可学习的"记忆查询"，把长上下文压缩成少量记忆 token，注意力复杂度从 O(L²) 降到 O(L)，为长文场景设计。RAPT：利用 RoPE 的几何衰减特性，把明显不重要的 token 直接门控掉，省计算。两个都有完整设计文档，代码在 Layer 库里，欢迎来挑毛病。

**S9 性能与工程细节（7:25）**（图7）
几个数字，全部普通工作站实测：模型 34MB，CPU 推理每秒 102 token，30 项测试全绿，零第三方依赖。文档 17 章，从架构到踩坑——踩坑那章我特别推荐。

**S10 结尾（7:55）**（图8）
项目开源，MIT，链接放描述区。如果你做 C++，想看一个不依赖任何框架的完整神经网络实现；或者你只是想搞清楚 GPT 训练底下到底在发生什么——star 它，跑一下，来 issue 里聊。我们下期见。

---

## 7. 后期与发布

**剪辑**
1. 顺序按 §3；S1 的 3 秒"闪回"用 R2 的生成结果段提速 2×。
2. 口播压在全片上；R2/R6 段落可留 1–2 秒环境音（键盘声）增加真实感。
3. 字幕：全程硬字幕（B 站自动生成后校对，重点数字 102 tokens/s、34MB、30/30 手动核对）。
4. 图2–图8 各加 2–3% 缓慢缩放；切图用 0.3s 交叉淡入，不用花哨转场。
5. BGM：低音量技术感电子（无歌词），R2 生成瞬间可把 BGM 压到 -18dB 再抬回。
6. 片尾 5 秒黑屏：仓库名 + 文档链接 + "MIT / AI-assisted"。

**描述区（B 站版模板）**
```
一个从零实现的 C++26 神经网络库：CPU/Vulkan 双后端，能真的训练 GPT。
- 仓库：<url>
- 文档：docs/ 17 章，推荐 01 架构 / 08 踩坑警示录 / 12 创新设计
- 本视频模型：TinyStories 200MB 训练，4 层 / 256 维 / 34MB / vocab 8206
- 实测：CPU 102 tokens/s，测试 30/30
- 本项目部分代码由 AI 辅助生成（DeepSeek V4 / Mi Mimo v2.5 / VS Code Copilot），版权归作者所有
```

**发布节奏**
1. D0：B 站 + YouTube（同片双语标题）；X/Twitter 发 60s 精简版（S1+S3+S10 剪掉，配 R2 最精彩 10 秒 + 图1 封面）+ 仓库链接。
2. D+1：Hacker News `Show HN`（标题：`Show HN: Neural network framework in C++26 that trains GPTs – no PyTorch, no CUDA`），正文引用 fact sheet 数字 + 一句 AI-assisted 声明。
3. D+3：知乎/掘金长文：把 S5–S8 扩成 4000 字技术文《AOT 闭合世界：把表达式当一门可编译的语言》。
4. 之后：docs/15（RAPT）、docs/16（AttnZip）各出一期视频或文章——内容池够再更 4–6 期。

**风险提醒**
- 不声称 CUDA 支持；不出现"比 PyTorch 快/慢"字样（无对照 benchmark，被问就答"未对照测试，欢迎跑"）。
- AI 辅助声明保留（README 已有），主动说比被挖出来好。
- R7 手写识别若连续失败，换数字重录，不必"保留失误"——真实感靠 R2 的模型质量自带，不靠 bug 表演。
