# 参与本项目（给非作者的入口说明）

> 本文写给**不是本仓作者**的人：想用库、报 bug、提 PR、加层/加原语、写文档、或把一份实验结果放进 `research/`。
> 只回答「我怎么参与」。**项目自身的开发规范**（14 条铁律、分层依赖、门禁口径、踩坑）在
> [`AGENTS.md`](AGENTS.md) 与 [`docs/todo/10-development-standards.md`](docs/todo/10-development-standards.md)；
> 文档分册定位（哪篇写给谁看）在 [`docs/README.md`](docs/README.md)。

---

## 0. 先读哪篇

| 你想做什么 | 从这里开始 |
|-----------|-----------|
| 用库写自己的模型/训练 | [`docs/development/cpp-api-manual.md`](docs/development/cpp-api-manual.md)、[`docs/usage/`](docs/usage/) |
| 把库嵌进自己的 CMake 工程 | [`docs/development/cmake-integration.md`](docs/development/cmake-integration.md)、[`docs/usage/05-consume-as-library.md`](docs/usage/05-consume-as-library.md) |
| 写自己的层 | [`docs/development/write-a-custom-layer.md`](docs/development/write-a-custom-layer.md) |
| 扩展引擎原语 / 改 GPU 后端 | [`docs/development/01-compute-engine-development.md`](docs/development/01-compute-engine-development.md) + `AGENTS.md` §7 |
| **改本仓的代码** | [`AGENTS.md`](AGENTS.md)（**先读 §5 铁律与 §10 高频坑**） |
| 看"下一步该做什么" | [`docs/todo/18-roadmap.md`](docs/todo/18-roadmap.md) |
| 看"为什么当初这么改" | [`docs/history/history.md`](docs/history/history.md)（全仓唯一历史归档） |

---

## 1. 环境与首次构建

需要 **CMake ≥ 3.30**、**Ninja**、一个支持 C++26 的编译器（CI 用 Clang；GCC 15 / MSVC 19.51 也验证过）。
Vulkan SDK（含 `glslc`）可选：找到就自动启用 GPU 后端，找不到就是纯 CPU 构建。
装了 [ccache](https://ccache.dev/) 会自动启用（详见 §5）。

```bash
# 最省事：用仓库自带的 preset（CMakePresets.json）
cmake --preset release      # Release + -march=native，纯跑应用
cmake --build --preset release

# 本仓日常开发（Release + 全部测试）
cmake --preset dev && cmake --build --preset dev && ctest --preset dev

# 也可以一条命令走完「配置 + 构建 + 测试」
cmake --workflow --preset dev

cmake --list-presets         # 看全部 preset：release/dev/debug/portable/cpu/probes
```

不想用 preset 也行，等价于：

```bash
cmake -B build -G Ninja && cmake --build build            # 默认 Release
cmake -B build -G Ninja -DNN_ENABLE_TESTS=ON && cmake --build build && ctest --test-dir build
```

常用开关（完整列表见 [`docs/development/cmake-integration.md`](docs/development/cmake-integration.md)）：

| 开关 | 默认 | 用途 |
|------|------|------|
| `NN_ENABLE_NATIVE` | `ON` | `-march=native`。**分发/换机请设 `OFF`**（否则可能 SIGILL） |
| `NN_ENABLE_TESTS` | `OFF` | 编测试与 gradcheck（在 `build/test/` 下，注册 ctest） |
| `NN_ENABLE_PROBES` | `OFF` | 诊断探针（热路径扫描/归因 + `mem_probe` 等）。⚠ 会改变库头语义，**不可与正常构建混用缓存/二进制** |
| `NN_ENABLE_GPU` | `AUTO` | `AUTO`/`ON`（缺 Vulkan 直接配置失败）/`OFF`（纯 CPU 最短路径） |
| `NN_BUILD_APPS` | 顶层=ON | 嵌入本库当你自己工程的一部分时自动为 `OFF` |

---

## 2. 跑测试

```bash
ctest --preset dev              # 或 ctest --test-dir build --output-on-failure
ctest --test-dir build -R gradcheck   # 只跑匹配的
```

**本地必须跑**：CI 目前只做「构建 + 打包可执行文件」（见 §6），**不跑 ctest**——
那是 [`docs/todo/18-roadmap.md`](docs/todo/18-roadmap.md) 里 P0-1 的待办。所以测试是你自己的门禁。

改到下面这些区域时，请把对应专项测试一起跑（都在 ctest 里）：

| 改动区域 | 至少跑 |
|---------|--------|
| 引擎原语 / 张量 | `*_test`（含 `f16_precision_test`、`write_epoch_test`） |
| 训练/反向 | 各 `gradcheck*`、`*_train_test` |
| GPU 后端 / DSL 融合 | 带 `--gpu` 的变体（见 [`docs/todo/12-compute-engine-inventory.md`](docs/todo/12-compute-engine-inventory.md) 的清单） |
| 三值 / BitLinear | `bitnet_struct_test`、`ternary_*`、`adam_quant_test` |
| 序列化 / 数据集 | `kvrec_test`、`nnvocab_test`、`dataset_test` |

```bash
# GPU 测试要在能看见 /dev/nvidia* 的环境里跑（容器/沙箱内会静默退回 llvmpipe 软件光栅）
NN_VULKAN_DEVICE=850M ./build/test/bitnet_struct_test --gpu     # 显式选卡
```

---

## 3. 代码规范（怎么不被 review 打回）

### 3.1 先记住这 5 条（其余见 `AGENTS.md` §5 的 14 条铁律）

1. **不许 `throw` / `try` / `catch`**（`-fno-exceptions` 强制）。错误一律 `Result<T> = std::expected<T, Error>`，
   手写错误检查用宏族：传播 `NN_TRY/NN_TRY_MSG/NN_TRY_CHECK`、终止 `NN_UNWRAP/NN_EXIT`、源头 `NN_FAIL`。
   前置条件校验用 `NN_ASSERT_ALWAYS` / `NN_FATAL`（NDEBUG 下同样生效）。
2. **不许 `new`/`delete`/裸指针所有权**：`std::vector` / `std::unique_ptr` / `std::span`。
3. **不穿透分层**：Matrix(L1) 不写算法、Layer(L2) 不写底层计算、shader 里不写 ReLU/Softmax 这类算法。
4. **张量存储不许绕过引擎**：建/读/写/reshape 一律走 `ComputeEngine`（`Tensor` 的存储访问器是私有的，
   绕过去编译不过）。
5. **改动要可复现**：涉及数值/性能的改动，请在 PR 描述里给出**实测数字与跑法**（本仓历史提交都这么做，
   见 [`docs/history/history.md`](docs/history/history.md)）。

### 3.2 代码风格

风格规则写在 [`docs/todo/10-development-standards.md`](docs/todo/10-development-standards.md) §8
（命名 / 缩进 4 空格 / include 顺序 / 注释语言），机器可读版本是同目录的 [`.clang-format`](.clang-format)。

```bash
# ✅ 推荐：只格式化**你这次改动**的行（历史文件是手工排版的，整文件格式化会产生大 diff）
git clang-format                 # 对暂存区相对 HEAD 的改动
git clang-format --diff          # 只看会改什么

# 单文件（新文件随意；旧文件请确认 diff 可接受）
clang-format -i path/to/new_file.hpp
```

> `.clang-format` 是**目标风格**，不是整仓重排脚本：对现有文件整体跑它约 4 成行会变
> （仓内 `if (c) stmt;` 单行与两行两种写法都大量存在）。**请不要在功能 PR 里顺手格式化无关文件。**
> 缩进/行尾/编码这类字符级约定由 [`.editorconfig`](.editorconfig) 与 [`.gitattributes`](.gitattributes) 统一（**全仓 LF**）。

### 3.3 提交信息

沿用本仓既有写法：**一行中文标题**（可用「范围：结论」形式）+ 需要时补正文解释**成因与验证方式**。

```
修复合并引入的视图种类硬编码：nn::expr::trans() 写死 15

合并时 HeadGroup / Transpose 各占 15 → Transpose 顺延为 16，但 `nn::expr::trans()`
仍返回 {15, 0, stride} → 静默构造出 HeadGroup 视图。症状：...
改为 static_cast<std::uint8_t>(ExprViewKind::Transpose) 并加注释。
实测（...）：修前 32/40（8 挂），修后见下一次全量 ctest。
```

要点：**说清"为什么"和"怎么验证的"**；改了行为/数字就给出前后对比；没验证的部分明说没验证。

---

## 4. 提 PR

1. Fork（或在本仓开分支），分支名 `<类型>/<主题>`，类型用 `feat` / `fix` / `docs` / `bench` / `refactor`
   （例：`fix/trans-view-kind`、`docs/consume-as-library`）。
2. **一次 PR 一件事**。大改动请先在 issue/讨论里对齐设计——本仓有「先写设计稿进 `docs/todo/`，
   裁定后再实施」的习惯（`docs/todo/` 就是项目开发者的工作区）。
3. 提交前自查：
   ```bash
   cmake --preset dev && cmake --build --preset dev && ctest --preset dev   # 测试必须绿（本机口径 40~42 项：CPU 40 / 开 GPU 42）
   python bench/gui_cli_audit.py         # 改了 gui.py / cli_controllers.py / 任一 CLI 选项后必跑（当前 exit 0）
   python scripts/audit_modern_style.py  # 现代 C++ 风格（裸指针 / new / delete / malloc）；⚠ 见下
   git clang-format --diff               # 风格：只格式化**你改动的行**
   ```
   ⚠ `audit_modern_style.py` 对本仓当前是 **FAIL（59 E / 147 W）**——E 几乎全部落在 Vulkan C API
   边界（`reinterpret_cast<VkBaseInStructure*>` 的 pNext 串链、`void* mapped`、`std::free`）与一处
   **刻意的**进程级单例 `new`（`compute_vk_backend.hpp` 的 `static GpuBackend* backend`）。所以它是
   **基线对照**工具：看**你改动的那几行**有没有新增 E，别指望整仓绿。要让它成为可执行门禁，得先给
   这些 C 边界补 `// nn-allow: <原因>` 豁免（脚本支持该机制，当前全仓 0 处豁免）——那是独立的一批改动，
   不在功能 PR 里顺手做。
   Windows 侧另有 `pwsh -File bench/doc_align_audit.ps1`（文档 ↔ 代码对齐审计：
   文件/符号/CLI/数字/测试名）与 `bench/doc_inventory.ps1`（引擎接口盘点 + L2 分层门禁），
   需要 PowerShell 7，改了文档或引擎接口时跑。
4. PR 描述里写：**动机 → 改了什么 → 怎么验证的 → 影响面/兼容性**（模型文件格式、CLI 参数、
   数值结果是否有变化）。破坏性变更请附迁移说明。

---

## 5. 关于构建速度（ccache 是本仓的默认路径）

本仓是 header-only，一次全量 Release + 测试构建在本机 8 核上约 **13 分钟**（大头是 LLVM 后端优化，
单个 TU 约 30s）。仓库已经做了这些事，你不需要额外配置：

* **PCH**：`nn.hpp` 只预编译一次，被 40+ 个目标共享（`NN_ENABLE_PCH`，默认开）。
* **ccache 自动启用**：顶层构建会自己探测 `ccache`/`sccache`；装了就用。
  仓库会把 ccache 需要的 PCH 兼容设置（`sloppiness=pch_defines,time_macros` +
  `-fno-pch-timestamp`）**钉进编译命令**，因此"用了 PCH 的 TU"也能命中缓存——
  二次构建（含 `ninja -t clean` 之后）本机实测 **58.8s**（全量 13m26s → 58.8s），剩余时间是 shader 生成与链接、不是编译。
* 想指定/禁用它：`-DNN_COMPILER_LAUNCHER=<path>`（空串 = 禁用）。

日常改一个文件的循环：`cmake --build --preset dev` 会自动只重编受影响的目标；
只是想验证某个改动不影响别处时，优先跑对应测试而不是整仓重建。

---

## 6. CI 会做什么

[`.github/workflows/cmake-single-platform.yml`](.github/workflows/cmake-single-platform.yml)：

* Linux + Windows，**Clang**，Release，`NN_ENABLE_NATIVE=OFF`（可移植基线），带 Vulkan SDK（GPU 编译）；
* `-DNN_ENABLE_TESTS=ON` 构建并**真跑 `ctest`**（需 Vulkan 的用例在无 GPU 的 runner 上按退出码 77 记 skip，非失败）；
* 再跑一轮 `NN_BIND_DEBUG=1 ctest`（**绑定不变量门禁**：任何"未绑定输入进引擎"一律报错，抓库内 stamp 漏网）；
* 构建 **8** 个应用入口（`mnist_train/infer`、`text_train/infer`、`tokenizer_train/infer`、`dataset_gen`、`dataset_convert`）并打包成 `.7z` 产物；
* push 到 `main` 时创建/更新 `nightly` 预发布。

> 本地复现 CI 的三条门禁：`ctest --preset dev`、`NN_BIND_DEBUG=1 ctest --test-dir build`（都应全绿）。

---

## 7. 文档贡献

分册定位只有一条总则（[`docs/README.md`](docs/README.md)）：
**除 `todo/` 与 `history/history.md` 外，`docs/` 下的文档都是写给"库外的人"的。**

| 目录 | 读者 | 收什么 |
|------|------|--------|
| `docs/introduction/` `docs/usage/` | 调用者 | 总览、教程 |
| `docs/development/` | **调用者**（扩展库的人） | 接入、API 手册、写自定义层、精度契约 |
| `docs/todo/` | 项目开发者 | 设计稿、路线图、盘点、规范、踩坑、整改 |
| `docs/history/history.md` | 项目开发者 | 全仓唯一历史归档（演进/勘误/被否决方案/A-B 数字） |

一句话判断：**这篇文档的读者是"库外的人"还是"改库的人"？** 后者进 `todo/`。

顺手能做的：文档里的 CLI 参数、文件路径、符号名请与代码核对（`bench/doc_align_audit.ps1` 就是干这个的）。

---

## 8. 实验与结论（`research/`）

`research/` 是**可引用的实测结论**归口：一次研究一个子目录（报告 + 脚本/探针 + 汇总数据），
结论要能被人**不读源码地引用**（给出结论、证据、复现命令、适用边界）。见 [`research/README.md`](research/README.md)。
不要把中间日志/权重仓进来（`.gitignore` 已忽略 `*.log` / `*.bin`，报告里给出重跑命令即可）。

---

## 9. 数据与第三方内容

代码是 MIT（见 [`LICENSE`](LICENSE)）。但 `scripts/` 下的下载脚本拉取的**数据集各有自己的许可证**
（TinyStories / OpenWebText / 中文 wiki 等），**不随本项目 MIT 分发**——请自行遵守其条款，
也不要把数据集或训练产物提交进仓库。

---

## 10. 行为准则

参与本项目即表示同意遵守 [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md)。

## 11. 许可证

提交贡献即表示你同意以本项目的 [MIT License](LICENSE) 授权你的贡献。
