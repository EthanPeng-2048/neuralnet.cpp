# 统一 Tensor / ComputeEngine / MemoryPool 底层架构总纲（2026-09-30）

> **本文是新总纲**：吸收并取代 `15-computeengine-refresh.md` 未实施的 P2-P6（本文 §5 分期），
> 并对 `13-refactor-backlog.md` §10.8 增补"访问不变量"注脚。历史归档见 `docs/history.md`。
> 状态：**设计裁定已完成（§3 共 10 项），未实施**。实施立项时读本文 + §5 分期 + 源码现状。
> 前置工程成果保留不动：15 的 P-1（PrecisionEngine 下沉）、P1（张量出生绑定）已落地（15 §7.1）。

## 0. 一句话

**一切 Tensor 的创建、访问、修改只能经由 ComputeEngine**；Matrix 降级为宿主 I/O 载体；
内存池等底层资源统一挂在引擎契约上——**契约统一、实现自由**。

## 1. 为什么现在是乱的（历史成因 + 现状事实）

**成因**（维护者口述，如实记录）：

1. 项目起于纯 CPU：没有引擎、没有内存池，计算 = `Matrix` 的运算符重载。
2. 引入 `ComputeEngine` 是为了服务 GPU 后端，**CPU 侧没有随之迁移**——引擎化只做了一半。
3. 底层因此双轨并行：上层（Layer/Model/Loss）已用父类+继承收口，底层（Tensor/存储/内存）
   却是"静态工厂直构 + 公开存储访问 + GPU 专属内存池"的松散组合。

**现状事实**（可复现，2026-09-30 核对）：

| 事实 | 数字 | 复现 |
|---|---|---|
| `Matrix` 出现（include 213 + src 373 次命中） | 61 文件 / 586 处 | `Select-String` 扫 `include/**/*.hpp` + `src/**/*.cpp`，pattern `\bMatrix\b`（命中行；src 若含 `.hpp` 另 +3 行/1 文件） |
| `cpu_matrix()/cpu_shared()` 调用 | include 5 文件（CPU/GPU 引擎 + 引擎基类 + DSL + 容器头）+ src 127 处（126 测试 + `text_train.cpp:756`） | `grep -rn "cpu_matrix()"` |
| `gpu_tensor()` 调用 | **全部**在 `compute_gpu_engine.hpp` 内部 59 处（引擎自己，合规）；`compute_tensor.hpp:276/294` 仅为访问器声明 | 同上换 pattern |
| `Tensor` 静态直构工厂（`Tensor::from_matrix/cpu/...`） | 98 处 / 8 文件（2026-09-30 复核；库外为主，15 §1 口径同源） | `Select-String` 两个 pattern（`Tensor::from_matrix`、`Tensor::cpu\b`）扫 `src/*_test.cpp`，命中行和 = 98 |
| 内存池 | 仅 Vulkan：`backend/compute_memory_pool.hpp`（`GpuBackend` 持双池 memory/transient）；`backend/compute_staging_ring.hpp:56/93/104/232` 与 `compute_vk_backend.hpp` 亦引用，均 Vulkan 后端 | `grep -rn MemoryPool include/` |
| CPU 池 | **不存在**：`Matrix` 直接 `std::vector`；`pool_stats()` CPU 返回空串（`compute_engine.hpp:144`） | 读引擎默认实现 |
| init RNG | 混乱：固定 seed 42 = `gpt:432/603`、`rapt:1155`、`zipt:814`；`std::random_device{}`（**跨进程不确定**）= `gpt:953`、`mlp:40`、`conv:107`、`zipt:89/1044`、`rapt:1462` | `grep -rn mt19937 include/` |
| 读 GPU 张量 | `to_matrix` 是唯一 PCIe 下载点；`ensure_gpu` 43 处隐式上传（15 §1） | `grep -n "= ensure_gpu(" compute_gpu_engine.hpp` |
| Layer init 模式 | 层内 `Matrix` 填数 → `engine.from_matrix` 上传（如 `compute_layer_mlp.hpp:46-64`） | 读 `Layer::init` |

## 2. 目标与非目标

**目标（五统一）**

1. **访问统一**：Tensor 存储私有，创建/读写/修改全部走引擎接口——不存在"绕过引擎摸张量"。
2. **创建统一**：`create_tensor(rows, cols, P, InitSpec)` 声明式初始化（全零/常数/分布），
   初始化策略（host 生成上传 vs 设备端原生生成）由引擎自选，**调用方不可见**。
3. **读写统一**：批量 `read/write` 是本体（GPU 自动 staging），索引级 `get/set` 是语法糖；
   数据集/预训练模型经 Matrix（宿主 I/O 载体）+ `from_matrix/to_matrix` 进出。
4. **内存契约统一**：`pool_stats`/`release_idle_pool_blocks` 在所有引擎上存在且语义完整
   （CPU 也如实上报），池实现按设备自由。
5. **数据格式统一**：L2+ 计算路径只见 Tensor；Matrix 只活在 I/O 层与引擎内部实现。

**非目标（明确不做）**

- 不改 AOT 闭合世界、batch-major 布局、`PrecisionProfile` 归属、DSL 表达式签名。
- 不强求 CPU/GPU **实现**对称（§4.5）。
- 不删除 Matrix（裁定 D4：保留降级）。
- 不动已完成的 15 P-1/P1 成果。
- 不承诺索引级 API 在 GPU 上的热循环性能（§7 风险 1）。

## 3. 核定裁定（10 项）

| # | 裁定 | 来源 |
|---|---|---|
| D1 | 与 15 号文档关系：**合并为新总纲**，15 降级为历史引用，P2-P6 由本文 §5 分期吸收 | 维护者选定 |
| D2 | Tensor 形态：**形态不重要，不变量重要**——Tensor 保持值语义类；"父类约束访问方式"的诉求由"存储私有 + 引擎唯一访问"不变量实现；13 §10.8 否决的句柄指针化**维持否决** | 维护者口述 + 确认 |
| D3 | 直接读写：**引擎级批量 read/write**（GPU 自动 staging），index 级为语法糖 | 维护者选定 |
| D4 | Matrix：**保留降级**——只做宿主 I/O 载体/参考数据/文件 codec，L2+ 计算路径禁用 | 维护者选定 |
| D5 | CPU/GPU 关系：**契约统一、实现自由**；一期 CPU 存储内部沿用 `MatrixT<P>` 但彻底私有化（引擎内 72 处内核调用零迁移），CPU 真池化是纯优化另期 | 本轮裁定（答复维护者反问"是否有必要 CPU/GPU 完全相同"） |
| D6 | 初始化：**声明式 InitSpec**；设备未实现原生生成时引擎内部走 host 生成+上传（隐式，调用方不可见），实现了走原生 | 维护者口述 |
| D7 | 库外收口：**同期全迁**——90 处测试直构 + tools 随存储私有化一起迁，不留过渡期 | 维护者选定 |
| D8 | 节奏：**分期推进**，沿用分层验收基线（§6），每期独立可回滚 | 维护者选定 |
| D9 | 宿主动词：**保留 `from_matrix/to_matrix` 名称**（Matrix 留用，名字诚实），新增 `read/write` 分组；15 P2 的"改名 to_host/from_host"**取消**，防线从"命名即防线"改为"分组 + grep 审计即防线" | 本轮裁定 |
| D10 | 修改类操作（含 `reshape`）一律引擎方法（由 D2 推导：修改 = 访问存储） | 本轮裁定 |

## 4. 目标架构

### 4.1 访问不变量（拟立为铁律新增条）

> **一切 Tensor 的创建、访问、修改必须经由 ComputeEngine。**
> Tensor 存储是私有实现细节；静态直构工厂与公开存储访问器从公共 API 消失。
> 库内豁免仅两条：`ComputeEngine::adopt`（已有，出生绑定通道）与引擎域内 friend
> （DSL 求值器、引擎内核）。库外（测试/tools/CLI）一律 `from_matrix/read/to_matrix`。

关键性质：**编译期强制**。访问器删除/私有化后，绕过 = 编译错误，比 `NN_BIND_DEBUG`
运行期检查更硬；P1 的出生绑定检查（`bind_check_`）继续作为运行期第二道网。

### 4.2 Tensor

- 保持**值语义**类（`Result<Tensor>` 203 处零改动），公开面收缩为：
  形状/精度/设备查询（`rows/cols/precision/device`）、`bound()/engine()`、`TensorRef`。
- 存储（`std::variant<shared_ptr<MatrixT<F16>>, shared_ptr<MatrixT<F32>>>` + GPU 槽位）**私有**；
  `cpu_matrix/cpu_shared/gpu_tensor/gpu_shared` 改为私有 + friend，或收敛为引擎内部 accessor。
- `Tensor::from_matrix/from_gpu/cpu/cpu_uninitialized` 等静态工厂从公共 API 移除，
  收编为引擎工厂的实现细节（`adopt` 通道保留）。
- `reshape` 移入引擎（D10）。GPU 零拷贝视图 / CPU 复制的语义不变（§10 高频坑 #6 照旧生效）。
- 15 P5/P6 的 `TensorStorage` 多态**降级为可选后置项（M7）**：存储已私有后，多态只剩
  "新增后端不改 `compute_tensor.hpp`"的收益；第三后端出现前不立项。

### 4.3 ComputeEngine 契约扩展（API 提案，名字可再定）

| 能力 | 提议 API | 说明 |
|---|---|---|
| 创建（含初始化） | `create_tensor(rows, cols, P, InitSpec)` | 扩展现有 `create_tensor`（`compute_engine.hpp:215`）；默认语义对齐现状零填充，`Uninitialized` 契约沿用 `cpu_uninitialized` 注释 |
| 上传（I/O 分组） | `from_matrix(const Matrix&, P) -> Result<Tensor>` | 保留现名（D9）；数据集/预训练模型入口 |
| 下载（I/O 分组） | `to_matrix(const Tensor&, P) -> Result<Matrix>` | 保留现名（D9）；对拍/落盘出口 |
| 批量读 | `read(const Tensor&, std::span<T>) -> Result<void>` | 元素类型与 `precision()` 匹配；GPU 隐含 flush + 同步 |
| 批量写 | `write(const Tensor&, std::span<const T>) -> Result<void>` | 覆盖既有存储、**不替换对象**（沿 `copy_into` 红线） |
| 索引读/写 | `get_index / set_index` | 语法糖，允许同步；GPU 上不承诺热循环性能 |
| 变形 | `reshape(const Tensor&, r, c) -> Result<Tensor>` | 语义同现状 `Tensor::reshape` |
| 跨设备/引擎 | `import(const Tensor&, P)` | 吸收 15 P3；43 处 `ensure_gpu` → `import` |
| 内存池 | `pool_stats()` / `release_idle_pool_blocks()` | 语义补齐见 §4.5 |

- 既有入口不动：`cast/cast_into/copy_into/begin_batch/end_batch/submit_scalar_readback/
  create_offload_buffer/...`（49 个 virtual 骨架不变，新增均为 NVI 扩展）。
- `read` 在 GPU 上隐含 flush + 同步（与现 `to_matrix` 同）；**逐 step loss 回读继续走
  `submit_scalar_readback` 快路**，不被 `read` 取代。
- `from_matrix/to_matrix` 只准出现在 I/O/测试代码；库内计算路径出现即审计违规（§4.6）。

### 4.4 声明式初始化 InitSpec

```cpp
// 提案（名字可再定）：层算分布参数，引擎负责填数
struct InitSpec {
    enum class Kind { Uninitialized, Zero, Constant, Uniform, Normal } kind;
    Scalar value = 0;      // Constant
    Scalar lo = 0, hi = 0; // Uniform（xavier/kaiming = 层算好 limit 后用 Uniform）
    Scalar mean = 0, stddev = 0; // Normal
    std::uint64_t seed = 42;     // 显式种子（U1：是否必填待裁）
};
// engine.create_tensor(rows, cols, p, InitSpec::uniform(-limit, limit, seed));
```

- **层负责算分布参数**：`limit = sqrt(6/(fan_in+fan_out))` 这类层语义留在 `Layer`，
  不进引擎（铁律 #3 分层职责）；引擎只管"把 shape×precision×分布的数填出来"。
- **策略分派**（D6）：引擎默认 host RNG 生成后上传（两设备同序列、确定性达标）；
  设备端 RNG 是引擎内部优化，调用方无感——落地时字节序列变化属**计划内基线重立**（§6）。
- **前置修复**：`mlp/conv/zipt` 的 `std::random_device{}` 播种跨进程不确定（违反铁律 #8 精神），
  M2 第一步先统一为显式默认 seed（行为变化单列验收），此后初值跨进程确定；
  各层散落的 `thread_local rng_` 随之消失，RNG 收归引擎。

### 4.5 内存池统一（契约层，D5）

- 统一的是**接口与语义**，不是实现：
  - `pool_stats()`：GPU 返回现有池统计；CPU 返回"直配模式"统计（活分配数/字节）——
    **从空串变为有意义**，GUI/训练日志一套代码读两个引擎。
  - `release_idle_pool_blocks()`：GPU 归还空闲整块（现状）；CPU 为语义成立的 no-op
    （直配无整块可归还），文档写明。
- 池归属现状保持：GPU 双池（`memory_pool_`/`transient_pool_`）留在 `GpuBackend`，
  **引擎是唯一对外窗口**；上层永远不直接摸 `MemoryPool`。
- CPU 是否建真池：**独立优化立项**（先用分配剖析量化 DSL 临时张量热点，再决定），
  不阻塞任何上层统一。

### 4.6 Matrix 降级（D4）

- **保留**：宿主 I/O 载体 + 测试对拍基准 + 文件 codec（`model_serialization.hpp` 的
  read/write_matrix、`cli_mnist_io.hpp`、`domain_mnist.hpp` 照旧）。
- **禁止**：出现在 L2+ 计算路径——Layer/Loss/Optimizer/Model 不得持有或交换 `Matrix`。
- `compute_cpu_engine.hpp` 内核继续吃 `MatrixT<P>`（引擎**内部** = 存储实现细节，合规，
  这正是 D5"一期零内核迁移"的依据）。
- 审计：分层 grep 脚本（落点见 U5），纳入既有 `bench/doc_inventory.ps1` 文档对齐审计体系。

### 4.7 DSL 与热点路径（零开销豁免）

- `expr_dsl.hpp` 是逐元素热点：`cpu_get_ptr` 若拷 shared_ptr，实测慢 8-70 倍
  （`compute_tensor.hpp:98-102` 注释在案）——**内部通道必须保持裸指针零开销**，
  不得为"统一"引入每元素虚调用/原子操作。
- 定位：DSL 求值器 = 引擎域内组件（`dsl::compute` 本就带 `ComputeEngine&`），
  经 friend/内部通道拿 span；既有结构已是"求值前物化一次、热循环裸指针"——
  matmul 叶子的 `prepare_cpu`（`expr_dsl.hpp:548-607`）用引擎物化 `c_cache_`
  后绑裸指针 `c_data_`（:605，幂等 :550），保持该结构。
- 验收硬项：`layer_bench` 无回退（沿 15 P5 验收要求）。

## 5. 分期实施

总原则：编译器报错驱动 + 每期独立验收 + 单期可回滚。四件套 = build 全绿 + ctest 20/20 +
CPU 字节基线 + scan 产物 hash + layer_bench（定义见 §6）。

| 期 | 内容 | 关键改动面 | 专属验收 |
|---|---|---|---|
| **M1** | **访问收口**：存储私有化、静态工厂收编、`reshape` 引擎化、库外同期全迁（D7） | `compute_tensor.hpp`；引擎/DSL 内部通道（include 5 文件）；src 127 处 + `text_train.cpp:756` | 四件套 + **编译期强制**（旧 API 已不存在，`grep cpu_matrix` 仅剩引擎内部） |
| **M2** | **声明式创建/初始化**：`InitSpec` + `Layer::init` 迁移 + RNG 收编 | `create_tensor` NVI 扩展；`compute_layer_*.hpp` 的 init（mlp/conv/gpt/zipt/rapt…） | 四件套 + **初值跨进程确定性**（两次运行初值逐字节同；`random_device` 层改造属计划内字节变化，先立新锚） |
| **M3** | **批量读写 API**：`read/write/get_index/set_index` + I/O 分组审计口径（D9） | `compute_engine.hpp` + CPU/GPU 两引擎；测试读写路径 | 四件套 + GPU staging 批量语义探针（写后读回对拍） |
| **M4** | **Matrix 降级收口**：L2+ 禁用规则成文 + 分层审计脚本 | 规则进 AGENTS.md 铁律；审计脚本（U5） | 审计脚本零违规 |
| **M5** | **内存池契约统一**：CPU `pool_stats` 语义补齐 | `compute_engine.hpp` 默认实现 + `CpuEngine` | 两引擎 `pool_stats` 非空且口径文档化 |
| **M6** | **吸收 15 P2/P3**：删每调用 engine 形参；43 处 `ensure_gpu`→`import`；跨引擎硬错误默认开 | 175 处 `ComputeEngine&` 打标清单（16 §3）；`NN_BIND_DEBUG` 门禁化 | 四件套 + `NN_BIND_DEBUG=1` ctest 门禁；顺带裁 15 D6/D7/D8 |
| **M7** | （可选后置）15 P5/P6 存储多态 + `as_cpu/as_gpu` | `compute_tensor.hpp` | 第三后端出现前**不立项** |

**顺序理由**：M1 先行——编译器立刻封锁一切绕路，后续各期都在不变量内做；M2 在 M3 前，
初始化是创建的一部分；M6 靠后——形参删除不依赖存储私有化，但受益于它（engine 绑定已收口）。
15 的 P4（宿主格式契约三处，15 §4.6）保持**正交另案**，不并入本表。

## 6. 验收基线（沿 15 §3.5 及其 D9"字节基线分层"裁定——注意区别于本文 §3 的 D9 宿主动词裁定）

- build 全绿 + ctest 20/20（GPU 用例退出码 77 = skip 如实记录；GPU 档基线口径见 16 §5）。
- **CPU 同设备 pre/post**：loss 序列 + 权重**逐字节**。
- **GPU 同设备 pre/post**：run-to-run 稳定 → 逐字节；不稳定 → 容差（GPU 档位实测仍缺——
  `gpu_stability_probe --gpu` 崩溃未修，16 §7-2，Windows 本机同样复现）。
- **scan 产物 hash 恒定**：`expr_specs.bin` = `bdc3a442…a58360`；`fused_registry.hpp`
  本机 pre/post 锚点见 15 §7.1。
- **layer_bench**：同窗交错 + 配对 A/B，无系统性回退（噪声 ±6% 口径）。
- **计划内字节变化例外**：M2（RNG 收编）与将来设备端 RNG 落地——先立新锚，
  新锚两次独立运行之间仍须逐字节；变化条目在该期验收中显式记录。

## 7. 风险与陷阱

1. **逐元素 API 性能陷阱**：GPU 上 `get_index/set_index` 若直译 = 每元素一次 staging 往返
   （连 `submit_scalar_readback` 都要专门排 D2H 队列），完全不可用。设计与注释双重标注
   "批量 read/write 是本体"，index API 不给热循环承诺。
2. **热点路径回退**：内部通道若拷 `shared_ptr`/加虚调用 → 8-70× 教训重演（§4.7）。
   M1 验收必含 layer_bench。
3. **计划内字节变化**：RNG 收编（M2）、设备端 RNG（未来）→ 按 §6 例外流程重立锚点，
   防止把计划内变化误判成回归。
4. **录制期生命周期**（铁律 #6）：`read/write` 可能进入 batch 录制窗口，M3 必须明确
   "read 隐含 flush/同步"与"写入张量须存活到 `end_batch` 之后"的调用约定。
5. **测试迁移量**：src 127 处 `cpu_matrix` 集中在 `expr_*_test`（填数 + 对拍），
   迁移模板化——填数：构造 Matrix → `from_matrix`；对拍 → `to_matrix`；D7 裁定同期完成，
   一次编译红海换一次性收口。
6. **既有故障不混入**：f16 loss 冻结（14）、`text_train --model zipt` abort、
   `gpu_stability_probe --gpu` 崩溃均非本范围——验收时按既有状态如实记录，不得当回归。

## 8. 与既有文档关系 / 未决点

**关系**

- `15-computeengine-refresh.md`：P-1/P1 成果保留；**P2→M6、P3→M6、P5/P6→M7**，
  P4 正交另案；15 本文头部已标注被本文吸收。
- `13-refactor-backlog.md` §10.8：句柄指针化**维持否决**，增补注脚"访问诉求由本文
  §4.1 不变量承接"。
- `16-computeengine-p0-inventory.md`：M0 证据沿用（175 打标 / 43 ensure_gpu / 381 宿主中转）。
- `AGENTS.md`：M1 起新增铁律条（访问必须过引擎）、M4 起新增 Matrix 分层禁用条；
  文档索引已同步。

**未决点（实施时裁定）**

| # | 问题 | 倾向 |
|---|---|---|
| U1 | `InitSpec::seed` 是否必填 | 必填，层传显式默认值 |
| U2 | `read/write` 元素类型形态：`span<T>` 精度精确匹配，还是按 `precision()` 重载 | 精确匹配 + 静态断言（防 f16/f32 槽错位） |
| U3 | CPU 真内存池立项与否 | 先分配剖析量化，另案 |
| U4 | 15 D6/D7/D8（宿主格式契约定序 / `Model::set_engine` 死码 / dummy 张量） | 随 M6 一并裁定 |
| U5 | 分层审计脚本落点：扩展 `bench/doc_inventory.ps1` vs 新建 audit | 倾向扩展 doc_inventory（已有文档↔代码对齐审计体系） |
