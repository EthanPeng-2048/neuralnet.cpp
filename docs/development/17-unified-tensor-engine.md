# 统一 Tensor / ComputeEngine / MemoryPool 底层架构总纲（2026-09-30）

> **本文是新总纲**：吸收并取代 `15-computeengine-refresh.md` 未实施的 P2-P6（本文 §5 分期），
> 并对 `13-refactor-backlog.md` §10.8 增补"访问不变量"注脚。历史归档见 `docs/history.md`。
> 状态：**设计裁定已完成（§3 共 11 项）；M1 访问收口已实施（2026-09-30，验收与
> 过程见 §5 M1 行 + `docs/history.md`）；M2 声明式初始化已实施（2026-09-30，
> 见 §5 M2 行）；M3 批量读写 API 已实施（2026-09-30，见 §5 M3 行）；M4 Matrix
> 降级收口已实施（2026-09-30，见 §3 D11 + §5 M4 行）；M5 内存池契约统一已实施
> （2026-09-30，见 §4.5 + §5 M5 行）；M6 分三段：段 A（15 D6/D7/D8 落地 +
> `NN_BIND_DEBUG` ctest 门禁）与段 B（43 处 `ensure_gpu`→`import`）已实施
> （2026-09-30，见 §5 M6 行与 §8 交接），**段 C（删每调用 engine 形参）未实施；
> M7 未实施**。
> 实施立项时读本文 + §5 分期 + 源码现状。
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

**现状事实**（可复现，2026-09-30 核对；**M1/M2 实施前口径**——M1 落地后库外直构/直读已归零、
M2 落地后 init RNG 已收编，下表保留为动机证据，复现命令仍可跑但库外命中应为 0、init 站点应无 `random_device`）：

| 事实 | 数字 | 复现 |
|---|---|---|
| `Matrix` 出现（include 213 + src 373 次命中） | 61 文件 / 586 处 | `Select-String` 扫 `include/**/*.hpp` + `src/**/*.cpp`，pattern `\bMatrix\b`（命中行；src 若含 `.hpp` 另 +3 行/1 文件） |
| `cpu_matrix()/cpu_shared()` 调用 | include 5 文件（CPU/GPU 引擎 + 引擎基类 + DSL + 容器头）+ src 127 处（126 测试 + `text_train.cpp:756`） | `grep -rn "cpu_matrix()"` |
| `gpu_tensor()` 调用 | **全部**在 `compute_gpu_engine.hpp` 内部 59 处（引擎自己，合规）；`compute_tensor.hpp:276/294` 仅为访问器声明 | 同上换 pattern |
| `Tensor` 静态直构工厂（`Tensor::from_matrix/cpu/...`） | 98 处 / 8 文件（2026-09-30 复核；库外为主，15 §1 口径同源） | `Select-String` 两个 pattern（`Tensor::from_matrix`、`Tensor::cpu\b`）扫 `src/*_test.cpp`，命中行和 = 98 |
| 内存池 | 仅 Vulkan：`backend/compute_memory_pool.hpp`（`GpuBackend` 持双池 memory/transient）；`backend/compute_staging_ring.hpp:56/93/104/232` 与 `compute_vk_backend.hpp` 亦引用，均 Vulkan 后端 | `grep -rn MemoryPool include/` |
| CPU 池 | **不存在**：`Matrix` 直接 `std::vector`；`pool_stats()` CPU 返回空串（`compute_engine.hpp:144`） | 读引擎默认实现 |（**M5 已补齐 2026-09-30**：CPU 现返回宿主直配账本 `direct{…}`，本行保留为动机证据）
| init RNG | 混乱：固定 seed 42 = `gpt:432/603`、`rapt:1155`、`zipt:814`；`std::random_device{}`（**跨进程不确定**）= `gpt:953`、`mlp:40`、`conv:107`、`zipt:89/1044`、`rapt:1462`（**M2 已收编**：init 站点全迁 InitSpec + 显式 seed；余下 `gpt:953`/`zipt:1044`/`rapt:1462` 为 `generate()` **采样** RNG——运行期采样随机性非 init，M2 裁定不动） | `grep -rn mt19937 include/` |
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

## 3. 核定裁定（11 项）

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
| D11 | **L2+ 宿主桥口径（M4 实施裁定）**：禁令本体 = **`Matrix` 类型**（铁律 #12，编译期可审计的分层线）；`from_matrix/to_matrix/copy_from` 因签名即含 `Matrix`，随之一并在 L2 禁用。层自算的辅助数据（索引/位置/掩码/斜率/编码表——**没有引擎侧生成原语**的那一类）经 span 宿主桥 `detail::upload_span/download_span/download_vector`（`compute_engine.hpp` 尾部，内部走 M3 `write/read`，f16 转换按 RHE 与 `from_matrix` 同口径）或 `create_tensor`+`InitSpec`/`zero`；**数据集/预训练权重/对拍/落盘仍走 I/O 层的 `from_matrix/to_matrix`**。由此 §4.3 末条"I/O 分组动词只准出现在 I/O/测试代码"收窄为"**Matrix 型三动词**只准出现在 I/O 层"，span 级桥是 L2 的受控例外（判据：不引入 Matrix 即不破坏数据格式统一）；新增引擎虚表项 0 | M4 实施时裁定（原 D3"库外豁免"与 §4.3 字面在 L2 上不可兼得：层无 arange/RoPE 表生成原语，严格执行需先加原语，超 M4 范围） |

## 4. 目标架构

### 4.1 访问不变量（已立为铁律——AGENTS.md §5 第 11 条，M1 落地 2026-09-30）

> **一切 Tensor 的创建、访问、修改必须经由 ComputeEngine。**
> Tensor 存储是私有实现细节；静态直构工厂与公开存储访问器从公共 API 消失。
> 库内豁免仅两条：`ComputeEngine::adopt`（已有，出生绑定通道）与引擎域内 friend
> （DSL 求值器、引擎内核）。库外（测试/tools/CLI）一律 `from_matrix/read/write/to_matrix`。

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
| 创建（含初始化） | `create_tensor(rows, cols, P, InitSpec)` | 扩展现有 `create_tensor`（`compute_engine.hpp:215`）；**M2 实施口径**：3 参重载保持既有纯分配语义（CPU 分配零填充 / GPU 分配未初始化，需零显式 `zero`），初值一律显式走 `InitSpec`（Zero/Constant/…）；`Uninitialized` 契约沿用 `cpu_uninitialized` 注释 |
| 上传（I/O 分组） | `from_matrix(const Matrix&, P) -> Result<Tensor>` | 保留现名（D9）；数据集/预训练模型入口 |
| 下载（I/O 分组） | `to_matrix(const Tensor&, P) -> Result<Matrix>` | 保留现名（D9）；对拍/落盘出口 |
| 批量读 | `read(const Tensor&, std::span<T>) -> Result<void>` | **M3 实施**：元素类型与 `precision()` **精确匹配**（U2——float↔F32、f16↔F16，错配运行期错误、类型非法编译期 static_assert，防 f16/f32 槽错位）；GPU 隐含 flush + 同步（走 `to_matrix` 同路） |
| 批量写 | `write(Tensor&, std::span<T>) -> Result<void>` | **M3 实施**（形参 `Tensor&`，与 `zero/copy_from` 的存储变更约定同，非提案中的 const）：覆盖既有存储、**不替换对象**（宿主直写既有存储；GPU 经 `copy_from` 的既有 drain 语义，f16 目标 = f32 上传 + `cast_into` 写回原存储）；span 形参为 `span<T>` 以同时承接 const/非 const 实参（模板推导中 `const T` 无法匹配非 const span），断言按去 cv 后元素类型校验 |
| 索引读/写 | `get_index / set_index` | **M3 实施**：宿主直读写、GPU = 批量 read/write 的语法糖（一整轮 staging 往返）；允许同步，不承诺热循环性能 |
| 变形 | `reshape(const Tensor&, r, c) -> Result<Tensor>` | 语义同现状 `Tensor::reshape` |
| 跨设备/引擎 | `import(const Tensor&, P)` | **M6 实施**（吸收 15 P3）：公共 NVI 入口（`bind_check_` + `stamp_`）+ 新虚 `import_impl`——**virtual 方法 48 → 49（`doc_inventory` 口径；`grep -c "\bvirtual\b"` 原始 51 处）**。语义按 15 §3.2：同设备同精度 = 零拷贝别名、同设备异精度 = 引擎内 cast、跨设备 = 经宿主中转（基类默认实现内含同设备快路）；`GpuEngine` override = 原 `ensure_gpu` 的宿主直传快路径（省一次宿主拷贝）。返回**新句柄**、不改写 `src`（就地重绑定须在调用点显式写）。16 §2 的 43 处 `ensure_gpu` 调用点已全部改名 `import`（命名即防线），并顺带修掉 `ensure_gpu` 硬取 `cpu_matrix()`（F32 槽）对 f16 源取空指针的隐患 |
| 内存池 | `pool_stats()` / `release_idle_pool_blocks()` | **M5 实施**：语义补齐见 §4.5（无池引擎返回直配账本） |

- 既有入口不动：`cast/cast_into/copy_into/begin_batch/end_batch/submit_scalar_readback/
  create_offload_buffer/...`（M1-M5 零虚表变化；**M6 新增 `import_impl` → virtual 方法
  48 → 49（`doc_inventory` 口径），其余新增均为非虚 NVI 扩展**）。
- `read` 在 GPU 上隐含 flush + 同步（与现 `to_matrix` 同）；**逐 step loss 回读继续走
  `submit_scalar_readback` 快路**，不被 `read` 取代。
- I/O 分组动词分两档（**D11 收窄，M4 实施**）：**Matrix 型三动词**
  （`from_matrix/to_matrix/copy_from`）只准出现在 I/O/测试/序列化代码，L2+ 出现即
  审计违规（§4.6、铁律 #12）；**span/标量级**（`read/write/get_index/set_index` 与
  `detail::upload_span/download_span/download_vector`）是 L2 的受控宿主桥，仅限
  层自算的小规模辅助数据，大批量数据仍属 I/O 层。
  `import` 是**引擎内/跨设备**拉取入口（M6）：库内只在引擎实现里出现（如 `GpuEngine`
  的宿主直传），L2 计算路径不调用它——跨引擎互操作在 P1 起就是硬错误。

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
  （**M2 已实施**：`kInitSeed=42` + 引擎内按创建序号混流——同 seed 不同张量不撞流，
  同形状多层不互为镜像；层手填 `Matrix` + `from_matrix` 的 init 路径一并清空。
  `generate()` 采样 RNG 与 `text_train` 数据洗牌 RNG 属运行期随机性，**裁定不迁**。）

### 4.5 内存池统一（契约层，D5）——**M5 已落地 2026-09-30**

- 统一的是**接口与语义**，不是实现：
  - `pool_stats()`：GPU 返回现有池统计（`persist{…} transient{…} pending=…`）；**无池引擎
    （CPU）在基类默认实现里返回宿主直配账本**——
    `direct{ blocks=<活分配数> live_bytes=<> peak_bytes=<> total_blocks=<> total_bytes=<> }`，
    **从空串变为有意义**，GUI/训练日志一套代码读两个引擎。
    账本 `nn::host_alloc_ledger()` 定义在 `algebra_matrix.hpp`：`MatrixT` 的
    `allocate_`/`release_storage_`（析构、移动赋值、`resize` 三处释放出口）各记一次，
    relaxed 原子、只在分配路径，元素热循环零参与。
  - `release_idle_pool_blocks()`：GPU 归还空闲整块（现状）；CPU 为**语义成立的 no-op**
    （直配无整块可归还），已写进 `compute_engine.hpp` 注释与 `04-memory-optimization.md`。
- 池归属现状保持：GPU 双池（`memory_pool_`/`transient_pool_`）留在 `GpuBackend`，
  **引擎是唯一对外窗口**；上层永远不直接摸 `MemoryPool`。
- CPU 是否建真池：**独立优化立项**（先用分配剖析量化 DSL 临时张量热点，再决定），
  不阻塞任何上层统一（U3 未决，维持）。

### 4.6 Matrix 降级（D4）

- **保留**：宿主 I/O 载体 + 测试对拍基准 + 文件 codec（`model_serialization.hpp` 的
  read/write_matrix、`cli_mnist_io.hpp`、`domain_mnist.hpp` 照旧）。
- **禁止**：出现在 L2+ 计算路径——Layer/Loss/Optimizer/Model 不得持有或交换 `Matrix`
  （**已立为铁律 #12，M4 落地 2026-09-30**）；层自算辅助数据经 D11 的 span 宿主桥。
- `compute_cpu_engine.hpp` 内核继续吃 `MatrixT<P>`（引擎**内部** = 存储实现细节，合规，
  这正是 D5"一期零内核迁移"的依据）。
- 审计：**已落地为 `bench/doc_inventory.ps1` 第 [4] 节**（U5 裁定：扩展 doc_inventory，
  不另建脚本——它本就枚举 L2 文件集 `compute_layer*.hpp` + loss + optimizer +
  model_container）。逐行剥离 `//` 与块注释后匹配三类：`Matrix`/`MatrixT` 类型（硬）、
  `from_matrix/to_matrix/copy_from`（硬）、宿主桥用量（披露）。**验收口径：
  `L2-VIOLATIONS: 0`**。

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
| **M1** ✅（2026-09-30） | **访问收口**：存储私有化、静态工厂收编、`reshape` 引擎化（`ComputeEngine::reshape`，D10）、库外同期全迁（D7） | `compute_tensor.hpp`；`detail::TensorAccess`（DSL 19 处改道）；`model_serialization.hpp` f16 分支；src 9 测试 + `text_train.cpp:756` + `tools/scan_exprs.cpp`（commits `3d76478`/`c8f308f`） | **四件套全过**：build 122/122；ctest 20/20；CPU 探针 `6f8849f14da23110` 与 pre 逐位一致；scan 双 hash 不变；layer_bench 配对 A/B（pre=HEAD~2 worktree 二进制，feedforward/swiglu/linear 各 4 轮交错，分布重叠无回退）；**编译期强制**：`grep \.cpu_matrix` 库外零命中（详见 `docs/history.md` M1 条） |
| **M2** ✅（2026-09-30） | **声明式创建/初始化**：`InitSpec` + `Layer::init` 迁移 + RNG 收编 | `create_tensor` NVI 扩展；`compute_layer_*.hpp` 的 init（mlp/conv/gpt/zipt/rapt…）（commits `8bd6bd5` 核心、`4492098` 层迁移） | **四件套全过** + **初值跨进程确定性**：build 122/122 零告警；ctest 20/20；`gpu_stability_probe --init-hash` 六模型两进程 `INIT1` 行全同（**新锚**：mnist_mlp `a22e807ee05ec3ac`、cnn `4020958a14160bbd`、mnist_transformer `04a72d865624042a`、gpt `8efa936ac5c8c9b2`、zipt `f90bf8c8783c2f89`、rapt `b037632f75b7159c`）；mnist 三架构（`--shuffle-steps false`）双进程 loss 序列 + 模型文件逐字节一致（**新训练锚**，仅墙钟行异）；`--steps 20` hash=`6f8849f14da23110` 与 M1 锚点逐位一致（覆写隔离下后端执行零变化）；scan 双 hash 不变（`expr_specs.bin`=bdc3a442…a58360、`fused_registry.hpp`=7a10412c…）；layer_bench 配对 4 轮 + feedforward 逆序 4 轮复测均 < ±6% 无系统性回退；`NN_BIND_DEBUG=1` 冒烟零诊断（详见 `docs/history.md` M2 条） |
| **M3** ✅（2026-09-30） | **批量读写 API**：`read/write/get_index/set_index` + I/O 分组审计口径（D9；U2 精确匹配） | `compute_engine.hpp` I/O 分组（基类非虚模板，复用 `to_matrix`/`copy_from`）+ CPU/GPU 两引擎；测试 f32 填充/恢复路径迁 `write`；探针 `--io-roundtrip`（commits `e0338d9` 核心、`4977a29` 探针、`3ddb13c` 测试） | **四件套全过**：build 全绿零告警；ctest 20/20；`gpu_stability_probe --io-roundtrip` CPU + GPU 全过（`f32_roundtrip`/`f16_roundtrip`/`batch_window` 三用例含录制窗口 write→read→end_batch→复读）；CPU 字节锚 `--steps 20` hash=`6f8849f14da23110` 不变；scan 双 hash 不变（`expr_specs.bin`=bdc3a442…a58360、`fused_registry.hpp`=7a10412c…）；**顺带修复 16 §7-2**（探针漏 `backend.initialize()` 的 GPU 崩溃，根因回填 16 §6/§7）——dev0/dev2/dev4 进程内两轮 + 跨进程 loss/hash 逐字节（详见 `docs/history.md` M3 条） |
| **M4** ✅（2026-09-30） | **Matrix 降级收口**：L2+ 禁用规则成文（铁律 #12）+ D11 宿主桥裁定 + 分层审计脚本（U5） | L2 全量迁移：`compute_layer_{attention,base,conv,gpt,rapt,transformer,zipt}.hpp` + `compute_loss.hpp` + `compute_optimizer.hpp`（45 处 Matrix 型 I/O + 32 处 `Matrix` 类型）；`compute_engine.hpp` 尾部新增 `detail::upload_span/download_span/download_vector`（宿主桥，0 虚表项）；`bench/doc_inventory.ps1` 新增第 [4] 节 | **审计零违规**（`L2-VIOLATIONS: 0`，宿主桥 41 处仅披露）+ 四件套全过：build 122/122 零告警；ctest 20/20；**CPU 字节锚 `6f8849f14da23110` 与 GPU dev2 `8ef51b2927253c50` 均与迁移前逐位一致**（迁移是纯载体替换，数值路径零改动）；scan 双 hash 不变；layer_bench 配对 A/B（pre=HEAD worktree 二进制，feedforward/swiglu/linear 各 4 轮交错）：feedforward fwd/train **-1.2%/-1.6%**、linear +1.4%/-4.2%、swiglu fwd -6.2%（0.4ms 亚毫秒级，PRE 自身极差 0.38–0.47ms 即 ±10%）——全部在 ±6% 噪声带内无系统性回退（详见 `docs/history.md` M4 条） |
| **M5** ✅（2026-09-30） | **内存池契约统一**：无池引擎 `pool_stats` 语义补齐 | `algebra_matrix.hpp`（`HostAllocLedger` + `MatrixT` 分配/释放记账三出口）+ `compute_engine.hpp` 基类默认 `pool_stats` | **两引擎 `pool_stats` 非空且口径文档化**（17 §4.5 + `04-memory-optimization.md` + `01-compute-engine-development.md`）：CPU `direct{ blocks/live_bytes/peak_bytes/total_blocks/total_bytes }` 实测随 model-built/optimizer-created 递增（`NN_MEM_STATS=1 text_train` 抓取），GPU `persist{…} transient{…} pending=…` 照旧；`release_idle_pool_blocks` CPU no-op 写明理由。四件套：build 122/122 零告警、ctest 20/20、CPU 锚 `6f8849f14da23110` 与 scan 双 hash 不变、`--io-roundtrip` CPU/GPU 全过、L2 审计仍 0 |
| **M6**（段 A/B ✅ 2026-09-30；**段 C 未实施**） | **吸收 15 P2/P3** 分三段：**A** = 15 D6/D7/D8 落地 + `NN_BIND_DEBUG` 门禁化；**B** = 43 处 `ensure_gpu`→`import`（新增跨设备拉取契约，虚表 49→50）；**C** = 删每调用 engine 形参（175 打标清单里的 110 DROP + 7 REVIEW） | 段 A：`model_serialization.hpp`（D6-1 白做上传）、`compute_engine.hpp`（D6-3 改纯虚、`to_prec` 补绑定、`bind_error_` 打 stderr）、`model_container.hpp`（D7）、offload 占位（D8）、`expr_dsl.hpp`（归约向量 adopt）；段 B：`compute_engine.hpp` `import/import_impl` + `compute_gpu_engine.hpp` 48 处改名（commits `8ecf1bd` 段 A） | **段 A+B 四件套全过**：build 零告警；ctest 20/20 **且 `NN_BIND_DEBUG=1` 同样 20/20（门禁成立）**；CPU 锚 `6f8849f14da23110` 与 **GPU dev2 `8ef51b2927253c50`** 均与改前逐位一致；scan 双 hash 不变；`--io-roundtrip` CPU/GPU 全过；L2 审计 `L2-VIOLATIONS: 0`。另证 D6-1 正确性：save→load→save **SHA256 逐字节一致**（f32 与 f16 param 两种模型）。段 C 验收待其实施时补（见 §8 交接） |
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
- **计划内字节变化例外**：M2（RNG 收编，**已按此立新锚 2026-09-30——见 §5 M2 行**）与将来设备端 RNG 落地——先立新锚，
  新锚两次独立运行之间仍须逐字节；变化条目在该期验收中显式记录。

## 7. 风险与陷阱

1. **逐元素 API 性能陷阱**：GPU 上 `get_index/set_index` 若直译 = 每元素一次 staging 往返
   （连 `submit_scalar_readback` 都要专门排 D2H 队列），完全不可用。设计与注释双重标注
   "批量 read/write 是本体"，index API 不给热循环承诺。
   （**M3 已落地**：`compute_engine.hpp` I/O 分组注释双重标注；宿主张量 index 直读写零往返，
   GPU 走批量 read/write——每次调用一整轮 staging。）
2. **热点路径回退**：内部通道若拷 `shared_ptr`/加虚调用 → 8-70× 教训重演（§4.7）。
   M1 验收必含 layer_bench。
3. **计划内字节变化**：RNG 收编（M2）、设备端 RNG（未来）→ 按 §6 例外流程重立锚点，
   防止把计划内变化误判成回归。
4. **录制期生命周期**（铁律 #6）：`read/write` 可能进入 batch 录制窗口，M3 必须明确
   "read 隐含 flush/同步"与"写入张量须存活到 `end_batch` 之后"的调用约定。
   （**M3 已成文**：`compute_engine.hpp` I/O 分组注释——read 走 `to_matrix` 同路隐含
   end_batch → wait_in_flight → 新帧；write 走 `copy_from` 既有 drain 语义。验收探针
   `--io-roundtrip` 含 `batch_window` 用例：录制窗口内 write → read → end_batch → 复读值不变。）
5. **测试迁移量**：src 127 处 `cpu_matrix` 集中在 `expr_*_test`（填数 + 对拍），
   迁移模板化——填数：构造 Matrix → `from_matrix`；对拍 → `to_matrix`；D7 裁定同期完成，
   一次编译红海换一次性收口。
6. **既有故障不混入**：f16 loss 冻结（14）、`text_train --model zipt` abort、
   `gpu_stability_probe --gpu` 崩溃均非本范围——验收时按既有状态如实记录，不得当回归。

## 8. 与既有文档关系 / 未决点

**关系**

- `15-computeengine-refresh.md`：P-1/P1 成果保留；**P2→M6 段 C（未实施）、P3→M6 段 B（2026-09-30 落地）、P5/P6→M7**，
  P4 正交另案；15 本文头部已标注被本文吸收。
- `13-refactor-backlog.md` §10.8：句柄指针化**维持否决**，增补注脚"访问诉求由本文
  §4.1 不变量承接"。
- `16-computeengine-p0-inventory.md`：M0 证据沿用（175 打标 / 43 ensure_gpu / 381 宿主中转）
  ——**43 处 ensure_gpu 已于 M6 段 B 全部改名 `import`（本表 §2 需按此读作历史口径）**；
  381 宿主中转中的 L2 部分已随 M4 归零（见 §5 M4 行）。
- `AGENTS.md`：**M1 已新增铁律第 11 条**（张量存储不可绕过引擎，2026-09-30）；
  **M4 已新增铁律第 12 条**（L2+ 计算路径禁用 Matrix，含宿主桥与审计口径，2026-09-30）；
  文档索引已同步。

**M6 段 C 交接（删每调用 engine 形参——本轮未实施，下一轮立项读此处 + 16 §3 + 15 §3.3）**

1. **范围**：16 §3 的 DROP 110 处 + REVIEW 7 处（KEEP 58 处按定义不动：`init` 绑定入口、
   Optimizer 构造、DSL 入口、Model 容器、domain 工厂）；`include` 内 `forward(engine,…)`
   80 处 / `backward(engine,…)` 57 处，`src` 调用点约 155 处（编译器穷尽驱动）。
2. **绑定设计**：`Layer` 需新增 `ComputeEngine* engine_`，并把 `init` 改成 **NVI
   （公共 `init(engine)` 绑定后转 `init_impl`）**——否则各层 override 的 `init` 不会调基类、
   `engine_` 留空。方法体内的 `engine` 用法保持不动的最小改法是函数首行加
   `ComputeEngine& engine = *engine_;` 局部别名（DSL 入口按 15 §3.3 保留形参，层传 `*engine_`）。
3. **未定点（需先裁定再动手）**：**Loss 没有天然绑定时机**（`nn::CrossEntropyLoss ce;` 默认
   构造、同一对象在测试里可跨 CPU/GPU 复用）——要么改构造签名 `CrossEntropyLoss(engine)`
   （src 约 19 处 + 文档范式 AGENTS §8），要么 Loss 保留 engine 形参并把 17 §5 M6 的
   "删形参"范围收窄为 Layer/辅助函数；15 §3.3 只说"Loss 排在 Layer 之后"，未给绑定方案。
   辅助函数（`RotaryEmbedding`/`PositionEncoder`/`ActivationOffloader`/`nn_dbg_scan`）同理：
   由持有层在 `init` 时 `bind(*engine_)`，或保留形参。
4. **验收**：四件套 + `NN_BIND_DEBUG=1` ctest 门禁（段 A 已打通）+ **layer_bench 配对 A/B**
   （参数删除是纯 API 变更，字节锚须逐位不变）。

**未决点（实施时裁定）**

| # | 问题 | 倾向 |
|---|---|---|
| U1 | `InitSpec::seed` 是否必填 | **已裁（M2 实施）**：分布类工厂签名 seed 必填（漏传 = 编译错误），层传显式默认值 `kInitSeed = 42`；引擎内按创建序号混流防同 seed 撞流 |
| U2 | `read/write` 元素类型形态：`span<T>` 精度精确匹配，还是按 `precision()` 重载 | **已裁（M3 实施）**：精确匹配 + 静态断言（防 f16/f32 槽错位）——错配运行期错误、元素类型非 float/f16 编译期报错；`read` 的 T 须非 const，`write` 形参 `span<T>` 去 cv 后校验（模板推导中 `const T` 无法匹配非 const span，需同时承接 const/非 const 实参） |
| U3 | CPU 真内存池立项与否 | 先分配剖析量化，另案 |
| U4 | 15 D6/D7/D8（宿主格式契约定序 / `Model::set_engine` 死码 / dummy 张量） | **已裁（M6 段 A 实施，2026-09-30）**：D6-1 = `load_model` v5 两支白做上传/往返**收敛为 `copy_from` 一行**（round-trip SHA256 逐字节一致）；D6-2 = **已由基类 `copy_from` 入口修复**（F16 目标走 `from_matrix`+`cast_into`，`copy_from_impl` 的 F32 槽假设仅在 F32 目标下可达），无代码改动；D6-3 = `matmul_with_bias_impl` 默认宿主兜底**改纯虚**（两引擎均已实现，删死码）；D7 = `Model::set_engine` **删除**（零调用方）；D8 = offload 占位 `Tensor::cpu(1,1)` → `create_tensor(1,1)` |
| U5 | 分层审计脚本落点：扩展 `bench/doc_inventory.ps1` vs 新建 audit | **已裁（M4 实施）**：扩展 `doc_inventory.ps1` 第 [4] 节——它本就枚举 L2 文件集（`compute_layer*.hpp` + loss + optimizer + model_container），不另建脚本 |
