# ComputeEngine Refresh —— 详细设计与未决问题（2026-09-28）

> 本文是 `docs/development/13-refactor-backlog.md` §10 的展开版：补全设计细节、
> 列出模糊点并给出各方案利弊。**本文只记录方案，不实施。**
> 立项时读本文 + 13 §10 + 对应源码现状。历史归档见 `docs/history.md`。
> 修订：D1 已裁定（PrecisionEngine 下沉，§4.1）；D9 已裁定（分层验收，§4.9）；
> D2 补真实代码片段并修正结论（§4.2）。
> **实施进度：P0 完成（盘点见 16）；P-1 完成（2026-09-29，验收：dev1 ctest 20/20、
> CPU 探针与 P0 基线逐字节、scan 产物 hash 不变、layer_bench 交错 A/B 无系统性回退，
> 过程归档 `docs/history.md`）。P1-P6 未实施。**

## 0. 一句话

把"张量出生即绑定引擎"下沉为构造期不变量，删掉 Layer/Loss/序列化的**每调用
engine 形参**，用 `import` 收敛跨设备互传，用存储多态消掉容器头的
`#ifdef NN_HAS_VULKAN`；**前置一步：PrecisionEngine 下沉进基类、该类整体删除**。

## 1. 现状事实（可复现坐标，2026-09-28 核对）

| 事实 | 数字 | 复现 |
|---|---|---|
| `include/` 中 `ComputeEngine&` 出现 | 175 处 | `grep -rn "ComputeEngine&" include/ \| wc -l` |
| `Result<Tensor>`（全仓头文件） | 203 处 | 同上换 pattern（值语义零改动的依据） |
| `GpuEngine::ensure_gpu` 调用点 | 43 处 | `grep -n "= ensure_gpu(" compute_gpu_engine.hpp` |
| `compute_engine.hpp` virtual | 50（49 + 析构） | `grep -c "virtual"` |
| `compute_tensor.hpp` 的 `#ifdef NN_HAS_VULKAN` | 7 处 | 行 28/65/99/161/205/245/304 |
| 宿主中转调用（from/to_matrix/copy_from，排除引擎自身） | ~596 处 | `grep` 后过滤引擎头 |
| `Model::set_engine` | **零调用方 = 死码** | `grep -rn "set_engine(" src/` 空 |
| `PrecisionEngine` | 独立装饰器类（>900 行），49 方法全量委托 `inner_` | `compute_precision_engine.hpp` |
| 测试直构张量（`Tensor::from_matrix`/`Tensor::cpu`） | 90 处 / 8 文件 | `grep -c src/*_test.cpp` |
| 三处绑定先例 | Model(observer_ptr)、Optimizer(`ComputeEngine&`)、`Layer::init` | 各头文件 |

## 2. 目标与非目标

**目标**
1. **P-1：PrecisionEngine 下沉删除**（§4.1）——消解 D1 绑定身份问题。
2. P1-P3：张量出生即绑定 → 删每调用 engine 形参（编译器穷尽驱动）。
3. P3：跨引擎混用升 `Result` 硬错误；`ensure_gpu` → 显式 `import`。
4. P5-P6：存储多态，新后端不改 `compute_tensor.hpp`。
5. P4（正交另案）：宿主格式契约修复。

**非目标（明确不做）**
- `Tensor` 抽象基类/句柄指针化（203 处 `Result<Tensor>` 连锁，13 §10.8 已否决）。
- DSL 入口 `dsl::compute(engine, expr, rows, cols[, P])` 签名不动。
- **去掉的是 PrecisionEngine 这个类，不是"边界 cast"本身**——f16 无变体算子
  "抬 f32 算 → 按 P 落回"的语义照旧，只是换地方住（§4.1）。
- 不改 AOT 闭合世界、不改 batch-major 布局、不动 `PrecisionProfile` 归属。

## 3. 设计详解

### 3.1 绑定的存储位置与传播规则

```cpp
class Tensor {
    observer_ptr<ComputeEngine> engine_ = nullptr;   // 新增，+8B
public:
    [[nodiscard]] bool bound() const noexcept { return engine_ != nullptr; }
    [[nodiscard]] ComputeEngine& engine() const;      // 未绑定 → NN_ASSERT
};
```
- **出生点唯一**：`engine.create_tensor` / `engine.from_host` / 算子输出。
  实现上在引擎工厂函数尾部统一 stamp，避免每个 override 各自漏 stamp。
- **库内配套通道**（D2 修正的关键）：`dsl::compute` 出口、引擎默认实现等
  "有引擎在场但产物走静态工厂"的位置，统一经内部
  `ComputeEngine::adopt(Tensor&&)`（包内/friend，非公有 API）stamp。
- **传播**：`clone/cast/reshape/slice_rows/...` 沿用 `src` 绑定；
  `Tensor{}` 空槽不绑定，`valid()==false`。
- **生命周期契约**：engine 必须比它创建的 Tensor 活得久；一期不加析构探测，
  出悬垂事故再评估 `weak_ptr<EngineToken>`（13 §10.3 第 3 条）。

### 3.2 `import` 协议（唯一对外互传入口）

```cpp
// 目标侧拉取；默认实现 = 经宿主中转；同引擎 fast path 在默认实现内。
[[nodiscard]] virtual Result<Tensor> import(const Tensor& src,
                                            Precision P = /*src 精度*/);
```
- 同引擎 + 同 P = 零拷贝别名；同引擎 + 异 P = 引擎内 cast。
- 异引擎 = `src.to_host → cast_on_host → this->from_host`；引擎可 override
  为同设备直拷（将来 peer copy 调用方零改动）。
- 三个宿主中转特例改名（P2）：`to_matrix→to_host`、`from_matrix→from_host`、
  `copy_from→upload`；`forward/backward` 内禁用（命名即防线，grep 旧名可审计）。
- 别名约束见 D4：**只在同绑定张量之间别名**。

### 3.3 删形参的顺序（P2）

编译器报错驱动，顺序：Layer（`forward/backward/zero_grad/forward_recompute`）
→ Loss → 辅助函数（RoPE、`ActivationOffloader`、`nn_dbg_scan`…）→ CLI/测试。
层内统一用成员 `engine_`（`init` 绑定）。DSL 入口保留形参，层传 `*engine_`。

### 3.4 存储多态（P5）

```cpp
class TensorStorage {                    // 定义在 compute_tensor.hpp
    Device dev_; Precision p_; size_t rows_, cols_;
    virtual Result<void> reshape_view(size_t r, size_t c) = 0;  // 第 8 条
    virtual ~TensorStorage();
};
// CpuStorage<P> / GpuStorage<P>（GpuStorage 只能由 GpuEngine::create_tensor
// make_shared —— 第 7 条，防 storage→backend→engine→tensor 成环）
```
- `Tensor` 只持 `shared_ptr<TensorStorage>`；`#ifdef` 与 backend include 从容器头消失。
- `as_cpu()/as_gpu()`（P6）：检查式 downcast，非虚，句柄仍值类型。
- 逐元素热点（`cpu_get_ptr` 8-70× 教训）：**访问器机制不变**——storage 多态
  只动"容器头怎么拿到 storage"，取出 `MatrixT<P>*` 后的热循环零变化。
  P5 验收必须含 `layer_bench` 无回退。

### 3.5 验收基线（分层，见 D9 裁定）

- build 全绿 + ctest 20/20（GPU 用例退出码 77 = skip 如实记录）。
- **CPU 同设备 pre/post**：loss 序列 + 权重 MD5 **逐字节**。
- **GPU 同设备 pre/post**：P0 实测 run-to-run 稳定性；稳定 → 逐字节，
  不稳定 → 容差（loss ε + 每张量 max-abs-diff ε）。
- **跨设备（CPU↔GPU）**：永不逐字节，只比容差。
- **scan 产物 hash**（`expr_specs.bin` / `fused_registry.hpp`）：与设备无关，
  恒逐字节。

## 4. 设计点逐条裁定

### 4.1 D1【已裁定】PrecisionEngine 直接去掉——下沉为基类 NVI

**裁定：能去，且必须去。** 精度在张量出生时已由 `PrecisionProfile` + 显式 P
定死，"装饰器每次现问内层要不要 cast"本身就是错位——**能力查询已经长在
接口上**（`supports_native_data_move` / `supports_native_f16_reduce` /
`supports_expr_precision_variant` 就是给这个决策用的），说明正确的分层是
"基类问能力 → 基类决定 cast → 引擎只管算"：

```cpp
class ComputeEngine {
public:
    // 非虚入口（NVI）：原 PrecisionEngine 的边界 cast 逻辑住这里
    [[nodiscard]] Result<Tensor> matmul(const Tensor& A, const Tensor& B,
                                        bool tA, bool tB, Precision P)
    {
        if (/* 输入含 f16 且 !supports_f16_matmul(A,B) */)
        {
            auto a32 = cast(A, Precision::F32);          // 抬
            auto b32 = cast(B, Precision::F32);
            auto r   = matmul_impl(*a32, *b32, tA, tB, Precision::F32);
            if (!r) return std::unexpected(r.error());
            return cast(std::move(*r), P);               // 按 P 落回
        }
        return matmul_impl(A, B, tA, tB, P);
    }
protected:
    virtual Result<Tensor> matmul_impl(...) = 0;         // 引擎只实现这个
};
```

**收益**
1. **D1 消失**：没有装饰器 = 没有"stamp 谁"问题，张量出生绑定的就是唯一引擎。
2. `text_train.cpp:955-966` 的 `optional<PrecisionEngine>` + `static_cast` 舞步删除；
   `scan_exprs` 的 f16 dry-run pass 同理（它现在显式包一层跑 f16 登记）。
3. 49 个方法的全量委托样板（`begin/end_batch`、工厂、扫描……逐个转发 `inner_`）消失。
4. in-place 存储精度不变量（`cast_into` 写回原存储）、f16 临时量归因
   （`note_temp_`）各归一处，不再有"适配层管一半、引擎管一半"。

**成本与风险**
1. 49 个 virtual 改名 `*_impl` + 三引擎 override 跟随（编译器驱动，机械改动）。
2. f32-only 路径多一个可预测分支（今天全 f32 不包适配层）→ `layer_bench` 对拍。
3. 归因 instrumentation 挪家（建议基类入口内、环境变量门控）。
4. `compute_engine.hpp` 会变大——与 §2 "拆大头文件"方向不冲突（删的是另一个
   文件），但拆分顺序要排在下沉之后。

**插入时机**：新增 **P-1**，排在 P0（字节基线已建立）之后、P1（加绑定）之前——
先下沉，P1 就不用讨论 D1；用 P0 基线验收下沉"逐字节零回归"。

### 4.2 D2【修正后】"唯一创建入口" vs 满地静态工厂——真实代码片段

原稿说 13 §10.3 第 1 条"非绑定态仅剩 `Tensor{}`"**过强**，核对后修正：冲突点比
想象的窄，库内几乎都有引擎在场，缺的只是一个 stamp 通道。

**片段 A：`expr_dsl.hpp` eval_cpu（约 :1290-1302）——CPU 求值出口**

```cpp
template <typename E>
[[nodiscard]] Tensor eval_cpu(const E& e, std::size_t rows, std::size_t cols,
                              Precision P = Precision::F32)
{
    Matrix out = Matrix::make_uninitialized(rows, cols);
    eval_into_span(e, out.span(), cols);
    if (P == Precision::F16)
    {
        MatrixT<Precision::F16> h(rows, cols);
        ...
        return Tensor::from_matrix(std::move(h));   // ← 产物未绑定任何引擎
    }
    return Tensor::from_matrix(std::move(out));     // ← 同上
}
```
调用方 `dsl::compute(eng, ...)` **手里有 `eng`**，但 CPU 分支直接
`return eval_cpu(...)`。产物是"valid 但未绑定"的 CPU 张量，随后流进下一层
`forward`——若 P1 检查写成 `input.engine() != engine_ → 报错`，**这条正常
CPU 训练路径会被误杀**。→ 修法：`dsl::compute` 出口统一
`return eng.adopt(eval_cpu(...))`（内部通道，非公有 API）。

**片段 B：`expr_dsl.hpp` 扫描占位（:1344-1358）**

```cpp
#ifdef NN_EXPR_SCAN
    (void)eng;                                     // ← 引擎在场，被刻意丢弃
    auto [spec, inputs] = to_expr_spec(e);
    ...
    fused::global_registry().add(spec, sig);
    return P == Precision::F16 ? Tensor::cpu<Precision::F16>(rows, cols)  // 未绑定
                               : Tensor::cpu(rows, cols);                 // 未绑定
#endif
```
占位张量会流进 dry-run 下游层（缓存、切片、后续表达式叶子）。→ 修法同 A：
占位也 `eng.adopt(...)`（scan 模式 eng 是真引擎，CpuEngine/PrecisionEngine）。

**片段 C：`src/fused_gpu_test.cpp` :72-76——测试双引擎互喂**

```cpp
const Tensor qc = Tensor::from_matrix(Matrix(q));  // 宿主直构，未绑定
auto cr = rope_cpu.apply(cpu, qc, seq, backward);  // 喂 CPU
auto gr = rope_gpu.apply(gpu, qc, seq, backward);  // 同一个 qc 再喂 GPU
```
90 处测试直构张量是**库外**代码。若严格执行"非绑定不得参与计算"，全要改成
`cpu.from_host(q)` + `gpu.from_host(q)`（或 `gpu.import(qc)`）。

**片段 D：`compute_engine.hpp` :104/:117——基类默认实现**

```cpp
[[nodiscard]] virtual Result<Tensor> create_offload_buffer(std::size_t)
{
    return Tensor::cpu(1, 1);    // ← 看似无引擎，其实 this 就在——可 stamp
}
```
说明：**引擎成员函数里的静态工厂调用都能 stamp（有 `this`）**，此前高估了范围。

**修正结论（分库内/库外）**
- **库内不变量保持**（13 §10.3 第 1 条不作废）：所有库内产物出生即绑定——
  靠两个配套：① `ComputeEngine::adopt` 内部 stamp 通道（DSL 出口、默认实现、
  扫描占位）；② 工厂尾部统一 stamp。
- **库外（测试/示例）**：P2 起迁移到 `engine.from_host`；编译器抓不到
  （运行期才炸）→ P1 检查带 `NN_BIND_DEBUG=1` 开关，ctest 在该开关下把
  "未绑定输入进引擎"也报错，抓漏网；P3 视情况转为默认开。

### 4.3 D3【倾向定案】绑定判等粒度

指针判等；**P1 检查仅在"两操作数都 bound 且指针不同"时报错**，未绑定输入按
4.2 库外豁免放行（避免 P1 零行为变化阶段 ctest 变红）。双引擎对拍测试在 P3
改显式 `import`。同设备多 `GpuEngine` 实例（共享 `GpuBackend::instance()`）
按不同引擎处理——这正是"显式化"的目的。待 P0 测试盘点最终确认。

### 4.4 D4【待定案】`import` 别名语义

同引擎 fast path 的零拷贝别名**只允许发生在同绑定张量之间**；跨引擎产物必是
新 storage + 新绑定。否则"绑定后不可换引擎"不变量被共享 storage 打破。
倾向此约束，待写进 13 §10.6。

### 4.5 D5【待定案】`to_host/upload` 的空槽与 Const

未绑定张量调 `to_host` → `Result` 错误（"unbound tensor: use Matrix directly"）。
`upload` 与 `cast_into` 同一红线：**不替换 dst 对象**（否则共享句柄的缓存静默失联）。

### 4.6 D6【待定序】P4 三处宿主格式契约

1. `model_serialization.hpp:651-674` 同精度路径丢弃 `from_matrix` 结果
   （每参数白做一次全量上传）——P2 改名时要预先标注"计划内行为变化"。
2. `CpuEngine::copy_from` 硬编码 F32 槽（`compute_cpu_engine.hpp:210`，
   f16 dst 在 NDEBUG 下空指针）——**先修再搬**还是搬完再修需定序。
3. `matmul_with_bias` 默认实现 host 往返兜底是死码（三引擎均 override）——
   删前 grep 确认无第四个实现者。

### 4.7 D7【待定案】`Model::set_engine` 死码

零调用方；绑定后"换引擎"与"层已 init 绑定"冲突。倾向直接删除；保留则须
定义为"只能在 `add` 层之前调用"。

### 4.8 D8【待定案】dummy/占位张量

`make_dummy_`（RAPT）本就走 `engine.create_tensor(1,1)` → 自动 stamp，无事；
`create_offload_buffer` 等 CPU 默认实现从 `Tensor::cpu(1,1)` 改为
`create_tensor(1,1)`（同 4.2 片段 D）。P2 顺手处理。

### 4.9 D9【已裁定】字节基线分层——跨设备永不逐字节

**裁定依据**：CPU/GPU 即使精度相同，硬件差异（归约顺序/分块、FMA 收缩、
shader 内循环结构）也会造成小幅偏移；GPU 侧另有原子累加顺序（embedding 反向
`scatter_add_rows` 类路径）这类 run-to-run 非确定源。所以：

| 对比维度 | 验收标准 |
|---|---|
| CPU，改动前 vs 改动后（同 seed 同机） | **逐字节**（分块边界只由 total 决定，确定性有保证） |
| GPU，改动前 vs 改动后（同 seed 同设备同驱动） | **P0 实测两次裸跑**：稳定 → 逐字节；不稳定 → 容差（loss ε + 每张量 max-abs-diff ε） |
| CPU vs GPU（同 seed） | **永不逐字节**，只比容差（现状测试已如此，如 `err < 1e-4`） |
| scan 产物 hash | 恒逐字节（构建期产物，与设备无关） |

P0 的"字节基线"因此拆成两半：**同设备 pre/post 锚点**（不变）+
**GPU run-to-run 稳定性实测**（新增，决定 P1/P5 的 GPU 验收档位）。

## 5. 分期速查（含新增 P-1）

> P0 已完成，盘点结果见 `docs/development/16-computeengine-p0-inventory.md`
> （GPU 档位实测尚缺，见 16 §7-2 未决项）。

| 期 | 内容 | 关键点 |
|---|---|---|
| P0 | 0 改动调研：43 ensure_gpu 分类 / 175 打标 / 字节+scan 基线 / 宿主中转盘点 / **GPU run-to-run 稳定性实测** | D9 实测、D6 调用方 |
| **P-1** ✅（2026-09-29） | **PrecisionEngine 下沉删除（NVI）**：基类非虚入口 + protected `*_impl`（CPU 33 / GPU 29 处机械改名），`compute_precision_engine.hpp` 删除，7 处使用方迁移 | 验收全过：dev1 ctest 20/20（含 4 个 f16 用例）、CPU 探针与 P0 基线逐字节、scan 产物 hash 不变、layer_bench 交错 4 轮无系统性回退（A/B 过程归档 history.md） |
| P1 | 加绑定 + 跨引擎检查（只加不改） | D2（adopt 通道）、D3 |
| P2 | 删形参 + 宿主动词改名 + 测试 `from_host` 迁移 | D6 定序、D7、D8 |
| P3 | 检查升硬错误 + 43 处 ensure_gpu → import + 序列化去 `model.engine()` | D4、D5 |
| P4 | 宿主格式契约（**正交另案**） | — |
| P5 | 存储多态 + reshape virtual | `layer_bench` 热点回归 |
| P6 | as_cpu/as_gpu 访问器（纯加法） | — |

## 6. 与既有文档关系

- 方案出处与选型记录：`docs/development/13-refactor-backlog.md` §10（含 §10.8 否决记录；
  §10.3 第 1 条按本文 4.2 补"库内/库外"脚注）。
- 引擎接口盘点：`docs/development/12-compute-engine-inventory.md`。
- 精度边界语义（P-1 下沉对象）：`docs/development/05-mixed-precision.md` §8/§12。
- 历史归档：`docs/history.md`。

## 7. 当前进度与交接（2026-09-29）

### 7.1 已完成

| 期 | 产出 | 验收（均可复现） |
|---|---|---|
| P0 | `16-computeengine-p0-inventory.md`（ensure_gpu 43 分类 / ComputeEngine& 175 打标 / 宿主中转 381 清单 / ctest 双基线）；`src/gpu_stability_probe.cpp`（app 目标，ctest 仍 20 个） | 见 16 §0 |
| P-1 | PrecisionEngine 删除 + NVI 下沉（本文 §4.1/§5）；7 处使用方迁移；history.md 归档条目 | ① `NN_VULKAN_DEVICE=1 ctest` = 20/20；② `build/gpu_stability_probe --steps 20` 与 P0 基线逐位一致（`grep -E '^(run1)'` diff 为空）；③ `sha256sum build/generated/expr_specs.bin` = `bdc3a442a5a47a4cf6c0b3ced83c05c870c6fc4ce503681a4a97aca5d6a58360`、`fused_registry.hpp` = `d751008c3ad6074ff444ae59a75ef7c0137de988428644843f37d6581b1df714`（scan 产物锚点，后续重构不得变动） |

### 7.2 下一步：P1（张量出生绑定 + 跨引擎检查，只加不改）

1. 先读：本文 §3.1（绑定存储与传播规则）+ §4.2 D2（`adopt` 库内通道、真实代码片段）+ §4.3 D3（指针相等判定；**双方都绑定且不同才报错**，单侧未绑定豁免）。
2. 改动面：`Tensor` 加 8B `engine_` observer；出生点（`create_tensor`/`from_matrix`/算子输出）统一 stamp；**P1 只加检查、不删形参**（删形参是 P2）。
3. 验收：与 P-1 相同四件套（ctest 20/20 / CPU 探针逐位 / scan hash / layer_bench 抽查）+ `NN_BIND_DEBUG=1` 打点（§4.2）；P1 结束时形参还在，仅混引擎误用会硬报错。
4. 注意：engine 生命周期需比其创建的 Tensor 长（§3.1）；P1 不加析构探测（13 §10.3 第 3 条留待出事故再评估）。

### 7.3 搜置与未决（后续期处理，均不阻塞 P1）

- **GPU 档位未定**：`gpu_stability_probe --gpu` 首个 `vkCreateBuffer` 即崩（16 §7-2，同设备同构建下既有测试全部正常 → 差异在探针路径，未定位）。修好后回填 16 §6 GPU 行，D9 的 GPU 验收档位（逐字节 vs 容差）才有结论。
- **Mali（GPU0）offload 非确定失败**（16 §5/§7-1）：设备级问题，另立 issue，不阻塞 refresh；**ctest 基线以 `NN_VULKAN_DEVICE=1`（Lavapipe 20/20）为准**，本机自动选卡会落到 Mali。
- P0 遗留：`ComputeEngine&` REVIEW 7 处待 P2 逐条判定（16 §3）；宿主中转口径勘误 596→381，P2 改名清单按 381（16 §4）。
- layer_bench 精确性能复测：本机为手机（DVFS/调度噪声 ±30%，A/B 结论只能到“无系统性回退”），精确数字待桌面平台（history.md P-1 条目）。

### 7.4 环境注记（本机 Termux/Android，后续会话复用）

- Vulkan 两卡：GPU0 = Mali（apiVersion 1.1.177，自动选卡落点）、GPU1 = Lavapipe（1.4.354）；选卡 `--gpu` / `NN_VULKAN_DEVICE`（`cli/cli_gpu_option.hpp`）。
- 验收命令集：
  ```bash
  cmake -B build -G Ninja -DNN_ENABLE_TESTS=ON && cmake --build build -j8
  NN_VULKAN_DEVICE=1 ctest --test-dir build          # 门槛：20/20
  build/gpu_stability_probe --steps 20               # CPU 逐位（exit 0）
  sha256sum build/generated/expr_specs.bin build/generated/fused_registry.hpp  # 对 §7.1 锚点
  ```
