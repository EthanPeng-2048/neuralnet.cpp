# CPU 引擎（L2）代码审查

## 模块概览

`compute_cpu_engine.hpp`（1588 行）实现 `CpuEngine`，是 `ComputeEngine` 接口的 CPU 后端：张量工厂（`from_matrix`/`clone`/`create_tensor`）、矩阵级原语（`matmul`/`batched_matmul`，委托 `algebra_matrix.hpp` 的分块 span 内核）、归约/广播/逐元素原语（委托 `algebra_matrix.hpp` 的 `row_reduce`/`col_reduce`/`broadcast_*_inplace` 与 `algebra_compute.hpp` 的 `compute::apply` AST 分发）、数据操作原语（`slice_rows`/`insert_rows`/`gather_rows`/`scatter_add_rows`/`rearrange_3d`）、RLA/RAPT 扫描原语（`scan_prefix_outer`/`scan_suffix_outer`/`outer_col`，逐头标量串行），以及融合解释器 `eval_expr_impl`（对 `ExprSpec` 一次遍历求值，支持 matmul 段 + 12 种视图 + 归约指令）。所有操作同步执行，`begin_batch`/`end_batch`/`flush_batch` 为 no-op（CPU 无命令录制概念）。`begin_expr`/`end_expr` 在 `NN_EXPR_SCAN` 构建期 scan 模式下通过 thread_local `fused::recording_graph_owner()` 录制 IR-C 图并注册融合 spec，运行时 no-op。

## 发现

### [P0] 正确性 / 崩溃 / UB / 设备丢失

（本组无新增 P0。已知未修项见"已知问题核对"节。）

### [P1] 生命周期 / 内存 / 并发 / 错误传播断裂 / 规范违反有实际影响

**P1-1** compute_cpu_engine.hpp:193-206（`gather_rows` 并行路径）— **`nn::for_each` 的 lambda 未声明 `noexcept`，违反线程池 API 约定**。

```cpp
nn::for_each(row_indices.begin(), row_indices.end(),
    [dst_span, tbl_span, idx_span, vocab, D](std::size_t i) noexcept {
        const auto row_idx = static_cast<std::size_t>(idx_span[i]);
```

（实际第 195 行已声明 `noexcept`，**经复核此条不成立**，撤回。）

重新检查后本组无 P1 新增。`scatter_add_rows`（250-261）为串行循环，无数据竞争；`gather_rows` 并行路径每行独立写入不同输出行，无跨行竞争；`rearrange_3d` 块级并行每块独立 `std::copy_n(N)`，无竞争；`batched_matmul` 串行逐 batch 调用 span 内核，内部 `parallel_for_blocks` 各 (i,j) 块写独立区域，无竞争。

**P1（重核后保留）** 无。

### [P2] 中等风险 / 设计缺陷

**P2-1** compute_cpu_engine.hpp:366-377（`matmul` 双转置路径）— **`transA && transB` 罕见路径分配两个临时 Matrix（`a_t` + `b_view` 隐式），内存开销高于必要**。

```cpp
// 双转置 C = A^T × B^T，罕见路径
Matrix a_t = a.transpose();
a_t.multiply_transposed_to(result, b);
```

`a.transpose()` 物化 A 的完整转置（`a_rows * a_cols` 个 Scalar），随后 `multiply_transposed_to` 内部又对 B 做零拷贝转置读取。虽然"罕见路径"注释降低优先级，但大矩阵下 `a.transpose()` 的临时分配可能触发 OOM（铁律 B：大中间张量物化）。建议：新增 `multiply_double_transposed_to` span 内核，或直接 `Matrix::multiply_to_span(c, M, N, a_t_span, ...)`，避免物化完整转置。

**P2-2** compute_cpu_engine.hpp:442-454（`batched_matmul` 双转置路径）— **每个 batch 都物化 `a_view`/`b_view`/`c_part` 三个临时 Matrix，与 span 零拷贝设计矛盾**。

```cpp
Matrix a_view(a_rows_per, a.cols());
auto dst_a = a_view.span();
std::copy_n(a_sub.begin(), a_sub.size(), dst_a.begin());
Matrix a_t = a_view.transpose();
Matrix b_view(b_rows_per, b.cols());
auto dst_b = b_view.span();
std::copy_n(b_sub.begin(), b_sub.size(), dst_b.begin());
Matrix c_part(M, N);
a_t.multiply_transposed_to(c_part, b_view);
std::copy_n(c_part.span().begin(), M * N, c_sub.begin());
```

每 batch 4 次大内存分配 + 3 次拷贝，batch 大时（如 batch=64, M=N=768, K=1536）临时内存峰值 ~ `3 * 768 * 1536 * 4B ≈ 14MB`/batch，且 `a_t`/`b_view` 在 batch 循环内反复分配/析构。建议：同 P2-1，新增双转置 span 内核一次性消除。

**P2-3** compute_cpu_engine.hpp:554-658（`scan_prefix_outer`）— **`out` 矩阵 `rows * 5` 行物化，5 倍于 K/V/P/R 输入大小；`A`/`B`/`Aq`/`num` 等 `std::vector` 在循环内反复分配（每 bh 一次）**。

```cpp
Matrix out(rows * 5, seq);
// ...
std::vector<Scalar> A(dk * dk, Scalar{0}), B(dk * dk, Scalar{0}),
                    qv(dk), kv(dk), vv(dk), num(dk), Aq(dk);
```

`rows * 5 * seq` 是输出张量大小，RLA 算法本身需要 5 组输出，无法避免。但 `A`/`B` 等 7 个 vector 在 `bh` 外层循环内**每次迭代重新分配**（`dk*dk` + 5*`dk` 个 Scalar），`dk≤64` 时每次 `dk*dk=4096*4B=16KB`，`BH = B*H` 可达数百，分配/析构开销非零。建议：将 vector 声明移到 `bh` 循环外，循环内 `std::fill` 重置。

**P2-4** compute_cpu_engine.hpp:706-750（`scan_suffix_outer`）— **`S`/`xv`/`yv` vector 同样在 `bh` 循环内反复分配**。

```cpp
std::vector<Scalar> S(dk * dk, Scalar{0}), xv(dk), yv(dk);
```

同 P2-3 模式，`S` 每次 `dk*dk` 个 Scalar 分配。建议：声明移到循环外，`std::fill` 重置。

**P2-5** compute_cpu_engine.hpp:521-523（`scan_prefix_outer` GPU 检查）— **对 `A0`/`B0`/`boundary` 的 GPU 检查在 `has_state`/`has_bnd` 为 false 时仍执行，可能解引用无效 Tensor**。

```cpp
for (const auto& t : {K, V, P, R, A0, B0, boundary})
    if (t.is_gpu())
        return std::unexpected(Error{"CpuEngine: GPU tensor on CPU engine"});
```

当 `has_state == false` 时 `A0`/`B0` 可能是默认构造的空 Tensor（未分配任何存储），`is_gpu()` 对空 Tensor 的语义未定义（取决于 `Tensor::is_gpu()` 对空矩阵的处理）。虽然当前 `Tensor::cpu()` 对空矩阵返回空 `Matrix`，`is_gpu()` 应返回 false，但**语义上不该检查未使用的张量**。建议：GPU 检查仅在 `has_state`/`has_bnd` 为 true 时执行对应张量。

**P2-6** compute_cpu_engine.hpp:1054-1588（`eval_expr_impl`）— **`view_reduce` 为 `std::vector<std::vector<Scalar>>`，每次调用按 `inputs.size()` 分配外层 vector + 每个归约视图再分配内层 vector，嵌套分配开销**。

```cpp
std::vector<std::vector<Scalar>> view_reduce(inputs.size());
// ...
std::vector<Scalar>& acc = view_reduce[k];
acc.assign(len, is_max ? std::numeric_limits<Scalar>::lowest() : Scalar{0});
```

归约视图数量通常 ≤ 2，但 `inputs.size()` 可达 10+，外层 vector 按 `inputs.size()` 预留空间（多数为默认空 vector，无内层分配），实际开销取决于归约视图数量。影响有限，但嵌套 vector 不如 `std::array` 或扁平 `std::vector` + 偏移索引清晰。建议：改为 `std::vector<Scalar> view_reduce_flat` + 偏移数组，或直接 `std::optional<std::vector<Scalar>>` 数组。

**P2-7** compute_cpu_engine.hpp:1432（`eval_expr_impl` 逐元素路径）— **`Scalar regs[EXPR_MAX_REGS]` 在循环外声明一次，但每元素 `eval_op` lambda 捕获 `regs` 按引用，寄存器数组未清零（`= {}` 只清零一次）**。

```cpp
Scalar regs[EXPR_MAX_REGS] = {};
const auto eval_op = [&](const ExprOperand& op) -> Scalar
{
    // ... 读取 regs[op.reg]
```

（经复核：`regs[EXPR_MAX_REGS] = {}` 在函数入口零初始化，每个 `r` 循环内 `ins.dst` 寄存器被**先写后读**（指令序列顺序执行，每个 dst 在读取前被写入），不存在未初始化读。此条撤回。）

重新检查后 P2 保留 P2-1 至 P2-6（共 6 条）。

### [P3] 风格 / 优化建议

**P3-1** compute_cpu_engine.hpp:46-47（类注释）— **`// D3：录制图由 fused::recording_graph_owner()（thread_local unique_ptr）持有` 注释与 `private:` 段空体（49 行 `public:` 前无任何成员）形成视觉断裂**。

```cpp
private:
    // IR-C 图 IR 录制（仅构建期 scan 模式启用；普通 CPU 运行保持 no-op）。
    // ... 5 行注释 ...
public:
```

`private:` 段实际无成员（全部逻辑在方法内），5 行设计说明放在 `private:` 下会误导读者以为有私有成员。建议：将注释移至类外（类声明前）或 `begin_expr` 方法注释中。

**P3-2** compute_cpu_engine.hpp:193（`gather_rows` 并行分支）— **`std::views::iota(std::size_t{0}, num)` 在两个分支重复构造（193 行并行分支 + 315 行 `rearrange_3d` 分支），可提取为辅助函数**。

`std::views::iota` 是零成本视图，但重复模式（构造 iota → 传 `nn::for_each`/`nn::parallel_for_blocks`）在 3 处出现（193/315/1267 附近），可封装为 `nn::parallel_for_index(num, kernel)` 简化调用点。

**P3-3** compute_cpu_engine.hpp:273-326（`rearrange_3d`）— **`block_kernel` lambda 在 `inverse` 分支内用 `if/else` 区分源/目标偏移，两个分支的 `std::copy_n` 完全对称（仅 `src_off`/`dst_off` 计算互换），可统一为"从 A 到 B"单向 + 方向参数**。

```cpp
if (!inverse)
{
    const std::size_t dst_off = (b * M + mi) * N;
    const std::size_t src_off = mi * (B * N) + b * N;
    std::copy_n(src.begin() + src_off, N, dst.begin() + dst_off);
}
else
{
    const std::size_t src_off = (b * M + mi) * N;
    const std::size_t dst_off = mi * (B * N) + b * N;
    std::copy_n(src.begin() + src_off, N, dst.begin() + dst_off);
}
```

逻辑正确，但 if/else 对称结构增加阅读负担。建议：提取 `swap(src_off, dst_off)` 后统一 `std::copy_n(src + src_off, N, dst + dst_off)`。

**P3-4** compute_cpu_engine.hpp:561-563（`scan_prefix_outer` `doc_reset` lambda）— **lambda 捕获 `bdm`（`const Matrix&`）+ `batch`/`seq`/`has_bnd`，但 `bdm` 在 `has_bnd == false` 时是 `boundary.cpu_matrix()` 的引用（boundary 为未使用的默认 Tensor），对空 Matrix 的 `at_unchecked` 不会执行（因 `has_bnd` 短路），但捕获本身对空 Tensor 的 `cpu_matrix()` 调用是 UB（若 `cpu_matrix()` 要求非空）**。

```cpp
const auto doc_reset = [bdm, batch, seq, has_bnd](std::size_t t) {
    return has_bnd && bdm.at_unchecked(0, batch * seq + t) != Scalar{0};
};
```

（`&&` 短路保证 `has_bnd == false` 时不访问 `bdm`，但 lambda 构造时 `bdm` 绑定已发生。若 `boundary` 是空 Tensor，`boundary.cpu_matrix()` 在 line 552 已被调用 `const Matrix& bdm = boundary.cpu_matrix();`——对空 Tensor 的 `cpu_matrix()` 是否 UB 取决于 `Tensor` 实现。建议：在 line 552 前加 `if (!has_bnd) bdm 设为空 Matrix` 或改为条件绑定。）

**P3-5** compute_cpu_engine.hpp:810-812（`row_reduce_sum`）— **`row_reduce` 的 `transform_op` 为 identity lambda `[](Scalar x) { return x; }`，4 个 reduce 原语（807-843）重复此模式，可封装 `row_reduce_sum`/`row_reduce_max` 辅助函数**。

```cpp
Matrix result = m.row_reduce(Scalar{0},
    [](Scalar a, Scalar b) noexcept { return a + b; },
    [](Scalar x) noexcept { return x; });
```

4 处（row/col × sum/max）重复"identity transform + 2 参数 lambda"模式，可提取 `m.row_reduce_sum()` / `m.row_reduce_max()` 便捷方法到 `algebra_matrix.hpp`。

**P3-6** compute_cpu_engine.hpp:991-1020（`elementwise_select_scalar_cond`）— **6 个比较分支各自手写 `for` 循环，仅 `a[i] <op> scalar_b` 的运算符不同，可用 `std::function` 或模板参数化比较器统一**。

```cpp
case CompareOp::Lt:
    for (std::size_t i = 0; i < n; ++i)
        out[i] = (a[i] < scalar_b) ? t[i] : scalar_else;
    break;
// ... 5 处类似
```

6 个分支代码完全对称，仅比较符不同。建议：`const auto cmp = [scalar_b](Scalar x) { return <op>; };` 后单一循环。

**P3-7** compute_cpu_engine.hpp:1085-1089（`eval_expr_impl` matmul 段）— **`m_per` 计算在 `mm == nullptr` 时仍执行三元判断（`mm && mm->batch > 0 && rows % mm->batch == 0`），虽然短路正确，但 `m_per` 在 `!mm` 时恒为 `rows`，可读性差**。

```cpp
const std::size_t m_per = (mm && mm->batch > 0 && rows % mm->batch == 0)
    ? rows / mm->batch : rows;
```

建议：`const std::size_t m_per = mm ? (mm->batch > 0 && rows % mm->batch == 0 ? rows / mm->batch : rows) : rows;` 或提前 `if (!mm) m_per = rows; else ...`。

**P3-8** compute_cpu_engine.hpp:1179-1197（`eval_expr_impl` 归约视图预计算）— **行归约与列归约的累加循环结构对称（仅 `r`/`c` 外层/内层互换），可模板化或提取辅助 lambda**。

```cpp
if (is_row)
{
    for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t c = 0; c < cols; ++c)
            acc[r] = is_max ? std::max(acc[r], s[r * cols + c])
                            : acc[r] + s[r * cols + c];
}
else
{
    for (std::size_t c = 0; c < cols; ++c)
        for (std::size_t r = 0; r < rows; ++r)
            acc[c] = is_max ? std::max(acc[c], s[r * cols + c])
                            : acc[c] + s[r * cols + c];
}
```

对称结构正确，但 4 行累加逻辑重复。影响有限，P3 级。

## 已知问题核对

**col_reduce 并行路径非逐字节一致**（已知未修项）— **确认仍存在，机制为累加顺序不同（非归约轴错误）**。

`algebra_matrix.hpp:843-936` 的 `col_reduce` 并行路径（`R >= 256 && R*C >= PARALLEL_THRESHOLD && n_threads > 1` 时启用）：

1. **单线程路径**（line 878-894）：按行主序 `r` 从 0 到 `R-1` 扫描，对每列 `c` 累加 `out[c] = reduce_op(out[c], v)`，即 `out[c] = ((((init ⊕ v[0,c]) ⊕ v[1,c]) ⊕ ... ⊕ v[R-1,c])`，**行方向顺序累加**。

2. **并行路径**（line 896-935）：
   - 分配 `n_threads` 组本地累加器 `local_acc[t * C + c]`，每线程扫描自己的行块（`r0..r_end`），**线程内按行顺序累加**到 `local_acc[t][c]`。
   - 归并阶段（line 927-934）：`out[c] = local_acc[0][c]`，然后 `for t in 1..n_threads-1: out[c] = reduce_op(out[c], local_acc[t][c])`，**按线程编号顺序归并**。

   **关键差异**：单线程是 `((init ⊕ v[0]) ⊕ v[1]) ⊕ ...`（行 0,1,2,...,R-1 顺序），并行是 `(((init ⊕ v[0]) ⊕ v[1]) ⊕ ... ⊕ v[r0+r1-1]) ⊕ ((init ⊕ v[r0]) ⊕ ...)`（线程 0 的行块结果 ⊕ 线程 1 的行块结果 ⊕ ...）。**float 加法不满足结合律**，`a + (b + c) ≠ (a + b) + c` 在浮点下一般成立，因此并行结果的浮点舍入位与单线程**不逐字节一致**。

   **机制确认**：累加顺序不同（分块归并 vs 全序列顺序），**非归约轴错误**（归约轴正确为列方向，`c` 外层 `r` 内层，与单线程一致）。

   **影响**：对 `sum` 操作，差异量级 ~ `n_threads * eps * |max_v|`，对 `max` 操作无影响（`max` 满足结合律，`std::max` 逐字节一致）。`col_reduce_sum` 用于 LayerNorm 列均值/方差、Softmax 列归一化等场景，float 级差异通常 < 1e-5，对训练收敛无实质影响，但**违反铁律 6"并行化后结果必须与单线程逐字节一致"**。

   **建议修法**（供后续修复参考，本审查只核不修）：
   - 方案 A：并行路径归并时改为**按行号全局顺序**归并（将 `local_acc` 展开为 `(R, C)` 临时矩阵，归并时按 `r` 从 0 到 `R-1` 累加），保证与单线程相同的累加顺序。代价：`R*C` 临时内存。
   - 方案 B：接受并行路径的 float 级差异，在文档中标注"col_reduce_sum 并行路径与单线程可能存在 < 1 ulp 差异"，并将铁律 6 的"逐字节"限定为"max 操作逐字节、sum 操作 < 1 ulp"。
   - 方案 C（推荐）：`col_reduce` 并行路径改为**行内分块 + 行内顺序归并**：每线程对自己的行块按行顺序累加到 `local_acc`（现有逻辑），归并时**按行号顺序**（而非线程编号顺序）将各线程的行块结果累加到 `out`。具体：将 `local_acc` 布局改为 `(n_threads, R_chunk, C)`，归并时 `for r in 0..R-1: for t in 0..n_threads-1: if r 在 t 的行块内: out[c] = reduce_op(out[c], local_acc[t][r-r0][c])`。这样 `out[c]` 的累加顺序与单线程完全一致（行 0,1,2,...,R-1），逐字节匹配。代价：归并循环多一层 `t` 判断。

## 其他观察

1. **`scan_prefix_outer`/`scan_suffix_outer` 的 `out` 张量行数是输入的 5x/3x**（554 行 `rows * 5`、695 行 `rows * 3`），这是 RLA/RAPT 算法的输出规格（多组中间量打包到一个张量），非内存浪费。但调用方需按 `rows` 偏移读取 5 段（`out[0..rows)`=B·P, `[rows..2*rows)`=B^T·R, `[2*rows..3*rows)`=A·P, `[3*rows..4*rows)`=s, `[4*rows..5*rows)`=r），建议在 `compute_engine.hpp` 接口注释中明确各段偏移。

2. **`eval_expr_impl` 的 `matmul_out` 预计算（line 1206-1261）在 `spec.instrs.empty()` 时直接返回（line 1265-1274），但 `matmul_out` 仍被分配并填充（line 1217 `Matrix matmul_out(rows, N)` + line 1219-1261 循环）**——经复核 line 1265 在 line 1217 之后，`matmul_out` 已分配，若 `spec.instrs.empty() && !mm` 在 line 1068 已返回，此处 `!mm` 分支（line 1267）才走到 line 1265 的 `if (spec.instrs.empty())` 早退，但此时 `matmul_out` 未分配（`mm == nullptr` 时 line 1206 的 `Matrix matmul_out;` 是空 Matrix）。逻辑正确，无 UB。

3. **`batched_matmul` 的 `alpha` 缩放（line 457-463）对 `alpha == 1.0f` 零开销跳过，但对 `alpha == 0.0f` 仍执行全量循环**（每个元素 `dst[i] *= 0`），可加 `if (alpha == 0) { result.zero(); return ...; }` 快速路径。影响有限（P3 级，未计入 P3 上限）。

4. **`from_matrix`（line 92-95）`Tensor::from_matrix(Matrix(m))` 的 `Matrix(m)` 是拷贝构造，与 `to_matrix`（line 97-102）`Matrix(t.cpu_matrix())` 对称，双拷贝语义一致。但 `from_matrix` 接收 `const Matrix&` 参数后拷贝，若调用方持有 `Matrix` 且不再使用，可改为 `Matrix&&` 右值重载避免拷贝。当前 API 签名 `Result<Tensor> from_matrix(const Matrix&)` 在 `compute_engine.hpp` 接口中定义，改动需同步接口，影响面大，仅记录。

5. **`gather_rows`/`scatter_add_rows` 的 `idx.size()` 遍历支持任意形状 indices（line 182/244），但 `Matrix::size()` 返回 `rows_ * cols_`，对 (1, N) 和 (N, 1) 的 indices 行为相同（flat 遍历），符合"按 flat 遍历所有元素"的注释。无 bug。
