#pragma once

// ── compute_engine.hpp — 计算引擎抽象接口 ─────────────────────────────────
// ComputeEngine 是与底层硬件接触的唯一抽象层。
//
// 设计原则（铁律）：
//   1. 本接口只提供 op-level 原语（矩阵乘法、加法、转置、归约、广播、
//      逐元素运算等），绝不包含任何算法。
//   2. ReLU、GeLU、LayerNorm、Softmax、Attention 等算法由 Layer 层
//      通过组合原语表达。
//   3. Layer 持有 ComputeEngine 引用，forward/backward 只写一次，
//      CPU/GPU 由引擎实现自动分发。
//
// 原语分类：
//   - 矩阵级：matmul, batched_matmul, transpose, add_inplace, scale_inplace, zero
//   - 归约级：row_reduce_sum, col_reduce_sum
//   - 广播级：broadcast_row_inplace, broadcast_col_inplace
//   - 逐元素：elementwise_unary, elementwise_binary, elementwise_binary_scalar
//   - 条件选择：elementwise_select_scalar_cond
//
// 批处理控制：
//   - begin_batch / end_batch：CPU 引擎为 no-op；GPU 引擎录制到
//     command buffer，end_batch 时统一提交。
// ─────────────────────────────────────────────────────────────────────────

#include <cstddef>
#include <span>
#include <string>
#include <vector>

#include "core_config.hpp"
#include "core_errors.hpp"
#include "compute_tensor.hpp"
#include "expr_spec.hpp"

namespace nn
{

// ══════════════════════════════════════════════════════════════════════════
// 算子枚举（op-level，不含算法语义）
// ══════════════════════════════════════════════════════════════════════════

// 一元算子
enum class UnaryOp : uint32_t
{
    Neg   = 0,  // -x
    Exp   = 1,  // e^x
    Log   = 2,  // ln(x)
    Sqrt  = 3,  // √x
    Rsqrt = 4,  // 1/√x
    Abs   = 5,  // |x|
    Tanh  = 6,  // tanh(x)
};

// 二元算子
enum class BinaryOp : uint32_t
{
    Add = 0,  // a + b
    Sub = 1,  // a - b
    Mul = 2,  // a * b
    Div = 3,  // a / b
    Max = 4,  // max(a, b)
    Min = 5,  // min(a, b)
};

// 比较算子（用于条件选择）
enum class CompareOp : uint32_t
{
    Lt = 0,  // a <  b
    Le = 1,  // a <= b
    Gt = 2,  // a >  b
    Ge = 3,  // a >= b
    Eq = 4,  // a == b
    Ne = 5,  // a != b
};

// 归约算子（matmul 融合原语用，op-level 无算法语义）
enum class ReduceOp : uint32_t
{
    Sum = 0,
    Max = 1,
    Min = 2,
};

// ══════════════════════════════════════════════════════════════════════════
// ComputeEngine — 计算引擎抽象接口
// ══════════════════════════════════════════════════════════════════════════
class ComputeEngine
{
public:
    virtual ~ComputeEngine() = default;

    // ── 设备查询 ──────────────────────────────────────────────────────────
    [[nodiscard]] virtual Device device() const noexcept = 0;

    // ── 批处理控制 ────────────────────────────────────────────────────────
    // CPU 引擎：no-op（操作立即同步执行）
    // GPU 引擎：begin 开始录制，end 统一提交 + fence wait
    [[nodiscard]] virtual Result<void> begin_batch() = 0;
    [[nodiscard]] virtual Result<void> end_batch() = 0;

    // ── 批处理中点刷新（防 TDR） ─────────────────────────────────────────
    // GPU 引擎：提交当前 command buffer 并等待完成，然后自动开始新的录制。
    // 可在 forward 与 backward 之间调用，将一次大提交拆分为多次小提交，
    // 避免单次提交时间过长触发 Windows TDR。
    // CPU 引擎：no-op。
    [[nodiscard]] virtual Result<void> flush_batch() { return {}; }

    // ── 显存回收（L2）─────────────────────────────────────────────────
    // GPU 引擎：在 end_batch（提交完成、延迟销毁已 flush）之后归还完全
    // 空闲的内存池底材给 GPU。CPU/CUDA 引擎：no-op。
    [[nodiscard]] virtual Result<void> release_idle_pool_blocks() { return {}; }

    // ── 显存池统计（L2 仪器化）──────────────────────────────────────
    // GPU 引擎返回池统计字符串（块数/占用/空闲/碎片）；CPU/CUDA 返回空。
    // 用于训练中显存采样与逐项归因。
    [[nodiscard]] virtual std::string pool_stats() const { return {}; }

    // ── 激活 offload（L1-offload）───────────────────────────────────
    // offload_store：把 GPU 激活复制到 host-visible 存储（释放 device-local
    //    VRAM），返回一个 opaque 句柄。GPU 引擎录制式（batch 内不提交）；
    //    CPU 引擎 no-op（返回 src 本身）。
    // offload_load：从 offload_store 的句柄复制回 GPU，恢复为 (rows, cols)。
    //    CPU 引擎 no-op（返回句柄 reshape 为 rows×cols）。
    [[nodiscard]] virtual Result<Tensor> offload_store(const Tensor& src) { return src; }

    [[nodiscard]] virtual Result<Tensor> offload_load(
        const Tensor& handle, std::size_t rows, std::size_t cols)
    {
        return handle.reshape(rows, cols);
    }

    // ── activation offload slab（L1-offload，持久复用缓冲） ────────────
    // 每个 GPTBlock / RAPTBlock 持有一块持久 host-visible slab，所有激活按
    // float 偏移写入/读出，跨 step 复用 → RAM = 激活实际体积（避免每 tensor
    // 独立 128MB 块导致的碎片膨胀）。CPU 引擎 no-op（开启只会得到 1×1 张量）。
    [[nodiscard]] virtual Result<Tensor> create_offload_buffer(std::size_t /*bytes*/)
    {
        return Tensor::cpu(1, 1);
    }
    // 把 src 复制到 buffer 的 offset（float 单位）处
    [[nodiscard]] virtual Result<void> offload_save(
        const Tensor& /*buffer*/, std::size_t /*offset*/, const Tensor& /*src*/)
    {
        return {};
    }
    // 从 buffer 的 offset（float 单位）处复制 rows×cols 到新 GPU tensor
    [[nodiscard]] virtual Result<Tensor> offload_restore(
        const Tensor& /*buffer*/, std::size_t /*offset*/,
        std::size_t /*rows*/, std::size_t /*cols*/)
    {
        return Tensor::cpu(1, 1);
    }

    // ── 异步标量回读（P0-2：非阻塞取 loss）──────────────────────────────
    // 动机：若用 to_matrix 取每步 loss，GPU 引擎会 end_batch + wait_in_flight
    // （等全部在飞帧）→ 每 step 一次全流水线 drain，host/GPU 无法重叠。
    //
    // submit_scalar_readback(slot, t)：把 t（(1,1) **F32**）的头 4 字节排入
    //   一次 D2H 拷贝并提交，**不等待**。
    //   ⚠ 调用约定：必须在产出 t 的主帧已提交之后调用——同一队列 FIFO 保证
    //   拷贝执行在生产者之后（在主帧提交前提交会读到上一轮旧值）。
    //   ⚠ t 的宿主 Tensor 必须存活到 poll 返回就绪（buffer 生命周期跨越提交）。
    // poll_scalar_readback(slot, out)：非阻塞查询。就绪写值并返回 true；
    //   未就绪返回 false（调用方稍后重试）。
    // scalar_readback_slots()：可用槽位数（调用方据此做环形复用）。
    //
    // 默认实现（CPU / 其他同步引擎）：submit 立刻取标量存值，poll 恒立刻就绪。
    [[nodiscard]] virtual Result<void> submit_scalar_readback(
        std::size_t slot, const Tensor& t)
    {
        if (!t.is_cpu())
            return std::unexpected(Error{
                "submit_scalar_readback: 默认实现仅支持 CPU 张量"});
        Scalar v = Scalar{0};
        if (t.precision() == Precision::F32)
        {
            const auto& m = t.cpu_matrix();
            v = m.span()[0];
        }
        else if (t.precision() == Precision::F16)
        {
            const auto& m = t.cpu_matrix<Precision::F16>();
            v = static_cast<Scalar>(m.span()[0]);
        }
        else
        {
            return std::unexpected(Error{
                "submit_scalar_readback: 不支持的精度"});
        }
        if (sync_readback_slots_.size() <= slot)
            sync_readback_slots_.resize(slot + 1, Scalar{0});
        sync_readback_slots_[slot] = v;
        return {};
    }

    [[nodiscard]] virtual Result<bool> poll_scalar_readback(
        std::size_t slot, Scalar& out)
    {
        if (slot >= sync_readback_slots_.size())
            return false;
        out = sync_readback_slots_[slot];
        return true;
    }

    [[nodiscard]] virtual std::size_t scalar_readback_slots() const { return 1; }

    // ── 张量工厂（统一接口，§6.4, §6.5）────────────────────────────────
    // P 由调用方显式指定（§8.5）：无隐式推导，无 Auto
    [[nodiscard]] virtual Tensor create_tensor(std::size_t rows, std::size_t cols, Precision P = Precision::F32) = 0;
    [[nodiscard]] virtual Result<Tensor> from_matrix(const Matrix& m, Precision P = Precision::F32) = 0;
    [[nodiscard]] virtual Result<Matrix> to_matrix(const Tensor& t, Precision P = Precision::F32) = 0;

    // ── D5：cast 原语（§7.5，唯一"变精度"算子，永远显式）────────────────
    // 升 cast（f16→f32）精确无损；降 cast（f32→f16）round-half-to-even。
    // 默认实现：同精度 = 返回 src 拷贝；跨精度 = 错误（引擎覆盖）。
    [[nodiscard]] virtual Result<Tensor> cast(const Tensor& src, Precision dst)
    {
        if (src.precision() == dst)
            return src;  // 同精度 = 返回共享所有权（零拷贝）
        return std::unexpected(Error{"cast: 该引擎不支持跨精度转换"});
    }

    // ── 深拷贝 Tensor（CPU 矩阵拷贝 / GPU buffer 拷贝，无 PCIe 传输） ──
    // 用于需要修改中间结果但不影响原 Tensor 的场景
    [[nodiscard]] virtual Result<Tensor> clone(const Tensor& src) = 0;

    // 将 CPU Matrix 数据写入已有 Tensor（CPU 拷贝 / GPU 上传）
    // 用于序列化加载、Optimizer 参数写回等场景
    [[nodiscard]] virtual Result<void> copy_from(Tensor& dst, const Matrix& src) = 0;

    // ── 行切片原语（op-level 数据操作，不含算法语义） ──────────────────
    // 返回 src 的行 [start_row, start_row + count) 的连续拷贝。
    // 用于多头注意力中 per-head Q/K/V 切片等场景。
    [[nodiscard]] virtual Result<Tensor> slice_rows(
        const Tensor& src, std::size_t start_row, std::size_t count) = 0;

    // 将 src 的所有行写入 dst 的行 [dst_start_row, dst_start_row + src.rows())。
    // 真·就地修改（GPU 用 vkCmdCopyBuffer with dstOffset）。
    // 用于多头注意力中 per-head 输出拼接等场景。
    [[nodiscard]] virtual Result<void> insert_rows(
        Tensor& dst, std::size_t dst_start_row, const Tensor& src) = 0;

    // ── 行 gather / scatter-add 原语（op-level 数据操作） ────────────────
    // gather_rows: 按 indices 从 table 中按行查表，等价于 tf.gather / torch.index_select
    //   table: (vocab, D)
    //   indices: (num_indices,) — 行索引；越界索引返回零行（防御性，不抛错）
    //   输出: (num_indices, D)，out[i] = table[indices[i]]
    // 典型用途：Token embedding 查表（避免 Layer 内手动 to_matrix + at_unchecked）
    [[nodiscard]] virtual Result<Tensor> gather_rows(
        const Tensor& table, const Tensor& indices) = 0;

    // scatter_add_rows: 按 indices 把 grad 的行原子累加到 dst 的对应行
    //   dst: (vocab, D)，原地修改
    //   indices: (num_indices,)
    //   grad: (num_indices, D)
    //   语义: dst[indices[i]] += grad[i]  (重复 indices 会被多次累加)
    // 典型用途：Embedding 反向梯度按 token ID 累加（替代 Layer 内手动循环）
    [[nodiscard]] virtual Result<void> scatter_add_rows(
        Tensor& dst, const Tensor& indices, const Tensor& grad) = 0;

    // 3D 维度转置：(M, B, N) ↔ (B, M, N)
    //   inverse=false: 输入 (M, B*N) → 输出 (B*M, N)
    //     out[b*M + m, n] = in[m, b*N + n]
    //   inverse=true:  输入 (B*M, N) → 输出 (M, B*N)
    //     out[m, b*N + n] = in[b*M + m, n]
    // 典型用途：MHA 批量化时把 (H*d_k, batch*seq) 重排为 (batch*H*d_k, seq)，
    //   使 batched_matmul 能按 batch*H 切分行块。
    [[nodiscard]] virtual Result<Tensor> rearrange_3d(
        const Tensor& x, std::size_t M, std::size_t B, std::size_t N,
        bool inverse = false) = 0;
    // ── 矩阵转置：A (R, C) → out (C, R) ──
    // 纯 layout 操作，零算法语义。用于 embedding 列布局转换等场景。
    [[nodiscard]] virtual Result<Tensor> transpose(const Tensor& A) = 0;

    // ── 卷积/池化窗口展开原语（op-level 数据搬运，零算法语义）─────────────
    // im2col：把 (C, H, W) 输入按滑动窗展开成 GEMM 的列矩阵。
    //   x:   (C*H*W, B)
    //   out: (C*k*k, B*OH*OW)
    //   out[(ci*k + kh)*k + kw, (oh*OW + ow)*B + b]
    //       = x[ci*H*W + (oh*stride + kh - pad)*W + (ow*stride + kw - pad), b]
    //   越界（padding 区）取 0。OH/OW 由调用方按 (H + 2*pad - k)/stride + 1 给出。
    //   **列序为「位置优先」(oh, ow, b)**：这样 Layer 只需 rearrange_3d + gather_rows
    //   各一次即可完成 (C, B*P) → (C*P, B) 的 samples 布局转换。
    // 用途：Conv2D 把卷积变成 matmul（W × col）；MaxPool2D 展开窗口后做列归约
    //   （取 k=pool, stride=stride, pad=0）。
    [[nodiscard]] virtual Result<Tensor> im2col(
        const Tensor& x,
        std::size_t C, std::size_t H, std::size_t W,
        std::size_t k, std::size_t stride, std::size_t pad,
        std::size_t OH, std::size_t OW) = 0;

    // col2im：im2col 的伴随（adjoint / 反向散射）。
    //   col: (C*k*k, B*OH*OW) → out: (C*H*W, B)
    //   out[ci*H*W + ih*W + iw, b]
    //       = Σ_{kh,kw} col[(ci*k + kh)*k + kw, (oh*OW + ow)*B + b]
    //   其中 oh = (ih + pad - kh)/stride（须整除且落在 [0, OH)），ow 同理。
    //   **重叠窗口（stride < k）的贡献在此累加**；每个输出元素由单个线程/循环
    //   完整求和，故无需原子操作/预清零。
    [[nodiscard]] virtual Result<Tensor> col2im(
        const Tensor& col,
        std::size_t C, std::size_t H, std::size_t W,
        std::size_t k, std::size_t stride, std::size_t pad,
        std::size_t OH, std::size_t OW) = 0;

    // ══════════════════════════════════════════════════════════════════════
    // 矩阵级原语
    // ══════════════════════════════════════════════════════════════════════

    // C = A × B（支持转置标志 + 精度参数，D5 §8.1）
    // transA: 使用 A^T，transB: 使用 B^T
    // P: 计算精度（F32 = 现状；F16 = f16 GEMM，§7.4）
    //   - Auto 推导（P=F32 默认）：操作数中最高精度
    //   - 显式 P：Layer 通过 PrecisionProfile.compute 传入
    [[nodiscard]] virtual Result<Tensor> matmul(
        const Tensor& A, const Tensor& B,
        bool transA = false, bool transB = false,
        Precision P = Precision::F32) = 0;

    // 批量矩阵乘法：对每个 batch b 计算 C_b = alpha * op(A_b, B_b)，结果垂直堆叠
    // A: (batch * A_rows_per_batch, A_cols) — 按 batch 切分为连续行块
    // B: (batch * B_rows_per_batch, B_cols)
    // 输出: (batch * M, N)，M/N 为每个 batch 的逻辑输出维度
    //   transA=0: A_b 为 (M, K)，transA=1: A_b 存储为 (K, M) 按 A_b^T 使用
    //   transB=0: B_b 为 (K, N)，transB=1: B_b 存储为 (N, K) 按 B_b^T 使用
    // alpha: 输出缩放系数（cuBLAS sgemm 语义），GPU 在 shader 写出时一次完成，
    //   供上层折叠 1/sqrt(d_k) 等系数，省去额外全矩阵 scale pass
    // P: 计算精度（D5 §8.1，同 matmul）
    // 典型用途：多头注意力的 Q^T×K 和 V×A 批量化（消除 per-head 循环）
    [[nodiscard]] virtual Result<Tensor> batched_matmul(
        const Tensor& A, const Tensor& B,
        std::size_t batch,
        bool transA = false, bool transB = false,
        Scalar alpha = Scalar{1},
        Precision P = Precision::F32) = 0;

    // ── 新增：matmul + broadcast bias（统一精度处理）────────────────────
    // out = A × B + bias（broadcast add，bias (out,1) → (out,batch)）
    // 引擎内部处理精度：P 指定 matmul 精度，bias add 在 P 精度下完成
    // Layer 无需判断精度，直接调用即可
    [[nodiscard]] virtual Result<Tensor> matmul_with_bias(
        const Tensor& A, const Tensor& B, const Tensor& bias,
        bool transA = false, bool transB = false,
        Precision P = Precision::F32)
    {
        // 默认实现：matmul + 逐行 add bias（兼容所有引擎）
        auto result = matmul(A, B, transA, transB, P);
        if (!result) return std::unexpected(result.error());

        // broadcast bias: (out,1) → (out,batch)
        // 用 engine.add_inplace 逐行加
        auto bias_mat = to_matrix(bias, P);
        if (!bias_mat) return std::unexpected(bias_mat.error());

        auto res_mat = to_matrix(*result, P);
        if (!res_mat) return std::unexpected(res_mat.error());

        for (std::size_t row = 0; row < A.rows(); ++row)
        {
            float b_val = bias_mat->at(row, 0);
            for (std::size_t col = 0; col < B.cols(); ++col)
                res_mat->set_value(row, col, res_mat->at(row, col) + b_val);
        }

        return from_matrix(*res_mat, P);
    }

    // ── 新增：梯度累加（自动 cast 到目标精度）────────────────────────────
    // dst += src（dst 始终 f32，src 可能是 f16）
    // 引擎内部 cast src 到 f32 后累加
    [[nodiscard]] virtual Result<void> accumulate(Tensor& dst, const Tensor& src)
    {
        if (dst.precision() != Precision::F32)
            return std::unexpected(Error{"accumulate: dst must be F32"});

        if (src.precision() == Precision::F32)
            return add_inplace(dst, src);

        // src 是 f16 → cast 到 f32 后累加
        auto src32 = cast(src, Precision::F32);
        if (!src32) return std::unexpected(src32.error());
        return add_inplace(dst, *src32);
    }

    // A += B（逐元素，同形状）
    [[nodiscard]] virtual Result<void> add_inplace(Tensor& A, const Tensor& B) = 0;

    // A *= scalar
    [[nodiscard]] virtual Result<void> scale_inplace(Tensor& A, Scalar s) = 0;

    // A += scalar * B（融合 axpy：单次 dispatch 替代 clone+scale+add 三步）
    // 用于 Optimizer 的 scale_add_ 辅助方法，减少 2 个临时 GPU buffer 分配
    [[nodiscard]] virtual Result<void> axpy_inplace(Tensor& A, Scalar scalar, const Tensor& B) = 0;

    // A = 0
    [[nodiscard]] virtual Result<void> zero(Tensor& A) = 0;

    // ══════════════════════════════════════════════════════════════════════
    // 扫描级原语（带状态的顺序归约 + matvec 读出；RLA 线性注意力积木）
    //
    // 引擎只提供"前缀/后缀顺序归约 + matvec 读出"；RLA 算法（L2 归一化
    // 分母 / ReLU 门控 / 梯度公式 / 文档重置策略）全部由 Layer 用这些原语
    // 与逐元素原语组合表达（铁律 3：shader 永不含算法）。
    //
    // 形状约定（batch-major，列序 i = b*seq + t；头 (b,h) 的行块起点
    // r0 = (b*H + h)*d_k，每头 d_k 行）：
    //   K/V/P/R（X/Y）: (B·H·d_k, seq)
    //   D             : (B·H·d_k², seq)，(b,h) 的 (a,b') 元素在
    //                    行 (b*H*d_k + a)*d_k + b'
    //   A0/B0         : (H·d_k, d_k) 初始运行态，行块 h = 第 h 头
    //                    （B>1 时按头循环）；has_state=false → 按零
    //                    处理（传 (1,1) dummy，规避 0 字节 buffer）
    //   boundary      : (1, B·seq)，1 = 文档起点（t==0 或与前一位置
    //                    文档不同）；has_bnd=false → 无文档感知
    //                    （传 (1,1) dummy）
    //   标量块（s/r）: 每 (b,h,t) 一个标量，在头块内 d_k 行重复存放
    //                    （避免块级广播原语）；实现写全部 d_k 行
    //                    的同一值，Layer 读任一行均可。
    // ══════════════════════════════════════════════════════════════════════

    // 前缀扫描：
    //   causal=true : 含自身前缀（i<=t）：A_t = A0 + Σ_{i≤t, 与 t 同文档}
    //                 k_i·k_i^T，B_t = B0 + Σ_{i≤t, 同文档} v_i·k_i^T；
    //                 文档边界处运行态清零（A0/B0 仅首个文档生效）。
    //   causal=false: 全集常数 A = A0 + Σ_all k·k^T，B = B0 + Σ_all v·k^T
    //                 （无边界重置）。
    // 输出 (B·H·5·d_k, seq)，行块（每块 (B·H·d_k, seq)）：
    //   [0) B·P   [1) A·P   [2) B^T·R   [3) s = P·(A·P)   [4) r = R·(B·P)
    //   其中 [3)/[4) 为逐列标量（头内逐行重复）。
    [[nodiscard]] virtual Result<Tensor> scan_prefix_outer(
        const Tensor& K, const Tensor& V, const Tensor& P, const Tensor& R,
        const Tensor& A0, const Tensor& B0, bool has_state,
        std::size_t dk, std::size_t heads, bool causal,
        const Tensor& boundary, bool has_bnd) = 0;

    // 后缀扫描（RLA 反向 pass 2）：
    //   causal=true : S_i = Σ_{t≥i, 与 i 同文档} D_t（i+1 为文档起点时
    //                 先清零再累加 D_i）；
    //   causal=false: S_i = D_i（Layer 预先把全集梯度沿 seq 广播）。
    // D (B·H·d_k², seq)，X/Y (B·H·d_k, seq)
    // 输出 (B·H·3·d_k, seq)，行块：[0) S·X   [1) S·Y   [2) S^T·Y
    [[nodiscard]] virtual Result<Tensor> scan_suffix_outer(
        const Tensor& D, const Tensor& X, const Tensor& Y,
        std::size_t dk, std::size_t heads, bool causal,
        const Tensor& boundary, bool has_bnd) = 0;

    // 逐列外积（RLA 反向的 dL/dA、dL/dB 物化）：
    //   out[(b,h): (a,b'), t] = P[a,t]·R[b',t] (· S[t] if has_scale)
    // P/R: (B·H·d_k, seq)；S: (B·H·d_k, seq)（标量头内逐行重复，
    // 实现读头块首行；has_scale=false → 传 dummy）
    // 输出: (B·H·d_k², seq)
    [[nodiscard]] virtual Result<Tensor> outer_col(
        const Tensor& P, const Tensor& R, const Tensor& S,
        std::size_t dk, bool has_scale) = 0;

    // ══════════════════════════════════════════════════════════════════════
    // 归约原语
    // ══════════════════════════════════════════════════════════════════════

    // 按行求和：A (rows, cols) → out (rows, 1)
    // out[r] = Σ_c A[r][c]
    [[nodiscard]] virtual Result<Tensor> row_reduce_sum(const Tensor& A) = 0;

    // 按列求和：A (rows, cols) → out (1, cols)
    // out[c] = Σ_r A[r][c]
    [[nodiscard]] virtual Result<Tensor> col_reduce_sum(const Tensor& A) = 0;

    // 按行求最大值：A (rows, cols) → out (rows, 1)
    // out[r] = max_c A[r][c]
    [[nodiscard]] virtual Result<Tensor> row_reduce_max(const Tensor& A) = 0;

    // 按列求最大值：A (rows, cols) → out (1, cols)
    // out[c] = max_r A[r][c]
    [[nodiscard]] virtual Result<Tensor> col_reduce_max(const Tensor& A) = 0;

    // ── 分组归约（segmented reduce，沿行方向按固定长度分组）───────────────
    // 与 row/col_reduce 的区别：归约轴不是"整行/整列"，而是**每连续 R 行为一组**。
    //   x: (G*R, N) → out: (G, N)
    //   grouped_reduce_sum: out[g, n] = Σ_{i<R} x[g*R + i, n]
    //   grouped_reduce_max: out[g, n] = max_i  x[g*R + i, n]
    // 用途：把"逐通道/逐头一次 dispatch"的层内循环压成**单次**原语调用
    //   （池化窗口归约、多头分组统计、分组归一化等）。
    [[nodiscard]] virtual Result<Tensor> grouped_reduce_sum(
        const Tensor& x, std::size_t G, std::size_t R) = 0;
    [[nodiscard]] virtual Result<Tensor> grouped_reduce_max(
        const Tensor& x, std::size_t G, std::size_t R) = 0;

    // ══════════════════════════════════════════════════════════════════════
    // 广播原语
    // ══════════════════════════════════════════════════════════════════════

    // 按行广播：A (R, C) op= row_vec (R, 1)
    // A[r][c] = op(A[r][c], row_vec[r][0])
    [[nodiscard]] virtual Result<void> broadcast_row_inplace(
        Tensor& A, const Tensor& row_vec, BinaryOp op) = 0;

    // 按列广播：A (R, C) op= col_vec (1, C)
    // A[r][c] = op(A[r][c], col_vec[0][c])
    [[nodiscard]] virtual Result<void> broadcast_col_inplace(
        Tensor& A, const Tensor& col_vec, BinaryOp op) = 0;

    // ══════════════════════════════════════════════════════════════════════
    // 逐元素原语
    // ══════════════════════════════════════════════════════════════════════

    // out = unary_op(A)
    [[nodiscard]] virtual Result<Tensor> elementwise_unary(
        UnaryOp op, const Tensor& A) = 0;

    // out = binary_op(A, B)
    [[nodiscard]] virtual Result<Tensor> elementwise_binary(
        BinaryOp op, const Tensor& A, const Tensor& B) = 0;

    // out = binary_op(A, scalar) 或 binary_op(scalar, A)
    // scalar_first=true: out = op(scalar, A)；false: out = op(A, scalar)
    [[nodiscard]] virtual Result<Tensor> elementwise_binary_scalar(
        BinaryOp op, const Tensor& A, Scalar s, bool scalar_first = false) = 0;

    // ══════════════════════════════════════════════════════════════════════
    // 条件选择原语
    // ══════════════════════════════════════════════════════════════════════

    // out = compare_op(A, scalar_b) ? then_t : scalar_else
    // 条件操作数 2 为标量，then 为张量，else 为标量。
    // 典型用途：ReLU 反向 (x > 0) ? grad : 0
    [[nodiscard]] virtual Result<Tensor> elementwise_select_scalar_cond(
        CompareOp cmp, const Tensor& A, Scalar scalar_b,
        const Tensor& then_t, Scalar scalar_else) = 0;

    // ══════════════════════════════════════════════════════════════════════
    // 表达式求值（逐元素融合的统一入口）
    // ══════════════════════════════════════════════════════════════════════

    // 对一个逐元素表达式求值，输出 (rows, cols)。所有输入同形状。
    //
    // 这是"函数式逐元素原语"的表达式升级：单行内多次计算（如 RoPE 的
    // q*cos + rotate(q)*sin、残差、激活）可合并为一次调用，减少临时 Tensor。
    // 执行策略由后端决定（CPU 编译期模板求值；Vulkan AOT 融合 shader，闭合世界），
    // Layer 侧无需关心——表达式是唯一逐元素编程模型。
    //
    // 语义与上限见 expr_spec.hpp（ExprSpec）。输出 = 最后一条指令的目标寄存器。
    [[nodiscard]] virtual Result<Tensor> eval_expr(
        const ExprSpec& spec,
        std::span<const Tensor> inputs,
        std::size_t rows, std::size_t cols) = 0;

    // ── 归约向量原生形状输出（LayerNorm/RMSNorm 小向量缓存等） ──────────
    // 语义同 eval_expr，但输出为归约向量本身（非广播）：
    //   行归约轴 → (rows,1)；列归约轴 → (1,cols)。
    // 要求表达式归约轴为 0/1（expr_spec_reduce_axis）。
    // 默认实现返回错误（未支持的引擎）；CPU/GPU 覆盖。
    [[nodiscard]] virtual Result<Tensor> eval_expr_reduce(
        const ExprSpec& spec,
        std::span<const Tensor> inputs,
        std::size_t rows, std::size_t cols)
    {
        (void)spec; (void)inputs; (void)rows; (void)cols;
        return std::unexpected(Error{"eval_expr_reduce: 该引擎不支持归约向量输出"});
    }

    // ── 目标传递（destination-passing）：结果写入已有张量 ────────────────
    // 语义同 eval_expr，但输出**直接写进 out**（不分配新张量），用于表达
    // "原地更新"语义：dst = f(dst, ...)（梯度累加 / 参数更新 / 就地缩放 /
    // 就地按行广播等）。out 允许与某个输入是同一 buffer：逐元素"先读完全部
    // 输入再写 out[i]"，就地安全。
    // 仅支持逐元素表达式（无归约）；归约向量输出用 eval_expr_reduce。
    // 默认实现返回错误（未支持的引擎）；CPU/GPU 覆盖为真原地实现。
    [[nodiscard]] virtual Result<void> eval_expr_into(
        const ExprSpec& spec,
        std::span<const Tensor> inputs,
        std::size_t rows, std::size_t cols, Tensor& out)
    {
        (void)spec; (void)inputs; (void)rows; (void)cols; (void)out;
        return std::unexpected(Error{
            "eval_expr_into: 该引擎不支持原地表达式求值"});
    }

private:
    // 默认（CPU / 同步引擎）标量回读槽：submit 立即存值，poll 立即就绪。
    // GPU 引擎覆写为异步槽位（见 GpuEngine / GpuBackend::rb_slots_）。
    std::vector<Scalar> sync_readback_slots_;
};

} // namespace nn

