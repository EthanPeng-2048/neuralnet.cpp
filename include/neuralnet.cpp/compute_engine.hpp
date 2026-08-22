#ifndef NN_COMPUTE_ENGINE_HPP
#define NN_COMPUTE_ENGINE_HPP

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

#include "config.hpp"
#include "core_errors.hpp"
#include "tensor.hpp"
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

    // ── 表达式录制（计算级融合，M2 框架） ───────────────────────────────
    // begin_expr 进入录制；期间 Layer 调 eval_expr / 组合原语；
    // end_expr 时引擎做融合分析，将可融合子序列合成单 kernel。
    //
    // M2 现状（地基）：
    //   - CPU 引擎：no-op（各表达式直接求值，行为不变）。
    //   - GPU 引擎：no-op（各原语正常 dispatch；录制融合分析在 M3 落地，
    //     届时 begin/end 之间的小中间量/逐元素链并入单 kernel）。
    // Layer 可先行用 begin_expr/end_expr 包住算法段落，语义不变。
    [[nodiscard]] virtual Result<void> begin_expr() = 0;
    [[nodiscard]] virtual Result<void> end_expr() = 0;

    // ── 批处理中点刷新（防 TDR） ─────────────────────────────────────────
    // GPU 引擎：提交当前 command buffer 并等待完成，然后自动开始新的录制。
    // 可在 forward 与 backward 之间调用，将一次大提交拆分为多次小提交，
    // 避免单次提交时间过长触发 Windows TDR。
    // CPU 引擎：no-op。
    [[nodiscard]] virtual Result<void> flush_batch() { return {}; }

    // ── 张量工厂 ──────────────────────────────────────────────────────────
    [[nodiscard]] virtual Tensor create_tensor(std::size_t rows, std::size_t cols) = 0;
    [[nodiscard]] virtual Result<Tensor> from_matrix(const Matrix& m) = 0;
    [[nodiscard]] virtual Result<Matrix> to_matrix(const Tensor& t) = 0;

    // 将 CPU Matrix 数据写入已有 Tensor（CPU 拷贝 / GPU 上传）
    // 用于序列化加载、Optimizer 参数写回等场景
    [[nodiscard]] virtual Result<void> copy_from(Tensor& dst, const Matrix& src) = 0;

    // 深拷贝 Tensor（CPU 矩阵拷贝 / GPU buffer 拷贝，无 PCIe 传输）
    // 用于需要修改中间结果但不影响原 Tensor 的场景
    [[nodiscard]] virtual Result<Tensor> clone(const Tensor& src) = 0;

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
    // ══════════════════════════════════════════════════════════════════════
    // 矩阵级原语
    // ══════════════════════════════════════════════════════════════════════

    // C = A × B（支持转置标志）
    // transA: 使用 A^T，transB: 使用 B^T
    [[nodiscard]] virtual Result<Tensor> matmul(
        const Tensor& A, const Tensor& B,
        bool transA = false, bool transB = false) = 0;

    // 批量矩阵乘法：对每个 batch b 计算 C_b = alpha * op(A_b, B_b)，结果垂直堆叠
    // A: (batch * A_rows_per_batch, A_cols) — 按 batch 切分为连续行块
    // B: (batch * B_rows_per_batch, B_cols)
    // 输出: (batch * M, N)，M/N 为每个 batch 的逻辑输出维度
    //   transA=0: A_b 为 (M, K)，transA=1: A_b 存储为 (K, M) 按 A_b^T 使用
    //   transB=0: B_b 为 (K, N)，transB=1: B_b 存储为 (N, K) 按 B_b^T 使用
    // alpha: 输出缩放系数（cuBLAS sgemm 语义），GPU 在 shader 写出时一次完成，
    //   供上层折叠 1/sqrt(d_k) 等系数，省去额外全矩阵 scale pass
    // 典型用途：多头注意力的 Q^T×K 和 V×A 批量化（消除 per-head 循环）
    [[nodiscard]] virtual Result<Tensor> batched_matmul(
        const Tensor& A, const Tensor& B,
        std::size_t batch,
        bool transA = false, bool transB = false,
        Scalar alpha = Scalar{1}) = 0;

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
};

} // namespace nn

#endif // NN_COMPUTE_ENGINE_HPP
