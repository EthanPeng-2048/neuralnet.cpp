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
//   - 矩阵级：matmul, transpose, add_inplace, scale_inplace, zero
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

#include "config.hpp"
#include "core_errors.hpp"
#include "tensor.hpp"

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

    // ══════════════════════════════════════════════════════════════════════
    // 矩阵级原语
    // ══════════════════════════════════════════════════════════════════════

    // C = A × B（支持转置标志）
    // transA: 使用 A^T，transB: 使用 B^T
    [[nodiscard]] virtual Result<Tensor> matmul(
        const Tensor& A, const Tensor& B,
        bool transA = false, bool transB = false) = 0;

    // A += B（逐元素，同形状）
    [[nodiscard]] virtual Result<void> add_inplace(Tensor& A, const Tensor& B) = 0;

    // A *= scalar
    [[nodiscard]] virtual Result<void> scale_inplace(Tensor& A, Scalar s) = 0;

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
};

} // namespace nn

#endif // NN_COMPUTE_ENGINE_HPP
