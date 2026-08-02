// ── cuda_kernels.h — CUDA 内核 C-linkage 接口 ─────────────────────────────
// 定义所有 CUDA 计算内核的函数签名。
// 实现在 cuda_kernels.cu 中，由 nvcc 编译为静态库。
// 主项目通过 CudaBackend 调用这些函数（C ABI，跨编译器链接安全）。
//
// 所有函数使用 float 类型（与 Scalar = float 一致）。
// 错误通过返回 int 传递：0 = 成功，非 0 = cudaError_t。
// ─────────────────────────────────────────────────────────────────────────

#ifndef NN_CUDA_KERNELS_H
#define NN_CUDA_KERNELS_H

#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

// ══════════════════════════════════════════════════════════════════════════
// 矩阵乘法
// ══════════════════════════════════════════════════════════════════════════

// C = A × B（支持转置标志）
// A: (A_rows, A_cols), B: (B_rows, B_cols), C: (M, N)
// transA: 使用 A^T, transB: 使用 B^T
// stream: CUDA 流（0 = 默认流）
int cuda_matmul(
    const float* A, const float* B, float* C,
    std::size_t A_rows, std::size_t A_cols,
    std::size_t B_rows, std::size_t B_cols,
    int transA, int transB,
    void* stream);

// 带预创建 cuBLAS 句柄的 matmul（避免每次调用创建/销毁句柄的开销）
// handle_ptr: cuBLAS 句柄指针（由 CudaBackend 创建并复用），nullptr 时退化为 cuda_matmul
// 注：当未启用 NN_HAS_CUBLAS 时，本函数退化为自定义内核（忽略 handle_ptr）
int cuda_matmul_with_handle(
    const float* A, const float* B, float* C,
    std::size_t A_rows, std::size_t A_cols,
    std::size_t B_rows, std::size_t B_cols,
    int transA, int transB,
    void* handle_ptr);

// 批量矩阵乘法：对每个 batch 计算 C_b = op(A_b, B_b)，结果垂直堆叠
// A: (batch * a_rows_per, A_cols), B: (batch * b_rows_per, B_cols)
// C: (batch * M, N)
int cuda_batched_matmul(
    const float* A, const float* B, float* C,
    std::size_t A_rows, std::size_t A_cols,
    std::size_t B_rows, std::size_t B_cols,
    unsigned int batch, int transA, int transB,
    void* stream);

// ══════════════════════════════════════════════════════════════════════════
// 逐元素运算
// ══════════════════════════════════════════════════════════════════════════

// 一元运算：out = op(A)
// op: 0=Neg, 1=Exp, 2=Log, 3=Sqrt, 4=Rsqrt, 5=Abs, 6=Tanh
int cuda_elementwise_unary(
    const float* A, float* out,
    unsigned int count, unsigned int op,
    void* stream);

// 二元运算：out = op(A, B)
// op: 0=Add, 1=Sub, 2=Mul, 3=Div, 4=Max, 5=Min
int cuda_elementwise_binary(
    const float* A, const float* B, float* out,
    unsigned int count, unsigned int op,
    void* stream);

// 标量二元运算：out = op(A, scalar) 或 op(scalar, A)
// scalar_first: true → op(scalar, A), false → op(A, scalar)
int cuda_elementwise_binary_scalar(
    const float* A, float* out,
    unsigned int count, unsigned int op,
    float scalar, int scalar_first,
    void* stream);

// 融合 axpy：out = A + scalar * B
int cuda_axpy(
    const float* A, const float* B, float* out,
    unsigned int count, float scalar,
    void* stream);

// 条件选择：out = cmp(A, scalar_b) ? then_v : scalar_else
// cmp: 0=Lt, 1=Le, 2=Gt, 3=Ge, 4=Eq, 5=Ne
int cuda_elementwise_select_scalar_cond(
    const float* A, const float* then_v, float* out,
    unsigned int count, unsigned int cmp,
    float scalar_b, float scalar_else,
    void* stream);

// ══════════════════════════════════════════════════════════════════════════
// 归约
// ══════════════════════════════════════════════════════════════════════════

// 行归约：A (rows, cols) → out (rows, 1)
// mode: 0=sum, 1=max
int cuda_reduce_row(
    const float* A, float* out,
    unsigned int rows, unsigned int cols, unsigned int mode,
    void* stream);

// 列归约：A (rows, cols) → out (1, cols)
// mode: 0=sum, 1=max
int cuda_reduce_col(
    const float* A, float* out,
    unsigned int rows, unsigned int cols, unsigned int mode,
    void* stream);

// ══════════════════════════════════════════════════════════════════════════
// 广播
// ══════════════════════════════════════════════════════════════════════════

// 行广播：out[r][c] = op(A[r][c], vec[r])
// op: 0=Add, 1=Sub, 2=Mul, 3=Div, 4=Max, 5=Min
int cuda_broadcast_row(
    const float* A, const float* vec, float* out,
    unsigned int rows, unsigned int cols, unsigned int op,
    void* stream);

// 列广播：out[r][c] = op(A[r][c], vec[c])
int cuda_broadcast_col(
    const float* A, const float* vec, float* out,
    unsigned int rows, unsigned int cols, unsigned int op,
    void* stream);

// ══════════════════════════════════════════════════════════════════════════
// 转置与重排
// ══════════════════════════════════════════════════════════════════════════

// 矩阵转置：A (R, C) → out (C, R)
int cuda_transpose(
    const float* A, float* out,
    unsigned int R, unsigned int C,
    void* stream);

// 3D 维度重排：(M, B, N) ↔ (B, M, N)
// inverse=0: (M, B*N) → (B*M, N)
// inverse=1: (B*M, N) → (M, B*N)
int cuda_rearrange_3d(
    const float* A, float* out,
    unsigned int M, unsigned int B, unsigned int N,
    unsigned int inverse,
    void* stream);

// ══════════════════════════════════════════════════════════════════════════
// Gather / Scatter
// ══════════════════════════════════════════════════════════════════════════

// 按行索引查表：out[i] = table[indices[i]] (i < num)
// table: (vocab, D), indices: (num,), out: (num, D)
// 越界索引返回零行
int cuda_gather(
    const float* table, const float* indices, float* out,
    unsigned int vocab, unsigned int D, unsigned int num,
    void* stream);

// 按行索引原子累加：dst[indices[i]] += grad[i]
// dst: (vocab, D) 原地修改, indices: (num,), grad: (num, D)
int cuda_scatter_add(
    float* dst, const float* indices, const float* grad,
    unsigned int vocab, unsigned int D, unsigned int num,
    void* stream);

#ifdef __cplusplus
}
#endif

#endif // NN_CUDA_KERNELS_H
