// ── cuda_kernels.cu — CUDA 计算内核实现 ─────────────────────────────────────
// 由 nvcc 编译为静态库，通过 C-linkage 接口供 CudaBackend 调用。
//
// 内核设计原则：
//   1. 与 Vulkan compute shader 语义完全一致
//   2. 使用 float 类型（Scalar = float）
//   3. 所有内核接受 CUDA stream 参数以支持异步执行
//   4. 使用 cuBLAS 加速 matmul（通过 NN_HAS_CUBLAS 宏控制）
//   5. cuBLAS 句柄由调用方（CudaBackend）管理并复用，避免每次调用创建/销毁
// ─────────────────────────────────────────────────────────────────────────

#include "cuda_kernels.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cfloat>

#ifdef NN_HAS_CUBLAS
#include <cublas_v2.h>
#endif

// ── CUDA 错误检查宏 ──────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            return static_cast<int>(err);                                      \
        }                                                                       \
    } while (0)

#define CUDA_CHECK_VOID(call)                                                  \
    do {                                                                        \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            return;                                                            \
        }                                                                       \
    } while (0)

#ifdef NN_HAS_CUBLAS
// cuBLAS 错误检查宏（将 cublasStatus_t 转为负数返回值）
#define CUBLAS_CHECK(call)                                                     \
    do {                                                                        \
        cublasStatus_t st = (call);                                            \
        if (st != CUBLAS_STATUS_SUCCESS) {                                     \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n",                     \
                    __FILE__, __LINE__, static_cast<int>(st));                 \
            return static_cast<int>(-st);                                      \
        }                                                                       \
    } while (0)
#endif

// ══════════════════════════════════════════════════════════════════════════
// 矩阵乘法内核
// ══════════════════════════════════════════════════════════════════════════

static constexpr int TILE = 16;

__global__ void matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    int transA, int transB)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t)
    {
        const int a_col = t * TILE + threadIdx.x;
        const int b_row = t * TILE + threadIdx.y;

        if (row < M && a_col < K)
            As[threadIdx.y][threadIdx.x] = transA ? A[a_col * M + row] : A[row * K + a_col];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (b_row < K && col < N)
            Bs[threadIdx.y][threadIdx.x] = transB ? B[col * K + b_row] : B[b_row * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE; ++k)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

// ══════════════════════════════════════════════════════════════════════════
// 批量矩阵乘法内核
// ══════════════════════════════════════════════════════════════════════════

__global__ void batched_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    int transA, int transB,
    int batch)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int b = blockIdx.z;
    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;

    const float* Ab = A + static_cast<std::size_t>(b) * M * K;
    const float* Bb = B + static_cast<std::size_t>(b) * K * N;
    float* Cb = C + static_cast<std::size_t>(b) * M * N;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t)
    {
        const int a_col = t * TILE + threadIdx.x;
        const int b_row = t * TILE + threadIdx.y;

        if (row < M && a_col < K)
            As[threadIdx.y][threadIdx.x] = transA ? Ab[a_col * M + row] : Ab[row * K + a_col];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (b_row < K && col < N)
            Bs[threadIdx.y][threadIdx.x] = transB ? Bb[col * K + b_row] : Bb[b_row * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE; ++k)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        Cb[row * N + col] = sum;
}

// ══════════════════════════════════════════════════════════════════════════
// 逐元素运算内核
// ══════════════════════════════════════════════════════════════════════════

__global__ void elementwise_unary_kernel(
    const float* __restrict__ A, float* __restrict__ out,
    unsigned int count, unsigned int op)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    const float x = A[i];
    switch (op) {
        case 0: out[i] = -x;          break; // Neg
        case 1: out[i] = expf(x);     break; // Exp
        case 2: out[i] = logf(x);     break; // Log
        case 3: out[i] = sqrtf(x);    break; // Sqrt
        case 4: out[i] = rsqrtf(x);   break; // Rsqrt
        case 5: out[i] = fabsf(x);    break; // Abs
        case 6: out[i] = tanhf(x);    break; // Tanh
        default: out[i] = x;          break;
    }
}

__global__ void elementwise_binary_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ out,
    unsigned int count, unsigned int op)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    const float a = A[i];
    const float b = B[i];
    switch (op) {
        case 0: out[i] = a + b;               break; // Add
        case 1: out[i] = a - b;               break; // Sub
        case 2: out[i] = a * b;               break; // Mul
        case 3: out[i] = a / b;               break; // Div
        case 4: out[i] = fmaxf(a, b);         break; // Max
        case 5: out[i] = fminf(a, b);         break; // Min
        default: out[i] = a;                  break;
    }
}

__global__ void elementwise_binary_scalar_kernel(
    const float* __restrict__ A, float* __restrict__ out,
    unsigned int count, unsigned int op,
    float scalar, int scalar_first)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    const float a = A[i];
    const float s = scalar;
    if (scalar_first) {
        switch (op) {
            case 0: out[i] = s + a;       break;
            case 1: out[i] = s - a;       break;
            case 2: out[i] = s * a;       break;
            case 3: out[i] = s / a;       break;
            case 4: out[i] = fmaxf(s, a); break;
            case 5: out[i] = fminf(s, a); break;
            default: out[i] = a;          break;
        }
    } else {
        switch (op) {
            case 0: out[i] = a + s;       break;
            case 1: out[i] = a - s;       break;
            case 2: out[i] = a * s;       break;
            case 3: out[i] = a / s;       break;
            case 4: out[i] = fmaxf(a, s); break;
            case 5: out[i] = fminf(a, s); break;
            default: out[i] = a;          break;
        }
    }
}

__global__ void axpy_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ out,
    unsigned int count, float scalar)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    out[i] = A[i] + scalar * B[i];
}

__global__ void select_scalar_cond_kernel(
    const float* __restrict__ A,
    const float* __restrict__ then_v,
    float* __restrict__ out,
    unsigned int count, unsigned int cmp,
    float scalar_b, float scalar_else)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    const float a = A[i];
    bool cond;
    switch (cmp) {
        case 0: cond = a <  scalar_b; break; // Lt
        case 1: cond = a <= scalar_b; break; // Le
        case 2: cond = a >  scalar_b; break; // Gt
        case 3: cond = a >= scalar_b; break; // Ge
        case 4: cond = a == scalar_b; break; // Eq
        case 5: cond = a != scalar_b; break; // Ne
        default: cond = false;        break;
    }
    out[i] = cond ? then_v[i] : scalar_else;
}

// ══════════════════════════════════════════════════════════════════════════
// 归约内核（每个 block 归约一行/列）
// ══════════════════════════════════════════════════════════════════════════

__global__ void reduce_row_sum_kernel(
    const float* __restrict__ A, float* __restrict__ out,
    unsigned int rows, unsigned int cols)
{
    const unsigned int r = blockIdx.x;
    if (r >= rows) return;

    extern __shared__ float sdata[];
    float sum = 0.0f;
    for (unsigned int c = threadIdx.x; c < cols; c += blockDim.x)
        sum += A[r * cols + c];

    sdata[threadIdx.x] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        out[r] = sdata[0];
}

__global__ void reduce_row_max_kernel(
    const float* __restrict__ A, float* __restrict__ out,
    unsigned int rows, unsigned int cols)
{
    const unsigned int r = blockIdx.x;
    if (r >= rows) return;

    extern __shared__ float sdata[];
    float val = -FLT_MAX;
    for (unsigned int c = threadIdx.x; c < cols; c += blockDim.x)
        val = fmaxf(val, A[r * cols + c]);

    sdata[threadIdx.x] = val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }

    if (threadIdx.x == 0)
        out[r] = sdata[0];
}

// 列归约（每个 block 处理一列，shared memory tree reduction）
// 与 reduce_row_*_kernel 对称：blockIdx.x = c（列号），线程协作归约所有行。
// 行主序矩阵 A[r * cols + c]：同一列的元素在内存中 stride=cols，
// 不连续访问，但通过线程级并行 + shared memory 树形归约仍优于单线程串行。
__global__ void reduce_col_sum_kernel(
    const float* __restrict__ A, float* __restrict__ out,
    unsigned int rows, unsigned int cols)
{
    const unsigned int c = blockIdx.x;
    if (c >= cols) return;

    extern __shared__ float sdata[];
    float sum = 0.0f;
    for (unsigned int r = threadIdx.x; r < rows; r += blockDim.x)
        sum += A[r * cols + c];

    sdata[threadIdx.x] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        out[c] = sdata[0];
}

__global__ void reduce_col_max_kernel(
    const float* __restrict__ A, float* __restrict__ out,
    unsigned int rows, unsigned int cols)
{
    const unsigned int c = blockIdx.x;
    if (c >= cols) return;

    extern __shared__ float sdata[];
    float val = -FLT_MAX;
    for (unsigned int r = threadIdx.x; r < rows; r += blockDim.x)
        val = fmaxf(val, A[r * cols + c]);

    sdata[threadIdx.x] = val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }

    if (threadIdx.x == 0)
        out[c] = sdata[0];
}

// ══════════════════════════════════════════════════════════════════════════
// 广播内核
// ══════════════════════════════════════════════════════════════════════════

__device__ inline float apply_binary_op(float a, float b, unsigned int op)
{
    switch (op) {
        case 0: return a + b;
        case 1: return a - b;
        case 2: return a * b;
        case 3: return a / b;
        case 4: return fmaxf(a, b);
        case 5: return fminf(a, b);
        default: return a;
    }
}

__global__ void broadcast_row_kernel(
    const float* __restrict__ A,
    const float* __restrict__ vec,
    float* __restrict__ out,
    unsigned int rows, unsigned int cols, unsigned int op)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    const unsigned int r = idx / cols;
    out[idx] = apply_binary_op(A[idx], vec[r], op);
}

__global__ void broadcast_col_kernel(
    const float* __restrict__ A,
    const float* __restrict__ vec,
    float* __restrict__ out,
    unsigned int rows, unsigned int cols, unsigned int op)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    const unsigned int c = idx % cols;
    out[idx] = apply_binary_op(A[idx], vec[c], op);
}

// ══════════════════════════════════════════════════════════════════════════
// 转置与重排内核
// ══════════════════════════════════════════════════════════════════════════

__global__ void transpose_kernel(
    const float* __restrict__ A, float* __restrict__ out,
    unsigned int R, unsigned int C)
{
    __shared__ float tile[TILE][TILE + 1]; // +1 避免 bank conflict

    const int r = blockIdx.y * TILE + threadIdx.y;
    const int c = blockIdx.x * TILE + threadIdx.x;

    if (r < R && c < C)
        tile[threadIdx.y][threadIdx.x] = A[r * C + c];
    __syncthreads();

    const int out_r = blockIdx.x * TILE + threadIdx.y;
    const int out_c = blockIdx.y * TILE + threadIdx.x;

    if (out_r < C && out_c < R)
        out[out_r * R + out_c] = tile[threadIdx.x][threadIdx.y];
}

__global__ void rearrange_3d_kernel(
    const float* __restrict__ A, float* __restrict__ out,
    unsigned int M, unsigned int B, unsigned int N,
    unsigned int inverse, unsigned int total)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    if (!inverse) {
        // (M, B*N) → (B*M, N)
        const unsigned int m = idx / (B * N);
        const unsigned int rem = idx % (B * N);
        const unsigned int b = rem / N;
        const unsigned int n = rem % N;
        out[b * M * N + m * N + n] = A[idx];
    } else {
        // (B*M, N) → (M, B*N)
        const unsigned int b = idx / (M * N);
        const unsigned int rem2 = idx % (M * N);
        const unsigned int m = rem2 / N;
        const unsigned int n = rem2 % N;
        out[m * B * N + b * N + n] = A[idx];
    }
}

// ══════════════════════════════════════════════════════════════════════════
// Gather / Scatter 内核
// ══════════════════════════════════════════════════════════════════════════

__global__ void gather_kernel(
    const float* __restrict__ table,
    const float* __restrict__ indices,
    float* __restrict__ out,
    unsigned int vocab, unsigned int D, unsigned int num)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num * D) return;

    const unsigned int i = idx / D;
    const unsigned int d = idx % D;
    const int row_idx = static_cast<int>(indices[i]);

    if (row_idx >= 0 && static_cast<unsigned int>(row_idx) < vocab)
        out[idx] = table[static_cast<unsigned int>(row_idx) * D + d];
    else
        out[idx] = 0.0f;
}

__global__ void scatter_add_kernel(
    float* __restrict__ dst,
    const float* __restrict__ indices,
    const float* __restrict__ grad,
    unsigned int vocab, unsigned int D, unsigned int num)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num * D) return;

    const unsigned int i = idx / D;
    const unsigned int d = idx % D;
    const int row_idx = static_cast<int>(indices[i]);

    if (row_idx >= 0 && static_cast<unsigned int>(row_idx) < vocab)
        atomicAdd(&dst[static_cast<unsigned int>(row_idx) * D + d], grad[idx]);
}

// ══════════════════════════════════════════════════════════════════════════
// C-linkage 接口实现
// ══════════════════════════════════════════════════════════════════════════

static constexpr int BLOCK_SIZE = 256;

static cudaStream_t to_stream(void* s)
{
    return s ? static_cast<cudaStream_t>(s) : cudaStream_t(0);
}

// ── 矩阵乘法 ─────────────────────────────────────────────────────────────

// 内部辅助：使用预创建的 cuBLAS 句柄执行 Sgemm（行主序 → 列主序转换）
// 行主序矩阵 A(A_rows × A_cols) 在内存中按行存储，行步长 = A_cols。
// cuBLAS 按列主序读取时，看到的是 (A_cols × A_rows) 的矩阵，leading dim = A_cols。
// 关键：leading dim 是内存布局属性，与 transA/transB（逻辑转置）无关，始终等于行步长。
#ifdef NN_HAS_CUBLAS
static int cublas_matmul_impl(
    cublasHandle_t handle,
    const float* A, const float* B, float* C,
    std::size_t A_rows, std::size_t A_cols,
    std::size_t B_rows, std::size_t B_cols,
    int transA, int transB)
{
    const int M = static_cast<int>(transA ? A_cols : A_rows);
    const int K = static_cast<int>(transA ? A_rows : A_cols);
    const int K2 = static_cast<int>(transB ? B_cols : B_rows);
    const int N = static_cast<int>(transB ? B_rows : B_cols);

    if (K != K2) return -1;

    // cuBLAS 列主序：行主序 C = A × B ⟺ 列主序 C^T = B^T × A^T
    // 调用形式：cublasSgemm(handle, opB, opA, N, M, K, ..., B, ldb, A, lda, ..., C, ldc)
    //   第一操作数 = B（行主序 B），其列主序 leading dim = B_cols（行步长）
    //   第二操作数 = A（行主序 A），其列主序 leading dim = A_cols（行步长）
    //   结果 C（行主序 M×N），其列主序 leading dim = N（行步长）
    const float alpha = 1.0f, beta = 0.0f;
    const int lda = static_cast<int>(A_cols);  // 行主序 A 的行步长（固定，与 transA 无关）
    const int ldb = static_cast<int>(B_cols);  // 行主序 B 的行步长（固定，与 transB 无关）
    const int ldc = N;                          // 行主序 C 的行步长

    cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

    CUBLAS_CHECK(cublasSgemm(handle,
        opB, opA,
        N, M, K,
        &alpha,
        B, ldb,
        A, lda,
        &beta,
        C, ldc));
    return 0;
}
#endif

extern "C" int cuda_matmul(
    const float* A, const float* B, float* C,
    std::size_t A_rows, std::size_t A_cols,
    std::size_t B_rows, std::size_t B_cols,
    int transA, int transB,
    void* stream)
{
    const int K = static_cast<int>(transA ? A_rows : A_cols);
    const int K2 = static_cast<int>(transB ? B_cols : B_rows);

    if (K != K2) return -1;

#ifdef NN_HAS_CUBLAS
    // 兼容路径：无外部句柄时创建临时句柄（推荐使用 cuda_matmul_with_handle 复用句柄）
    cublasHandle_t handle;
    cublasStatus_t st = cublasCreate(&handle);
    if (st != CUBLAS_STATUS_SUCCESS) return static_cast<int>(-st);
    cublasSetStream(handle, to_stream(stream));

    int ret = cublas_matmul_impl(handle, A, B, C,
                                 A_rows, A_cols, B_rows, B_cols, transA, transB);

    cublasDestroy(handle);
    return ret;
#else
    // 自定义内核
    const int M = static_cast<int>(transA ? A_cols : A_rows);
    const int N = static_cast<int>(transB ? B_rows : B_cols);
    dim3 threads(TILE, TILE);
    dim3 blocks((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    matmul_kernel<<<blocks, threads, 0, to_stream(stream)>>>(
        A, B, C, M, N, K, transA, transB);
    CUDA_CHECK(cudaGetLastError());
    return 0;
#endif
}

// 带预创建 cuBLAS 句柄的 matmul（避免每次调用创建/销毁句柄的开销）
// handle_ptr 为 nullptr 时退化为 cuda_matmul
extern "C" int cuda_matmul_with_handle(
    const float* A, const float* B, float* C,
    std::size_t A_rows, std::size_t A_cols,
    std::size_t B_rows, std::size_t B_cols,
    int transA, int transB,
    void* handle_ptr)
{
#ifdef NN_HAS_CUBLAS
    if (!handle_ptr)
        return cuda_matmul(A, B, C, A_rows, A_cols, B_rows, B_cols, transA, transB, nullptr);

    cublasHandle_t handle = static_cast<cublasHandle_t>(handle_ptr);
    return cublas_matmul_impl(handle, A, B, C,
                              A_rows, A_cols, B_rows, B_cols, transA, transB);
#else
    // 无 cuBLAS 时退化为自定义内核（忽略 handle_ptr）
    return cuda_matmul(A, B, C, A_rows, A_cols, B_rows, B_cols, transA, transB, nullptr);
#endif
}

extern "C" int cuda_batched_matmul(
    const float* A, const float* B, float* C,
    std::size_t A_rows, std::size_t A_cols,
    std::size_t B_rows, std::size_t B_cols,
    unsigned int batch, int transA, int transB,
    void* stream)
{
    if (batch == 0) return -1;

    const int a_rows_per = static_cast<int>(A_rows / batch);
    const int b_rows_per = static_cast<int>(B_rows / batch);
    const int M = transA ? static_cast<int>(A_cols) : a_rows_per;
    const int K = transA ? a_rows_per : static_cast<int>(A_cols);
    const int K2 = transB ? static_cast<int>(B_cols) : b_rows_per;
    const int N = transB ? b_rows_per : static_cast<int>(B_cols);

    if (K != K2) return -1;

    dim3 threads(TILE, TILE);
    dim3 blocks((N + TILE - 1) / TILE, (M + TILE - 1) / TILE, batch);
    batched_matmul_kernel<<<blocks, threads, 0, to_stream(stream)>>>(
        A, B, C, M, N, K, transA, transB, static_cast<int>(batch));
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// ── 逐元素运算 ───────────────────────────────────────────────────────────

extern "C" int cuda_elementwise_unary(
    const float* A, float* out,
    unsigned int count, unsigned int op,
    void* stream)
{
    const int blocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    elementwise_unary_kernel<<<blocks, BLOCK_SIZE, 0, to_stream(stream)>>>(
        A, out, count, op);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

extern "C" int cuda_elementwise_binary(
    const float* A, const float* B, float* out,
    unsigned int count, unsigned int op,
    void* stream)
{
    const int blocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    elementwise_binary_kernel<<<blocks, BLOCK_SIZE, 0, to_stream(stream)>>>(
        A, B, out, count, op);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

extern "C" int cuda_elementwise_binary_scalar(
    const float* A, float* out,
    unsigned int count, unsigned int op,
    float scalar, int scalar_first,
    void* stream)
{
    const int blocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    elementwise_binary_scalar_kernel<<<blocks, BLOCK_SIZE, 0, to_stream(stream)>>>(
        A, out, count, op, scalar, scalar_first);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

extern "C" int cuda_axpy(
    const float* A, const float* B, float* out,
    unsigned int count, float scalar,
    void* stream)
{
    const int blocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    axpy_kernel<<<blocks, BLOCK_SIZE, 0, to_stream(stream)>>>(
        A, B, out, count, scalar);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

extern "C" int cuda_elementwise_select_scalar_cond(
    const float* A, const float* then_v, float* out,
    unsigned int count, unsigned int cmp,
    float scalar_b, float scalar_else,
    void* stream)
{
    const int blocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    select_scalar_cond_kernel<<<blocks, BLOCK_SIZE, 0, to_stream(stream)>>>(
        A, then_v, out, count, cmp, scalar_b, scalar_else);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// ── 归约 ─────────────────────────────────────────────────────────────────

extern "C" int cuda_reduce_row(
    const float* A, float* out,
    unsigned int rows, unsigned int cols, unsigned int mode,
    void* stream)
{
    const int threads = 256;
    const size_t shared_mem = threads * sizeof(float);

    if (mode == 0)
        reduce_row_sum_kernel<<<rows, threads, shared_mem, to_stream(stream)>>>(
            A, out, rows, cols);
    else
        reduce_row_max_kernel<<<rows, threads, shared_mem, to_stream(stream)>>>(
            A, out, rows, cols);

    CUDA_CHECK(cudaGetLastError());
    return 0;
}

extern "C" int cuda_reduce_col(
    const float* A, float* out,
    unsigned int rows, unsigned int cols, unsigned int mode,
    void* stream)
{
    // 新版 col_reduce：每个 block 处理一列，线程协作归约所有行（shared memory tree reduction）
    // grid = (cols,)，每个 block threads 个线程协作
    const int threads = 256;
    const size_t shared_mem = threads * sizeof(float);

    if (mode == 0)
        reduce_col_sum_kernel<<<cols, threads, shared_mem, to_stream(stream)>>>(
            A, out, rows, cols);
    else
        reduce_col_max_kernel<<<cols, threads, shared_mem, to_stream(stream)>>>(
            A, out, rows, cols);

    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// ── 广播 ─────────────────────────────────────────────────────────────────

extern "C" int cuda_broadcast_row(
    const float* A, const float* vec, float* out,
    unsigned int rows, unsigned int cols, unsigned int op,
    void* stream)
{
    const unsigned int total = rows * cols;
    const int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    broadcast_row_kernel<<<blocks, BLOCK_SIZE, 0, to_stream(stream)>>>(
        A, vec, out, rows, cols, op);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

extern "C" int cuda_broadcast_col(
    const float* A, const float* vec, float* out,
    unsigned int rows, unsigned int cols, unsigned int op,
    void* stream)
{
    const unsigned int total = rows * cols;
    const int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    broadcast_col_kernel<<<blocks, BLOCK_SIZE, 0, to_stream(stream)>>>(
        A, vec, out, rows, cols, op);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// ── 转置与重排 ───────────────────────────────────────────────────────────

extern "C" int cuda_transpose(
    const float* A, float* out,
    unsigned int R, unsigned int C,
    void* stream)
{
    dim3 threads(TILE, TILE);
    dim3 blocks((C + TILE - 1) / TILE, (R + TILE - 1) / TILE);
    transpose_kernel<<<blocks, threads, 0, to_stream(stream)>>>(
        A, out, R, C);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

extern "C" int cuda_rearrange_3d(
    const float* A, float* out,
    unsigned int M, unsigned int B, unsigned int N,
    unsigned int inverse,
    void* stream)
{
    const unsigned int total = M * B * N;
    const int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    rearrange_3d_kernel<<<blocks, BLOCK_SIZE, 0, to_stream(stream)>>>(
        A, out, M, B, N, inverse, total);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// ── Gather / Scatter ─────────────────────────────────────────────────────

extern "C" int cuda_gather(
    const float* table, const float* indices, float* out,
    unsigned int vocab, unsigned int D, unsigned int num,
    void* stream)
{
    const unsigned int total = num * D;
    const int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    gather_kernel<<<blocks, BLOCK_SIZE, 0, to_stream(stream)>>>(
        table, indices, out, vocab, D, num);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

extern "C" int cuda_scatter_add(
    float* dst, const float* indices, const float* grad,
    unsigned int vocab, unsigned int D, unsigned int num,
    void* stream)
{
    const unsigned int total = num * D;
    const int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    scatter_add_kernel<<<blocks, BLOCK_SIZE, 0, to_stream(stream)>>>(
        dst, indices, grad, vocab, D, num);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}
