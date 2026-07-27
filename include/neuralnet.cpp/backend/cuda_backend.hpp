// ── cuda_backend.hpp — CUDA 计算后端 ─────────────────────────────────────
// CudaBackend 管理 CUDA 设备、cuBLAS 句柄、CUDA 流，提供 GPU 操作 API。
// 内核函数通过 C-linkage 接口调用 nvcc 编译的静态库。
//
// 架构：
//   CudaBackend（本文件）→ 调用 cuda_kernels.h 中的 extern "C" 函数
//   cuda_kernels.cu（nvcc 编译）→ 实现 CUDA 内核
//
// 同步模型：
//   默认使用 CUDA 默认流（同步执行）。
//   begin_batch/end_batch 预留用于流式批处理（当前为 no-op）。
// ─────────────────────────────────────────────────────────────────────────

#ifndef NN_CUDA_BACKEND_HPP
#define NN_CUDA_BACKEND_HPP

#ifdef NN_HAS_CUDA

#include <cuda_runtime.h>
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

#include "../core_errors.hpp"
#include "../config.hpp"

// cuBLAS（可选，用于 matmul 加速）
#ifdef NN_HAS_CUBLAS
#include <cublas_v2.h>
#endif

// CUDA 内核 C-linkage 接口
#include "cuda_kernels.h"

namespace nn
{

// ── CUDA 错误检查 ─────────────────────────────────────────────────────────
namespace cuda_detail
{

[[nodiscard]] inline Result<void> cuda_check(cudaError_t err, const char* file, int line)
{
    if (err != cudaSuccess)
    {
        return std::unexpected(Error{
            std::string("CUDA error ") + cudaGetErrorString(err) +
            " at " + file + ":" + std::to_string(line)});
    }
    return {};
}

} // namespace cuda_detail

#define NN_CUDA_CHECK(call) cuda_detail::cuda_check((call), __FILE__, __LINE__)

// ══════════════════════════════════════════════════════════════════════════
// CudaBuffer — GPU 显存 RAII 封装
// ══════════════════════════════════════════════════════════════════════════

class CudaBuffer
{
private:
    void* ptr_ = nullptr;
    std::size_t size_ = 0; // 元素数

public:
    CudaBuffer() = default;

    CudaBuffer(void* ptr, std::size_t size) : ptr_(ptr), size_(size) {}

    ~CudaBuffer()
    {
        if (ptr_)
            cudaFree(ptr_);
    }

    // 移动语义
    CudaBuffer(CudaBuffer&& o) noexcept : ptr_(o.ptr_), size_(o.size_)
    {
        o.ptr_ = nullptr;
        o.size_ = 0;
    }

    CudaBuffer& operator=(CudaBuffer&& o) noexcept
    {
        if (this != &o)
        {
            if (ptr_) cudaFree(ptr_);
            ptr_ = o.ptr_;
            size_ = o.size_;
            o.ptr_ = nullptr;
            o.size_ = 0;
        }
        return *this;
    }

    // 禁止拷贝
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    [[nodiscard]] static Result<CudaBuffer> allocate(std::size_t elem_count)
    {
        void* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, elem_count * sizeof(float));
        if (err != cudaSuccess)
            return std::unexpected(Error{
                std::string("cudaMalloc failed: ") + cudaGetErrorString(err) +
                " (" + std::to_string(elem_count * sizeof(float)) + " bytes)"});
        return CudaBuffer(ptr, elem_count);
    }

    [[nodiscard]] void* ptr() const noexcept { return ptr_; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] bool valid() const noexcept { return ptr_ != nullptr; }
};

// ══════════════════════════════════════════════════════════════════════════
// CudaTensor — GPU 张量抽象
// ══════════════════════════════════════════════════════════════════════════

class CudaTensor
{
private:
    std::shared_ptr<CudaBuffer> buffer_;
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;

public:
    CudaTensor() = default;

    CudaTensor(std::shared_ptr<CudaBuffer> buffer, std::size_t rows, std::size_t cols)
        : buffer_(std::move(buffer)), rows_(rows), cols_(cols) {}

    // 从 CPU Matrix 创建（上传数据）
    [[nodiscard]] static Result<CudaTensor> from_matrix(
        const Matrix& cpu_mat, class CudaBackend& backend);

    // 创建空的 GPU Tensor（用于输出）
    [[nodiscard]] static Result<CudaTensor> create_empty(
        std::size_t rows, std::size_t cols, class CudaBackend& backend);

    // 转换为 CPU Matrix（下载数据）
    [[nodiscard]] Result<Matrix> to_matrix(class CudaBackend& backend) const;

    // 访问器
    [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::size_t cols() const noexcept { return cols_; }
    [[nodiscard]] bool valid() const noexcept { return buffer_ && buffer_->valid(); }
    [[nodiscard]] const CudaBuffer& buffer() const noexcept { return *buffer_; }
    [[nodiscard]] float* data() const noexcept { return buffer_ ? static_cast<float*>(buffer_->ptr()) : nullptr; }

    // 零拷贝 reshape
    [[nodiscard]] CudaTensor with_shape(std::size_t new_rows, std::size_t new_cols) const
    {
        return CudaTensor(buffer_, new_rows, new_cols);
    }
};

// ══════════════════════════════════════════════════════════════════════════
// CudaBackend — CUDA 计算后端单例
// ══════════════════════════════════════════════════════════════════════════

class CudaBackend
{
private:
    int device_id_ = 0;
    cudaDeviceProp device_props_{};
    bool initialized_ = false;
    std::mutex init_mutex_;

    CudaBackend() = default;

public:
    ~CudaBackend() = default;

    // 禁止拷贝和移动
    CudaBackend(const CudaBackend&) = delete;
    CudaBackend& operator=(const CudaBackend&) = delete;
    CudaBackend(CudaBackend&&) = delete;
    CudaBackend& operator=(CudaBackend&&) = delete;

    [[nodiscard]] static CudaBackend& instance()
    {
        static CudaBackend backend;
        return backend;
    }

    // ── 初始化 ─────────────────────────────────────────────────────────────
    [[nodiscard]] Result<void> initialize()
    {
        std::lock_guard lock(init_mutex_);
        if (initialized_) return {};

        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0)
            return std::unexpected(Error{"No CUDA devices found"});

        // 选择第一个设备
        err = cudaSetDevice(device_id_);
        if (err != cudaSuccess)
            return std::unexpected(Error{"cudaSetDevice failed"});

        err = cudaGetDeviceProperties(&device_props_, device_id_);
        if (err != cudaSuccess)
            return std::unexpected(Error{"cudaGetDeviceProperties failed"});

        initialized_ = true;
        return {};
    }

    [[nodiscard]] bool is_initialized() const noexcept { return initialized_; }
    [[nodiscard]] bool gpu_available() const noexcept { return initialized_; }

    [[nodiscard]] const cudaDeviceProp& device_props() const noexcept { return device_props_; }
    [[nodiscard]] int device_id() const noexcept { return device_id_; }

    // ── 批处理控制（当前为 no-op，预留 CUDA 流支持）─────────────────────
    [[nodiscard]] Result<void> begin_batch() { return {}; }
    [[nodiscard]] Result<void> end_batch() { return {}; }
    [[nodiscard]] bool in_batch() const noexcept { return false; }

    // ── 上传/下载 ─────────────────────────────────────────────────────────
    [[nodiscard]] Result<void> upload_blocking(CudaTensor& dst, std::span<const Scalar> cpu_data)
    {
        if (!dst.valid())
            return std::unexpected(Error{"Invalid destination CudaTensor"});
        if (cpu_data.size() != dst.rows() * dst.cols())
            return std::unexpected(Error{"Upload size mismatch"});

        auto err = cudaMemcpy(dst.data(), cpu_data.data(),
                              cpu_data.size_bytes(), cudaMemcpyHostToDevice);
        return NN_CUDA_CHECK(err);
    }

    [[nodiscard]] Result<void> download_blocking(const CudaTensor& src, std::span<Scalar> cpu_data)
    {
        if (!src.valid())
            return std::unexpected(Error{"Invalid source CudaTensor"});
        if (cpu_data.size() != src.rows() * src.cols())
            return std::unexpected(Error{"Download size mismatch"});

        auto err = cudaMemcpy(cpu_data.data(), src.data(),
                              cpu_data.size_bytes(), cudaMemcpyDeviceToHost);
        return NN_CUDA_CHECK(err);
    }

    // ── 纯 GPU 矩阵乘法 ──────────────────────────────────────────────────
    [[nodiscard]] Result<CudaTensor> matmul_gpu(
        const CudaTensor& A, const CudaTensor& B,
        uint32_t transA = 0, uint32_t transB = 0)
    {
        const auto M = transA ? A.cols() : A.rows();
        const auto K = transA ? A.rows() : A.cols();
        const auto N = transB ? B.rows() : B.cols();
        const auto K_B = transB ? B.cols() : B.rows();
        if (K != K_B)
            return std::unexpected(Error{"matmul: K dimension mismatch"});

        auto C_res = CudaTensor::create_empty(M, N, *this);
        if (!C_res) return std::unexpected(C_res.error());

        int err = cuda_matmul(
            A.data(), B.data(), C_res->data(),
            A.rows(), A.cols(), B.rows(), B.cols(),
            transA, transB, nullptr);
        if (err != 0)
            return std::unexpected(Error{"cuda_matmul failed: " + std::to_string(err)});

        cudaDeviceSynchronize();
        return std::move(*C_res);
    }

    // ── 批量矩阵乘法 ─────────────────────────────────────────────────────
    [[nodiscard]] Result<CudaTensor> batched_matmul_gpu(
        const CudaTensor& A, const CudaTensor& B,
        uint32_t batch,
        uint32_t transA = 0, uint32_t transB = 0)
    {
        if (batch == 0)
            return std::unexpected(Error{"batched_matmul: batch must be > 0"});
        if (A.rows() % batch != 0 || B.rows() % batch != 0)
            return std::unexpected(Error{"batched_matmul: rows not divisible by batch"});

        const auto a_rows_per = A.rows() / batch;
        const auto M = transA ? A.cols() : a_rows_per;
        const auto K = transA ? a_rows_per : A.cols();
        const auto b_rows_per = B.rows() / batch;
        const auto K_B = transB ? B.cols() : b_rows_per;
        const auto N = transB ? b_rows_per : B.cols();
        if (K != K_B)
            return std::unexpected(Error{"batched_matmul: K dimension mismatch"});

        auto C_res = CudaTensor::create_empty(batch * M, N, *this);
        if (!C_res) return std::unexpected(C_res.error());

        int err = cuda_batched_matmul(
            A.data(), B.data(), C_res->data(),
            A.rows(), A.cols(), B.rows(), B.cols(),
            batch, transA, transB, nullptr);
        if (err != 0)
            return std::unexpected(Error{"cuda_batched_matmul failed"});

        cudaDeviceSynchronize();
        return std::move(*C_res);
    }

    // ── 逐元素运算 ────────────────────────────────────────────────────────
    [[nodiscard]] Result<CudaTensor> elementwise_unary_gpu(
        const CudaTensor& A, uint32_t op)
    {
        auto out_res = CudaTensor::create_empty(A.rows(), A.cols(), *this);
        if (!out_res) return std::unexpected(out_res.error());

        const auto count = static_cast<uint32_t>(A.rows() * A.cols());
        int err = cuda_elementwise_unary(A.data(), out_res->data(), count, op, nullptr);
        if (err != 0) return std::unexpected(Error{"cuda_elementwise_unary failed"});

        cudaDeviceSynchronize();
        return std::move(*out_res);
    }

    [[nodiscard]] Result<CudaTensor> elementwise_binary_gpu(
        const CudaTensor& A, const CudaTensor& B, uint32_t op)
    {
        auto out_res = CudaTensor::create_empty(A.rows(), A.cols(), *this);
        if (!out_res) return std::unexpected(out_res.error());

        const auto count = static_cast<uint32_t>(A.rows() * A.cols());
        int err = cuda_elementwise_binary(A.data(), B.data(), out_res->data(), count, op, nullptr);
        if (err != 0) return std::unexpected(Error{"cuda_elementwise_binary failed"});

        cudaDeviceSynchronize();
        return std::move(*out_res);
    }

    [[nodiscard]] Result<CudaTensor> elementwise_binary_scalar_gpu(
        const CudaTensor& A, uint32_t op, float scalar, bool scalar_first)
    {
        auto out_res = CudaTensor::create_empty(A.rows(), A.cols(), *this);
        if (!out_res) return std::unexpected(out_res.error());

        const auto count = static_cast<uint32_t>(A.rows() * A.cols());
        int err = cuda_elementwise_binary_scalar(
            A.data(), out_res->data(), count, op, scalar, scalar_first ? 1 : 0, nullptr);
        if (err != 0) return std::unexpected(Error{"cuda_elementwise_binary_scalar failed"});

        cudaDeviceSynchronize();
        return std::move(*out_res);
    }

    [[nodiscard]] Result<CudaTensor> axpy_gpu(
        const CudaTensor& A, const CudaTensor& B, float scalar)
    {
        auto out_res = CudaTensor::create_empty(A.rows(), A.cols(), *this);
        if (!out_res) return std::unexpected(out_res.error());

        const auto count = static_cast<uint32_t>(A.rows() * A.cols());
        int err = cuda_axpy(A.data(), B.data(), out_res->data(), count, scalar, nullptr);
        if (err != 0) return std::unexpected(Error{"cuda_axpy failed"});

        cudaDeviceSynchronize();
        return std::move(*out_res);
    }

    [[nodiscard]] Result<CudaTensor> elementwise_select_scalar_cond_gpu(
        const CudaTensor& A, const CudaTensor& then_t,
        uint32_t cmp, float scalar_b, float scalar_else)
    {
        auto out_res = CudaTensor::create_empty(A.rows(), A.cols(), *this);
        if (!out_res) return std::unexpected(out_res.error());

        const auto count = static_cast<uint32_t>(A.rows() * A.cols());
        int err = cuda_elementwise_select_scalar_cond(
            A.data(), then_t.data(), out_res->data(), count, cmp, scalar_b, scalar_else, nullptr);
        if (err != 0) return std::unexpected(Error{"cuda_select_scalar_cond failed"});

        cudaDeviceSynchronize();
        return std::move(*out_res);
    }

    // ── 归约 ──────────────────────────────────────────────────────────────
    [[nodiscard]] Result<CudaTensor> reduce_gpu(
        const CudaTensor& input, uint32_t mode, uint32_t reduce_op)
    {
        const auto rows = static_cast<uint32_t>(input.rows());
        const auto cols = static_cast<uint32_t>(input.cols());

        std::size_t out_rows = (mode == 0) ? rows : 1;
        std::size_t out_cols = (mode == 0) ? 1 : cols;
        auto out_res = CudaTensor::create_empty(out_rows, out_cols, *this);
        if (!out_res) return std::unexpected(out_res.error());

        int err;
        if (mode == 0)
            err = cuda_reduce_row(input.data(), out_res->data(), rows, cols, reduce_op, nullptr);
        else
            err = cuda_reduce_col(input.data(), out_res->data(), rows, cols, reduce_op, nullptr);

        if (err != 0) return std::unexpected(Error{"cuda_reduce failed"});

        cudaDeviceSynchronize();
        return std::move(*out_res);
    }

    // ── 广播 ──────────────────────────────────────────────────────────────
    [[nodiscard]] Result<CudaTensor> broadcast_gpu(
        const CudaTensor& A, const CudaTensor& vec,
        uint32_t mode, uint32_t op)
    {
        auto out_res = CudaTensor::create_empty(A.rows(), A.cols(), *this);
        if (!out_res) return std::unexpected(out_res.error());

        const auto rows = static_cast<uint32_t>(A.rows());
        const auto cols = static_cast<uint32_t>(A.cols());

        int err;
        if (mode == 0)
            err = cuda_broadcast_row(A.data(), vec.data(), out_res->data(), rows, cols, op, nullptr);
        else
            err = cuda_broadcast_col(A.data(), vec.data(), out_res->data(), rows, cols, op, nullptr);

        if (err != 0) return std::unexpected(Error{"cuda_broadcast failed"});

        cudaDeviceSynchronize();
        return std::move(*out_res);
    }

    // ── 转置 ──────────────────────────────────────────────────────────────
    [[nodiscard]] Result<CudaTensor> transpose_gpu(const CudaTensor& A)
    {
        const auto R = static_cast<uint32_t>(A.rows());
        const auto C = static_cast<uint32_t>(A.cols());

        auto out_res = CudaTensor::create_empty(C, R, *this);
        if (!out_res) return std::unexpected(out_res.error());

        int err = cuda_transpose(A.data(), out_res->data(), R, C, nullptr);
        if (err != 0) return std::unexpected(Error{"cuda_transpose failed"});

        cudaDeviceSynchronize();
        return std::move(*out_res);
    }

    // ── 3D 重排 ───────────────────────────────────────────────────────────
    [[nodiscard]] Result<CudaTensor> rearrange_3d_gpu(
        const CudaTensor& input,
        uint32_t M, uint32_t B, uint32_t N, uint32_t inverse)
    {
        const auto total = M * B * N;
        if (total != input.rows() * input.cols())
            return std::unexpected(Error{"rearrange_3d: element count mismatch"});

        const auto out_rows = inverse ? M : (static_cast<std::size_t>(B) * M);
        const auto out_cols = inverse ? (static_cast<std::size_t>(B) * N) : N;
        auto out_res = CudaTensor::create_empty(out_rows, out_cols, *this);
        if (!out_res) return std::unexpected(out_res.error());

        int err = cuda_rearrange_3d(input.data(), out_res->data(), M, B, N, inverse, nullptr);
        if (err != 0) return std::unexpected(Error{"cuda_rearrange_3d failed"});

        cudaDeviceSynchronize();
        return std::move(*out_res);
    }

    // ── Fill zero ─────────────────────────────────────────────────────────
    [[nodiscard]] Result<void> fill_zero_gpu(CudaTensor& tensor)
    {
        auto err = cudaMemset(tensor.data(), 0, tensor.rows() * tensor.cols() * sizeof(float));
        return NN_CUDA_CHECK(err);
    }

    // ── Clone ─────────────────────────────────────────────────────────────
    [[nodiscard]] Result<CudaTensor> clone_gpu(const CudaTensor& src)
    {
        auto dst_res = CudaTensor::create_empty(src.rows(), src.cols(), *this);
        if (!dst_res) return std::unexpected(dst_res.error());

        auto err = cudaMemcpy(dst_res->data(), src.data(),
                              src.rows() * src.cols() * sizeof(float),
                              cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess)
            return std::unexpected(Error{"cudaMemcpy D2D failed"});

        return std::move(*dst_res);
    }

    // ── 行切片 ────────────────────────────────────────────────────────────
    [[nodiscard]] Result<CudaTensor> slice_rows_gpu(
        const CudaTensor& src, std::size_t start_row, std::size_t count)
    {
        if (start_row + count > src.rows())
            return std::unexpected(Error{"slice_rows: out of bounds"});

        auto dst_res = CudaTensor::create_empty(count, src.cols(), *this);
        if (!dst_res) return std::unexpected(dst_res.error());

        const std::size_t bytes = count * src.cols() * sizeof(float);
        const std::size_t src_offset = start_row * src.cols() * sizeof(float);

        auto err = cudaMemcpy(dst_res->data(),
                              static_cast<const char*>(src.buffer().ptr()) + src_offset,
                              bytes, cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess)
            return std::unexpected(Error{"slice_rows cudaMemcpy failed"});

        return std::move(*dst_res);
    }

    // ── 行插入 ────────────────────────────────────────────────────────────
    [[nodiscard]] Result<void> insert_rows_gpu(
        CudaTensor& dst, std::size_t dst_start_row, const CudaTensor& src)
    {
        if (dst.cols() != src.cols())
            return std::unexpected(Error{"insert_rows: column mismatch"});
        if (dst_start_row + src.rows() > dst.rows())
            return std::unexpected(Error{"insert_rows: out of bounds"});

        const std::size_t bytes = src.rows() * src.cols() * sizeof(float);
        const std::size_t dst_offset = dst_start_row * dst.cols() * sizeof(float);

        auto err = cudaMemcpy(static_cast<char*>(dst.buffer().ptr()) + dst_offset,
                              src.data(), bytes, cudaMemcpyDeviceToDevice);
        return NN_CUDA_CHECK(err);
    }

    // ── Gather ────────────────────────────────────────────────────────────
    [[nodiscard]] Result<CudaTensor> gather_gpu(
        const CudaTensor& table, const CudaTensor& indices)
    {
        const auto vocab = static_cast<uint32_t>(table.rows());
        const auto D = static_cast<uint32_t>(table.cols());
        const auto num = static_cast<uint32_t>(indices.rows() * indices.cols());

        auto out_res = CudaTensor::create_empty(num, D, *this);
        if (!out_res) return std::unexpected(out_res.error());

        int err = cuda_gather(
            table.data(), indices.data(),
            out_res->data(), vocab, D, num, nullptr);
        if (err != 0) return std::unexpected(Error{"cuda_gather failed"});

        cudaDeviceSynchronize();
        return std::move(*out_res);
    }

    // ── Scatter Add ───────────────────────────────────────────────────────
    [[nodiscard]] Result<void> scatter_add_gpu(
        CudaTensor& dst, const CudaTensor& indices, const CudaTensor& grad)
    {
        const auto vocab = static_cast<uint32_t>(dst.rows());
        const auto D = static_cast<uint32_t>(dst.cols());
        const auto num = static_cast<uint32_t>(indices.rows() * indices.cols());

        int err = cuda_scatter_add(
            dst.data(), indices.data(),
            grad.data(), vocab, D, num, nullptr);
        if (err != 0) return std::unexpected(Error{"cuda_scatter_add failed"});

        cudaDeviceSynchronize();
        return {};
    }
};

// ══════════════════════════════════════════════════════════════════════════
// CudaTensor 方法实现（需要 CudaBackend 完整定义）
// ══════════════════════════════════════════════════════════════════════════

inline Result<CudaTensor> CudaTensor::from_matrix(const Matrix& cpu_mat, CudaBackend& backend)
{
    if (cpu_mat.empty())
        return std::unexpected(Error{"Empty matrix"});

    auto buf_res = CudaBuffer::allocate(cpu_mat.size());
    if (!buf_res) return std::unexpected(buf_res.error());

    auto tensor = CudaTensor(
        std::make_shared<CudaBuffer>(std::move(*buf_res)),
        cpu_mat.rows(), cpu_mat.cols());

    auto upload_res = backend.upload_blocking(tensor, cpu_mat.span());
    if (!upload_res) return std::unexpected(upload_res.error());

    return tensor;
}

inline Result<CudaTensor> CudaTensor::create_empty(
    std::size_t rows, std::size_t cols, CudaBackend& /*backend*/)
{
    auto buf_res = CudaBuffer::allocate(rows * cols);
    if (!buf_res) return std::unexpected(buf_res.error());

    return CudaTensor(std::make_shared<CudaBuffer>(std::move(*buf_res)), rows, cols);
}

inline Result<Matrix> CudaTensor::to_matrix(CudaBackend& backend) const
{
    if (!valid())
        return std::unexpected(Error{"Invalid CudaTensor"});

    Matrix cpu_mat(rows_, cols_);
    auto dl_res = backend.download_blocking(*this, cpu_mat.span());
    if (!dl_res) return std::unexpected(dl_res.error());

    return cpu_mat;
}

} // namespace nn

#endif // NN_HAS_CUDA
#endif // NN_CUDA_BACKEND_HPP
