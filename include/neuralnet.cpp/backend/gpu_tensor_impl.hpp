// ── gpu_tensor_impl.hpp ─────────────────────────────────────────────────
// GpuTensor 方法实现（需要 Matrix 和 GpuBackend 的完整定义）
// 此文件在 algebra_matrix.hpp 末尾自动包含，解决循环依赖问题。
// ─────────────────────────────────────────────────────────────────────────

#ifndef NN_GPU_TENSOR_IMPL_HPP
#define NN_GPU_TENSOR_IMPL_HPP

#ifdef NN_HAS_VULKAN

namespace nn
{

inline Result<GpuTensor> GpuTensor::from_matrix(const Matrix& cpu_mat, GpuBackend& backend)
{
    if (cpu_mat.empty())
        return std::unexpected(Error{"Empty matrix"});

    auto buf_res = GpuBuffer::create_device_local(
        backend.device().device(), backend.memory_pool(),
        cpu_mat.size(),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    if (!buf_res)
        return std::unexpected(buf_res.error());

    auto tensor = GpuTensor(
        std::make_shared<GpuBuffer>(std::move(*buf_res)),
        cpu_mat.rows(), cpu_mat.cols());

    auto upload_res = backend.upload_blocking(tensor, cpu_mat.span());
    if (!upload_res)
        return std::unexpected(upload_res.error());

    return tensor;
}

inline Result<GpuTensor> GpuTensor::create_empty(
    std::size_t rows, std::size_t cols, GpuBackend& backend)
{
    auto buf_res = GpuBuffer::create_device_local(
        backend.device().device(), backend.memory_pool(),
        rows * cols,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    if (!buf_res)
        return std::unexpected(buf_res.error());

    return GpuTensor(std::make_shared<GpuBuffer>(std::move(*buf_res)), rows, cols);
}

inline Result<Matrix> GpuTensor::to_matrix(GpuBackend& backend) const
{
    if (!valid())
        return std::unexpected(Error{"Invalid GpuTensor"});

    Matrix cpu_mat(rows_, cols_);
    auto dl_res = backend.download_blocking(*this, cpu_mat.span());
    if (!dl_res)
        return std::unexpected(dl_res.error());

    return cpu_mat;
}

} // namespace nn

#endif // NN_HAS_VULKAN
#endif // NN_GPU_TENSOR_IMPL_HPP
