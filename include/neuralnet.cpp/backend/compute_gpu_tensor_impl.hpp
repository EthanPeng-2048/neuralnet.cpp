// ── compute_gpu_tensor_impl.hpp ─────────────────────────────────────────────────
// GpuTensor 方法实现（需要 Matrix 和 GpuBackend 的完整定义）
// 此文件在 algebra_matrix.hpp 末尾自动包含，解决循环依赖问题。
// ─────────────────────────────────────────────────────────────────────────

#ifndef NN_COMPUTE_GPU_TENSOR_IMPL_HPP
#define NN_COMPUTE_GPU_TENSOR_IMPL_HPP

#ifdef NN_HAS_VULKAN

namespace nn
{

// 张量 buffer 的标准 usage 标志：
//   STORAGE_BUFFER: 绑定到 shader 进行 GPU 计算
//   TRANSFER_DST:   允许 vkCmdFillBuffer(zero) / vkCmdCopyBuffer(insert_rows) 写入
//   TRANSFER_SRC:   允许 vkCmdCopyBuffer(clone / slice_rows) 读取
static constexpr VkBufferUsageFlags TENSOR_BUFFER_USAGE =
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

inline Result<GpuTensor> GpuTensor::from_matrix(const Matrix& cpu_mat, GpuBackend& backend)
{
    if (cpu_mat.empty())
        return std::unexpected(Error{"Empty matrix"});

    auto buf_res = GpuBuffer::create_device_local(
        backend.device().device(), backend.memory_pool(),
        cpu_mat.size(),
        TENSOR_BUFFER_USAGE);
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
    // TRANSFER_DST_BIT: 允许 vkCmdFillBuffer (zero) 和 vkCmdCopyBuffer (insert_rows) 写入
    // TRANSFER_SRC_BIT: 允许 vkCmdCopyBuffer (clone/slice_rows) 读取
    auto buf_res = GpuBuffer::create_device_local(
        backend.device().device(), backend.memory_pool(),
        rows * cols,
        TENSOR_BUFFER_USAGE);
    if (!buf_res)
        return std::unexpected(buf_res.error());

    return GpuTensor(std::make_shared<GpuBuffer>(std::move(*buf_res)), rows, cols);
}

inline Result<GpuTensor> GpuTensor::create_host_visible_empty(
    std::size_t rows, std::size_t cols, GpuBackend& backend)
{
    // Host Visible 存储（activation offload）：数据驻留主机可见内存。
    // 仅用于 vkCmdCopyBuffer 中转（TRANSFER_DST/SRC）。
    auto buf_res = GpuBuffer::create_host_visible(
        backend.device().device(), backend.memory_pool(),
        rows * cols,
        TENSOR_BUFFER_USAGE);
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
#endif // NN_COMPUTE_GPU_TENSOR_IMPL_HPP
