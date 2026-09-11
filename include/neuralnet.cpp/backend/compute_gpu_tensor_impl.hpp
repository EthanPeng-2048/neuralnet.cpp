// ── compute_gpu_tensor_impl.hpp ─────────────────────────────────────────────────
// GpuTensorT<P> 方法实现（需要 MatrixT<P> 和 GpuBackend 的完整定义）
// 此文件在 algebra_matrix.hpp 末尾自动包含，解决循环依赖问题。
// ─────────────────────────────────────────────────────────────────────────

#pragma once

#ifdef NN_HAS_VULKAN

namespace nn
{

// 张量 buffer 的标准 usage 标志：
//   STORAGE_BUFFER: 绑定到 shader 进行 GPU 计算
//   TRANSFER_DST:   允许 vkCmdFillBuffer(zero) / vkCmdCopyBuffer(insert_rows) 写入
//   TRANSFER_SRC:   允许 vkCmdCopyBuffer(clone / slice_rows) 读取
static constexpr VkBufferUsageFlags TENSOR_BUFFER_USAGE =
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

template <Precision P>
inline Result<GpuTensorT<P>> GpuTensorT<P>::from_matrix(const MatrixT<P>& cpu_mat, GpuBackend& backend)
{
    if (cpu_mat.empty())
        return std::unexpected(Error{"Empty matrix"});

    // buffer 字节数 = 元素数 × sizeof(elem<P>)（§6.3）
    const std::size_t byte_count = cpu_mat.size() * sizeof(elem<P>);
    auto buf_res = GpuBuffer::create_device_local(
        backend.device().device(), backend.memory_pool(),
        byte_count,
        TENSOR_BUFFER_USAGE);
    if (!buf_res)
        return std::unexpected(buf_res.error());

    auto tensor = GpuTensorT<P>(
        std::make_shared<GpuBuffer>(std::move(*buf_res)),
        cpu_mat.rows(), cpu_mat.cols());

    // 同精度跨设备 = 原始字节拷贝（§6.5）
    auto upload_res = backend.upload_blocking(tensor, cpu_mat.span());
    if (!upload_res)
        return std::unexpected(upload_res.error());

    return tensor;
}

template <Precision P>
inline Result<GpuTensorT<P>> GpuTensorT<P>::create_empty(
    std::size_t rows, std::size_t cols, GpuBackend& backend)
{
    // buffer 字节数 = 元素数 × sizeof(elem<P>)（§6.3）
    const std::size_t byte_count = rows * cols * sizeof(elem<P>);
    // TRANSFER_DST_BIT: 允许 vkCmdFillBuffer (zero) 和 vkCmdCopyBuffer (insert_rows) 写入
    // TRANSFER_SRC_BIT: 允许 vkCmdCopyBuffer (clone/slice_rows) 读取
    auto buf_res = GpuBuffer::create_device_local(
        backend.device().device(), backend.memory_pool(),
        byte_count,
        TENSOR_BUFFER_USAGE);
    if (!buf_res)
        return std::unexpected(buf_res.error());

    return GpuTensorT<P>(std::make_shared<GpuBuffer>(std::move(*buf_res)), rows, cols);
}

template <Precision P>
inline Result<GpuTensorT<P>> GpuTensorT<P>::create_host_visible_empty(
    std::size_t rows, std::size_t cols, GpuBackend& backend)
{
    // Host Visible 存储（activation offload）：数据驻留主机可见内存（§6.6）。
    // 仅用于 vkCmdCopyBuffer 中转（TRANSFER_DST/SRC）。
    const std::size_t byte_count = rows * cols * sizeof(elem<P>);
    auto buf_res = GpuBuffer::create_host_visible(
        backend.device().device(), backend.memory_pool(),
        byte_count,
        TENSOR_BUFFER_USAGE);
    if (!buf_res)
        return std::unexpected(buf_res.error());

    return GpuTensorT<P>(std::make_shared<GpuBuffer>(std::move(*buf_res)), rows, cols);
}

template <Precision P>
inline Result<MatrixT<P>> GpuTensorT<P>::to_matrix(GpuBackend& backend) const
{
    if (!valid())
        return std::unexpected(Error{"Invalid GpuTensor"});

    MatrixT<P> cpu_mat(rows_, cols_);
    auto dl_res = backend.download_blocking(*this, cpu_mat.span());
    if (!dl_res)
        return std::unexpected(dl_res.error());

    return cpu_mat;
}

} // namespace nn

#endif // NN_HAS_VULKAN
