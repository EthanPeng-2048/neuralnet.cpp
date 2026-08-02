#ifndef NN_TENSOR_HPP
#define NN_TENSOR_HPP

// ── tensor.hpp — 统一张量类型 ─────────────────────────────────────────────
// Tensor 是跨设备的统一数据容器：CPU 持有 Matrix，GPU 持有 GpuTensor。
// Layer 和 ComputeEngine 只操作 Tensor，不关心底层存储设备。
//
// 设计要点：
//   - 使用 shared_ptr 内部持有存储，拷贝廉价（零拷贝传递）
//   - CPU/GPU 存储互斥：device() 决定哪个指针有效
//   - 跨设备传输由 ComputeEngine 负责，Tensor 本身不主动迁移
// ─────────────────────────────────────────────────────────────────────────

#include <cstddef>
#include <functional>
#include <memory>

#include "config.hpp"
#include "core_errors.hpp"
#include "algebra_matrix.hpp"

#ifdef NN_HAS_VULKAN
#include "backend/vk_backend.hpp"
#endif

#ifdef NN_HAS_CUDA
#include "backend/cuda_backend.hpp"
#endif

namespace nn
{

// ── 计算设备 ──────────────────────────────────────────────────────────────
enum class Device : uint8_t
{
    CPU,
    GPU,
};

// ══════════════════════════════════════════════════════════════════════════
// Tensor — 统一跨设备张量
// ══════════════════════════════════════════════════════════════════════════
class Tensor
{
private:
    Device device_ = Device::CPU;
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;

    // CPU 存储（device_ == CPU 时有效）
    std::shared_ptr<Matrix> cpu_data_;

#ifdef NN_HAS_VULKAN
    // GPU 存储（device_ == GPU 时有效）
    std::shared_ptr<GpuTensor> gpu_data_;
#endif

#ifdef NN_HAS_CUDA
    // CUDA 存储（device_ == GPU 时有效）
    std::shared_ptr<CudaTensor> cuda_data_;
#endif

public:
    Tensor() = default;

    // ── CPU 构造 ──────────────────────────────────────────────────────────
    explicit Tensor(std::shared_ptr<Matrix> m)
        : device_(Device::CPU), rows_(m->rows()), cols_(m->cols()), cpu_data_(std::move(m)) {}

    // 从 Matrix 构造（共享所有权）
    static Tensor from_matrix(Matrix m)
    {
        return Tensor(std::make_shared<Matrix>(std::move(m)));
    }

    // 创建 CPU 空张量
    static Tensor cpu(std::size_t rows, std::size_t cols)
    {
        return Tensor(std::make_shared<Matrix>(rows, cols));
    }

#ifdef NN_HAS_VULKAN
    // ── GPU 构造 ──────────────────────────────────────────────────────────
    explicit Tensor(std::shared_ptr<GpuTensor> t)
        : device_(Device::GPU), rows_(t->rows()), cols_(t->cols()), gpu_data_(std::move(t)) {}

    static Tensor from_gpu(GpuTensor t)
    {
        return Tensor(std::make_shared<GpuTensor>(std::move(t)));
    }
#endif

#ifdef NN_HAS_CUDA
    // ── CUDA 构造 ─────────────────────────────────────────────────────────
    explicit Tensor(std::shared_ptr<CudaTensor> t)
        : device_(Device::GPU), rows_(t->rows()), cols_(t->cols()), cuda_data_(std::move(t)) {}

    static Tensor from_cuda(CudaTensor t)
    {
        return Tensor(std::make_shared<CudaTensor>(std::move(t)));
    }
#endif

    // ── 访问器 ────────────────────────────────────────────────────────────
    [[nodiscard]] Device device() const noexcept { return device_; }
    [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::size_t cols() const noexcept { return cols_; }
    [[nodiscard]] std::size_t size() const noexcept { return rows_ * cols_; }
    [[nodiscard]] bool is_cpu() const noexcept { return device_ == Device::CPU; }
    [[nodiscard]] bool is_gpu() const noexcept { return device_ == Device::GPU; }
    [[nodiscard]] bool valid() const noexcept
    {
        if (device_ == Device::CPU)
            return static_cast<bool>(cpu_data_);
#ifdef NN_HAS_CUDA
        if (cuda_data_)
            return static_cast<bool>(cuda_data_);
#endif
#ifdef NN_HAS_VULKAN
        return static_cast<bool>(gpu_data_);
#else
        return false;
#endif
    }

    // ── CPU 存储访问 ──────────────────────────────────────────────────────
    [[nodiscard]] Matrix& cpu_matrix()
    {
        NN_ASSERT(device_ == Device::CPU && cpu_data_, "cpu_matrix() on non-CPU tensor");
        return *cpu_data_;
    }

    [[nodiscard]] const Matrix& cpu_matrix() const
    {
        NN_ASSERT(device_ == Device::CPU && cpu_data_, "cpu_matrix() on non-CPU tensor");
        return *cpu_data_;
    }

    [[nodiscard]] std::shared_ptr<Matrix> cpu_shared() const noexcept { return cpu_data_; }

#ifdef NN_HAS_VULKAN
    // ── GPU 存储访问 ──────────────────────────────────────────────────────
    [[nodiscard]] GpuTensor& gpu_tensor()
    {
        NN_ASSERT(device_ == Device::GPU && gpu_data_, "gpu_tensor() on non-GPU tensor");
        return *gpu_data_;
    }

    [[nodiscard]] const GpuTensor& gpu_tensor() const
    {
        NN_ASSERT(device_ == Device::GPU && gpu_data_, "gpu_tensor() on non-GPU tensor");
        return *gpu_data_;
    }

    [[nodiscard]] std::shared_ptr<GpuTensor> gpu_shared() const noexcept { return gpu_data_; }
#endif

#ifdef NN_HAS_CUDA
    // ── CUDA 存储访问 ─────────────────────────────────────────────────────
    [[nodiscard]] CudaTensor& cuda_tensor()
    {
        NN_ASSERT(device_ == Device::GPU && cuda_data_, "cuda_tensor() on non-CUDA tensor");
        return *cuda_data_;
    }

    [[nodiscard]] const CudaTensor& cuda_tensor() const
    {
        NN_ASSERT(device_ == Device::GPU && cuda_data_, "cuda_tensor() on non-CUDA tensor");
        return *cuda_data_;
    }

    [[nodiscard]] std::shared_ptr<CudaTensor> cuda_shared() const noexcept { return cuda_data_; }
#endif

    // ── 形状描述（调试用） ────────────────────────────────────────────────
    [[nodiscard]] std::string shape_str() const
    {
        return "(" + std::to_string(rows_) + "x" + std::to_string(cols_) + ")"
             + (device_ == Device::CPU ? "[CPU]" : "[GPU]");
    }

    // ── 零拷贝 reshape（共享底层 buffer，仅改变形状元数据）────────────────
    // 后端无关：CPU/Vulkan/CUDA 均适用，避免上层代码直接访问 gpu_tensor()/cuda_tensor()
    [[nodiscard]] Tensor reshape(std::size_t new_rows, std::size_t new_cols) const
    {
        NN_ASSERT(rows_ * cols_ == new_rows * new_cols,
                  "reshape: element count mismatch");
        Tensor t;
        t.device_ = device_;
        t.rows_ = new_rows;
        t.cols_ = new_cols;
        t.cpu_data_ = cpu_data_;
#ifdef NN_HAS_VULKAN
        t.gpu_data_ = gpu_data_;
#endif
#ifdef NN_HAS_CUDA
        t.cuda_data_ = cuda_data_;
#endif
        return t;
    }
};

// ── 非拥有型引用包装器（替代裸指针） ─────────────────────────────────────
// 使用 std::reference_wrapper 替代 T*，明确表达 "引用但不拥有" 语义。
// 用于 Optimizer 持有的 params_/grads_ 和 Layer::parameters() 返回值。
using TensorRef = std::reference_wrapper<Tensor>;

} // namespace nn

#endif // NN_TENSOR_HPP
