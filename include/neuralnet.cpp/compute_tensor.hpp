#pragma once

// ── compute_tensor.hpp — 统一张量类型（多精度，docs/23 §6.1）──────────────────
// Tensor 是跨设备的统一数据容器：CPU 持有 Matrix<P>，GPU 持有 GpuTensorT<P>。
// Layer 和 ComputeEngine 只操作 Tensor，不关心底层存储设备或精度。
//
// 设计要点（docs/23 §6.1 Q2）：
//   - Tensor 非模板：运行时 precision_ + 类型擦除存储（std::variant）
//   - Matrix<P> / GpuTensorT<P> 为存储/代数层模板（模板化到设备）
//   - 访问器按 P 模板化（默认 F32 → 现有调用点零改动）
//   - precision_ 与 device_ 共同唯一确定存储的有效类型
//   - 使用 shared_ptr 内部持有存储，拷贝廉价（零拷贝传递）
// ─────────────────────────────────────────────────────────────────────────

#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <variant>

#include "core_config.hpp"
#include "core_errors.hpp"
#include "precision.hpp"
#include "algebra_matrix.hpp"

#ifdef NN_HAS_VULKAN
#include "backend/compute_vk_backend.hpp"
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
// Tensor — 统一跨设备张量（多精度，docs/23 §6.1）
//
// 内存布局（variant 交替表）：
//   cpu_data_ : variant< shared_ptr<MatrixT<F16>>,  shared_ptr<MatrixT<F32>> >
//               index 0 = F16,  index 1 = F32（与 Precision 枚举值一致）
//   gpu_data_ : variant< shared_ptr<GpuTensorT<F16>>, shared_ptr<GpuTensorT<F32>> >
//               index 0 = F16,  index 1 = F32
// ───────────────────────────────────────────────────────────────────────────
class Tensor
{
private:
    Device device_ = Device::CPU;
    Precision precision_ = Precision::F32;
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;

    // ── CPU 存储：类型擦除（std::variant，§6.1）──────────────────────────
    // Phase 1：F16（index 0）/ F32（index 1）
    using CpuF16 = std::shared_ptr<MatrixT<Precision::F16>>;
    using CpuF32 = std::shared_ptr<MatrixT<Precision::F32>>;
    std::variant<CpuF16, CpuF32> cpu_data_;   // default: index 0 (F16 slot, empty ptr)

#ifdef NN_HAS_VULKAN
    // ── GPU 存储：类型擦除（std::variant，§6.3）──────────────────────────
    using GpuF16 = std::shared_ptr<GpuTensorT<Precision::F16>>;
    using GpuF32 = std::shared_ptr<GpuTensorT<Precision::F32>>;
    std::variant<GpuF16, GpuF32> gpu_data_;
#endif


    // ── variant 安全访问辅助（避免 std::get UB，§6.1）────────────────────
    template <Precision P>
    [[nodiscard]] std::shared_ptr<MatrixT<P>> cpu_get() const noexcept
    {
        constexpr auto slot = static_cast<std::size_t>(P);
        return cpu_data_.index() == slot ? std::get<slot>(cpu_data_) : nullptr;
    }

    // 同 cpu_get，但返回**裸指针、不拷贝 shared_ptr**。
    // 逐元素热点路径（DSL 模板求值的叶子 eval、逐元素原语）会按元素调用
    // cpu_matrix()；若每次拷贝 shared_ptr，则每元素两次原子引用计数操作
    // （实测 ~20ns/元素，使 DSL 模板路径比等价手写循环慢 8-70 倍）。
    // 调用方（cpu_matrix 的调用者）本身持有该 Tensor，故裸指针在其生存期内有效。
    template <Precision P>
    [[nodiscard]] MatrixT<P>* cpu_get_ptr() noexcept
    {
        constexpr auto slot = static_cast<std::size_t>(P);
        return cpu_data_.index() == slot ? std::get<slot>(cpu_data_).get() : nullptr;
    }
    template <Precision P>
    [[nodiscard]] const MatrixT<P>* cpu_get_ptr() const noexcept
    {
        constexpr auto slot = static_cast<std::size_t>(P);
        return cpu_data_.index() == slot ? std::get<slot>(cpu_data_).get() : nullptr;
    }

#ifdef NN_HAS_VULKAN
    template <Precision P>
    [[nodiscard]] std::shared_ptr<GpuTensorT<P>> gpu_get() const noexcept
    {
        constexpr auto slot = static_cast<std::size_t>(P);
        return gpu_data_.index() == slot ? std::get<slot>(gpu_data_) : nullptr;
    }
#endif

public:
    Tensor() = default;

    // ── CPU 构造（f32，现有 API 零改动）──────────────────────────────────
    explicit Tensor(std::shared_ptr<Matrix> m)
        : device_(Device::CPU), precision_(Precision::F32),
          rows_(m->rows()), cols_(m->cols()),
          cpu_data_(std::in_place_index<1>, std::move(m)) {}

    // ── CPU 构造（f16，新，§6.4）─────────────────────────────────────────
    explicit Tensor(std::shared_ptr<MatrixT<Precision::F16>> m)
        : device_(Device::CPU), precision_(Precision::F16),
          rows_(m->rows()), cols_(m->cols()),
          cpu_data_(std::in_place_index<0>, std::move(m)) {}

    // ── 从 Matrix 创建（f32 by-value，现有 API）──────────────────────────
    static Tensor from_matrix(Matrix m)
    {
        return Tensor(std::make_shared<Matrix>(std::move(m)));
    }

    // ── 从 MatrixT<F16> 创建（f16 by-value，新，§6.4）───────────────────
    static Tensor from_matrix(MatrixT<Precision::F16> m)
    {
        return Tensor(std::make_shared<MatrixT<Precision::F16>>(std::move(m)));
    }

    // ── 创建 CPU 空张量（f32，现有 API）──────────────────────────────────
    static Tensor cpu(std::size_t rows, std::size_t cols)
    {
        return Tensor(std::make_shared<Matrix>(rows, cols));
    }

    // ── 创建 CPU 未初始化张量（f32）───────────────────────────────────────
    // 语义契约：调用方必须在任何读取之前把**全部**元素写满。
    // 用于"输出会被完整覆盖"的引擎内部路径（eval_expr / elementwise_* 的
    // 输出缓冲）：省掉"分配 + 写满一遍零 + 马上被全覆盖"里的那一遍零写。
    // 实测本机单线程写满 1.57MB 要 0.50ms（~3.2 GB/s），是纯浪费。
    [[nodiscard]] static Tensor cpu_uninitialized(std::size_t rows, std::size_t cols)
    {
        return Tensor(std::make_shared<Matrix>(rows, cols, Matrix::uninitialized_tag{}));
    }

    // ── 创建 CPU 空张量（显式 P，§6.4）───────────────────────────────────
    // compile-time P；BF16/F64 → static_assert 报错
    template <Precision P>
    [[nodiscard]] static Tensor cpu(std::size_t rows, std::size_t cols)
    {
        static_assert(P == Precision::F16 || P == Precision::F32,
                      "Phase 1 仅支持 F16/F32（BF16/F64 为保留值）");
        return Tensor(std::make_shared<MatrixT<P>>(rows, cols));
    }

#ifdef NN_HAS_VULKAN
    // ── GPU 构造（f32，现有 API）──────────────────────────────────────────
    explicit Tensor(std::shared_ptr<GpuTensor> t)
        : device_(Device::GPU), precision_(Precision::F32),
          rows_(t->rows()), cols_(t->cols()),
          gpu_data_(std::in_place_index<1>, std::move(t)) {}

    // ── GPU 构造（f16，新，§6.4）─────────────────────────────────────────
    explicit Tensor(std::shared_ptr<GpuTensorT<Precision::F16>> t)
        : device_(Device::GPU), precision_(Precision::F16),
          rows_(t->rows()), cols_(t->cols()),
          gpu_data_(std::in_place_index<0>, std::move(t)) {}

    // ── 从 GpuTensor 创建（f32，现有 API）────────────────────────────────
    static Tensor from_gpu(GpuTensor t)
    {
        return Tensor(std::make_shared<GpuTensor>(std::move(t)));
    }

    // ── 从 GpuTensorT<F16> 创建（f16，新）────────────────────────────────
    static Tensor from_gpu(GpuTensorT<Precision::F16> t)
    {
        return Tensor(std::make_shared<GpuTensorT<Precision::F16>>(std::move(t)));
    }
#endif

    // ── 访问器 ────────────────────────────────────────────────────────────
    [[nodiscard]] Device device() const noexcept { return device_; }
    [[nodiscard]] Precision precision() const noexcept { return precision_; }
    [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::size_t cols() const noexcept { return cols_; }
    [[nodiscard]] std::size_t size() const noexcept { return rows_ * cols_; }
    [[nodiscard]] bool is_cpu() const noexcept { return device_ == Device::CPU; }
    [[nodiscard]] bool is_gpu() const noexcept { return device_ == Device::GPU; }

    // ── valid：存储匹配 precision_ 且非空 ────────────────────────────────
    [[nodiscard]] bool valid() const noexcept
    {
        if (device_ == Device::CPU)
        {
            if (precision_ == Precision::F16)
                return cpu_data_.index() == 0 && static_cast<bool>(cpu_get<Precision::F16>());
            return cpu_data_.index() == 1 && static_cast<bool>(cpu_get<Precision::F32>());
        }
#ifdef NN_HAS_VULKAN
        if (precision_ == Precision::F16)
            return gpu_data_.index() == 0 && static_cast<bool>(gpu_get<Precision::F16>());
        return gpu_data_.index() == 1 && static_cast<bool>(gpu_get<Precision::F32>());
#else
        return false;
#endif
    }

    // ── CPU 存储访问（模板化，默认 F32 → 现有调用点零改动，§6.1）───────
    template <Precision P = Precision::F32>
    [[nodiscard]] MatrixT<P>& cpu_matrix()
    {
        static_assert(P == Precision::F16 || P == Precision::F32,
                      "Phase 1 仅支持 F16/F32");
        // 用 cpu_get_ptr（不拷贝 shared_ptr）：该函数在逐元素热点上会被按元素
        // 调用，拷贝 shared_ptr 的原子引用计数是实测的主要开销来源。
        MatrixT<P>* p = cpu_get_ptr<P>();
        NN_ASSERT(device_ == Device::CPU && p,
                  "cpu_matrix<P>(): tensor has no P-precision CPU storage");
        return *p;
    }

    template <Precision P = Precision::F32>
    [[nodiscard]] const MatrixT<P>& cpu_matrix() const
    {
        static_assert(P == Precision::F16 || P == Precision::F32,
                      "Phase 1 仅支持 F16/F32");
        const MatrixT<P>* p = cpu_get_ptr<P>();
        NN_ASSERT(device_ == Device::CPU && p,
                  "cpu_matrix<P>() const: tensor has no P-precision CPU storage");
        return *p;
    }

    template <Precision P = Precision::F32>
    [[nodiscard]] std::shared_ptr<MatrixT<P>> cpu_shared() const noexcept
    {
        return cpu_get<P>();
    }

#ifdef NN_HAS_VULKAN
    // ── GPU 存储访问（模板化，默认 F32 → 现有调用点零改动，§6.1）───────
    template <Precision P = Precision::F32>
    [[nodiscard]] GpuTensorT<P>& gpu_tensor()
    {
        static_assert(P == Precision::F16 || P == Precision::F32,
                      "Phase 1 仅支持 F16/F32");
        auto p = gpu_get<P>();
        NN_ASSERT(device_ == Device::GPU && p,
                  "gpu_tensor<P>(): tensor has no P-precision GPU storage");
        return *p;
    }

    template <Precision P = Precision::F32>
    [[nodiscard]] const GpuTensorT<P>& gpu_tensor() const
    {
        static_assert(P == Precision::F16 || P == Precision::F32,
                      "Phase 1 仅支持 F16/F32");
        auto p = gpu_get<P>();
        NN_ASSERT(device_ == Device::GPU && p,
                  "gpu_tensor<P>() const: tensor has no P-precision GPU storage");
        return *p;
    }

    template <Precision P = Precision::F32>
    [[nodiscard]] std::shared_ptr<GpuTensorT<P>> gpu_shared() const noexcept
    {
        return gpu_get<P>();
    }
#endif

    // ── 形状描述（调试用） ────────────────────────────────────────────────
    [[nodiscard]] std::string shape_str() const
    {
        return "(" + std::to_string(rows_) + "x" + std::to_string(cols_) + ")"
             + (device_ == Device::CPU ? "[CPU]" : "[GPU]")
             + precision_name(precision_);
    }

    // ── 零拷贝 reshape（共享底层 buffer，仅改变形状元数据）────────────────
    // 后端无关：CPU/Vulkan 均适用，避免上层代码直接访问 gpu_tensor()
    [[nodiscard]] Tensor reshape(std::size_t new_rows, std::size_t new_cols) const
    {
        NN_ASSERT(rows_ * cols_ == new_rows * new_cols,
                  "reshape: element count mismatch");
        Tensor t;
        t.device_ = device_;
        t.precision_ = precision_;
        t.rows_ = new_rows;
        t.cols_ = new_cols;
#ifdef NN_HAS_VULKAN
        t.gpu_data_ = gpu_data_;
#endif
        if (device_ == Device::CPU)
        {
            // CPU 的 Matrix 无"零拷贝视图"能力：reshape 需复制数据到新形状的 Matrix。
            // 按 precision_ 选择正确的 variant 槽位（§6.1）。
            if (precision_ == Precision::F16)
            {
                auto m = std::make_shared<MatrixT<Precision::F16>>(new_rows, new_cols);
                const auto src = std::get<0>(cpu_data_)->span();
                auto dst = m->span();
                std::copy(src.begin(), src.end(), dst.begin());
                t.cpu_data_.template emplace<0>(std::move(m));
            }
            else
            {
                auto m = std::make_shared<MatrixT<Precision::F32>>(new_rows, new_cols);
                const auto src = std::get<1>(cpu_data_)->span();
                auto dst = m->span();
                std::copy(src.begin(), src.end(), dst.begin());
                t.cpu_data_.template emplace<1>(std::move(m));
            }
        }
        else
        {
            t.cpu_data_ = cpu_data_;  // GPU 模式：CPU variant 为空（保持一致）
        }
        return t;
    }
};

// ── 非拥有型引用包装器（替代裸指针） ─────────────────────────────────────
// 使用 std::reference_wrapper 替代 T*，明确表达 "引用但不拥有" 语义。
// 用于 Optimizer 持有的 params_/grads_ 和 Layer::parameters() 返回值。
using TensorRef = std::reference_wrapper<Tensor>;

} // namespace nn
