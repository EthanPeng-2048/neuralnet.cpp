#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <execution>
#include <functional>
#include <memory>
#include <numeric>
#include <random>
#include <ranges>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "core_errors.hpp"
#include "core_assert.hpp"
#include "core_threadpool.hpp"
#include "core_config.hpp"
#include "algebra_span.hpp"
#include "algebra_compute.hpp"

// GPU 加速支持（可选，由 CMake 的 NN_HAS_VULKAN 宏控制）
#ifdef NN_HAS_VULKAN
#include "backend/compute_vk_backend.hpp"
#endif

namespace nn
{
    // ═══════════════════════════════════════════════════════════════════════
    //  Matrix 类 — 矩阵存储与运算原语
    //
    //  错误处理约定（与上层 L2 计算层不同）：
    //    - L1 代数层使用 NN_ASSERT 进行形状校验（编程错误检查）
    //    - L2+ 计算层使用 Result<T> 进行运行时错误处理（用户输入校验）
    //    - 原因：Matrix 是底层原语，assert 不影响 Release 性能；
    //            上层需要向用户报告错误，故使用 Result<T>
    //    - 参见 DEVELOPMENT_STANDARDS.md "分层职责单一" 章节
    // ═══════════════════════════════════════════════════════════════════════
    // ── MatrixT<P> — 类型化矩阵存储（docs/23 D2：存储精度是 P）─────────────
    // element = elem<P>（f16 / f32）；F32 实例与旧 Matrix 逐字节一致（T1 零回归）。
    // F16 实例遵循 §7.2 形式化定义：
    //   matmul / 归约：f16 载入 / f32 累加 / f16 写出（acc_type = f32）
    //   逐元素：f32 参考计算 + 每算子输出舍入到 f16（f16 运算符语义）
    // 注意：C++ 不允许模板与同名的 using 别名共存，故模板名为 MatrixT，
    //       f32 便捷别名保持 Matrix 不变（所有既有代码零改动）。
    template <Precision P>
    class MatrixT
    {
    public:
        static constexpr Precision precision = P;
        using element = elem<P>;
        using acc_type = acc<P>;  // F32: float（= element，现状）；F16: float（f32 累加）

    private:
        // ── 存储：std::unique_ptr<element[]> + 显式长度 ──────────────────
        // 为何不用 std::vector：vector 的"按尺寸构造"必然值初始化（写满一遍零），
        // 而"输出会被完整覆盖"的路径（如 dsl::compute 的逐元素结果）不需要这一遍。
        // 实测本机单线程写满 1.57MB 要 0.50ms（~3.2 GB/s，内存带宽受限），
        // 是每个全尺寸中间量的纯浪费。std::make_unique_for_overwrite（C++20）
        // 正是为"内存马上会被覆盖"设计。
        // 规范 §3.1：不出现 new[]/delete[]/裸指针所有权 —— unique_ptr 是
        // 规范推荐的替代；不穿透接口（storage 仅本类可见）。
        std::unique_ptr<element[]> data_{};
        std::size_t size_{0};
        std::size_t rows_{0};
        std::size_t cols_{0};

        [[nodiscard]] constexpr std::size_t index(std::size_t row, std::size_t col) const noexcept
        {
            return row * cols_ + col;
        }

        // 只分配、不初始化（元素值不确定；调用方必须保证写满后才读）
        [[nodiscard]] static std::unique_ptr<element[]> allocate_(std::size_t n)
        {
            return n == 0 ? std::unique_ptr<element[]>{}
                          : std::make_unique_for_overwrite<element[]>(n);
        }
        [[nodiscard]] const element *ptr() const noexcept { return data_.get(); }
        [[nodiscard]] element *ptr() noexcept { return data_.get(); }

        static void require_same_shape(const MatrixT &lhs, const MatrixT &rhs, [[maybe_unused]] std::string_view message)
        {
            if (lhs.rows_ != rhs.rows_ || lhs.cols_ != rhs.cols_)
            {
                NN_ASSERT(false, message.data());
            }
        }

    public:
        MatrixT() = default;

        // 零填充构造（保持既有语义：默认构造出的矩阵全零）
        explicit MatrixT(std::size_t rows, std::size_t cols)
            : data_(allocate_(rows * cols)), size_(rows * cols), rows_(rows), cols_(cols)
        {
            if (size_ != 0)
                std::fill_n(ptr(), size_, element{});
        }

        // 从标量值初始化矩阵（host 标量为 f32；F16 实例构造时舍入到 f16）
        MatrixT(std::size_t rows, std::size_t cols, element value)
            : data_(allocate_(rows * cols)), size_(rows * cols), rows_(rows), cols_(cols)
        {
            if (size_ != 0)
                std::fill_n(ptr(), size_, value);
        }

        // ── 未初始化构造（输出会被完整覆盖的路径专用）──────────────────────
        // 语义契约：调用方必须在任何读取之前把**全部** size() 个元素写满。
        // 用于把"分配 + 写满零 + 马上全覆盖"里的那一遍零写掉。
        struct uninitialized_tag {};
        MatrixT(std::size_t rows, std::size_t cols, uninitialized_tag)
            : data_(allocate_(rows * cols)), size_(rows * cols), rows_(rows), cols_(cols) {}

        [[nodiscard]] static MatrixT make_uninitialized(std::size_t rows, std::size_t cols)
        {
            return MatrixT(rows, cols, uninitialized_tag{});
        }

        // ── 拷贝（深拷贝）/ 移动（指针转移）────────────────────────────────
        // 不能用默认实现：unique_ptr 不可拷贝，且默认移动会把 size_ 留在被移对象里
        // （表现为"size 非零但 data 为空"的悬空视图）。
        MatrixT(const MatrixT &other)
            : data_(allocate_(other.size_)), size_(other.size_),
              rows_(other.rows_), cols_(other.cols_)
        {
            if (size_ != 0)
                std::copy_n(other.ptr(), size_, ptr());
        }
        MatrixT(MatrixT &&other) noexcept
            : data_(std::move(other.data_)), size_(other.size_),
              rows_(other.rows_), cols_(other.cols_)
        {
            other.size_ = 0;
            other.rows_ = 0;
            other.cols_ = 0;
        }
        MatrixT &operator=(const MatrixT &other)
        {
            if (this != &other)
            {
                MatrixT tmp(other);
                *this = std::move(tmp);
            }
            return *this;
        }
        MatrixT &operator=(MatrixT &&other) noexcept
        {
            if (this != &other)
            {
                data_ = std::move(other.data_);
                size_ = other.size_;
                rows_ = other.rows_;
                cols_ = other.cols_;
                other.size_ = 0;
                other.rows_ = 0;
                other.cols_ = 0;
            }
            return *this;
        }
        ~MatrixT() = default;

        // ── 就地调整大小（复用已有内存；新增部分零填充，与原 vector 语义一致）──
        void resize(std::size_t rows, std::size_t cols)
        {
            if (rows_ == rows && cols_ == cols) return; // 尺寸不变，零开销
            const std::size_t n = rows * cols;
            auto nd = allocate_(n);
            if (nd && size_ != 0)
                std::copy_n(ptr(), std::min(size_, n), nd.get());
            if (nd && n > size_)
                std::fill_n(nd.get() + size_, n - size_, element{});
            data_ = std::move(nd);
            size_ = n;
            rows_ = rows;
            cols_ = cols;
        }

        // ── std::span 访问（C++20 现代接口，推荐使用） ────────────────────
        // 零开销抽象：编译后等价于裸指针 + 大小，可替代所有 data_ptr() 场景
        [[nodiscard]] std::span<const element> span() const
        {
            return {ptr(), size_};
        }
        [[nodiscard]] std::span<element> span() noexcept { return {ptr(), size_}; }

        // 访问器
        [[nodiscard]] constexpr std::size_t rows() const noexcept { return rows_; }
        [[nodiscard]] constexpr std::size_t cols() const noexcept { return cols_; }
        [[nodiscard]] constexpr std::size_t size() const noexcept { return size_; }
        [[nodiscard]] constexpr bool empty() const noexcept { return size_ == 0; }
        [[nodiscard]] element at(std::size_t row, std::size_t col) const
        {
            if (row >= rows_ || col >= cols_)
            {
                NN_ASSERT(false, "Matrix index out of range");
            }
            return ptr()[index(row, col)];
        }
        void set_value(std::size_t row, std::size_t col, element value)
        {
            if (row >= rows_ || col >= cols_)
            {
                NN_ASSERT(false, "Matrix index out of range");
            }
            ptr()[index(row, col)] = value;
        }
        [[nodiscard]] element at_unchecked(std::size_t row, std::size_t col) const noexcept { return ptr()[index(row, col)]; } // 无校验
        void set_value_unchecked(std::size_t row, std::size_t col, element value) noexcept { ptr()[index(row, col)] = value; } // 无校验

        // ── 转置（返回新矩阵） ─────────────────────────────────────────────
        [[nodiscard]] MatrixT transpose() const
        {
            MatrixT result(cols_, rows_);
            transpose_to(result);
            return result;
        }

        // ── 转置到预分配缓冲区（零分配热路径） ─────────────────────────────
        void transpose_to(MatrixT &result) const
        {
            NN_ASSERT(&result != this, "transpose_to: self-referencing not supported");
            result.resize(cols_, rows_);
            if (rows_ == 0 || cols_ == 0) return;

            const auto src = span();
            auto dst = result.span();
            const std::size_t R = rows_;
            const std::size_t C = cols_;

            const std::size_t i_blocks = (R + BLOCK_SIZE - 1) / BLOCK_SIZE;
            const std::size_t j_blocks = (C + BLOCK_SIZE - 1) / BLOCK_SIZE;

            // 通过 nn::parallel_for_blocks 分发块级并行：分块数很少（1~几十），但其内循环计算量大。
            const auto n_blocks = i_blocks * j_blocks;
            if (n_blocks <= 1)
            {
                for (std::size_t i = 0; i < R; ++i)
                    for (std::size_t j = 0; j < C; ++j)
                        dst[j * R + i] = src[i * C + j];
            }
            else
            {
                auto block_indices = std::views::iota(std::size_t{0}, n_blocks);
                nn::parallel_for_blocks(
                    block_indices.begin(), block_indices.end(),
                    [src, dst, R, C, j_blocks](std::size_t block_idx) noexcept
                    {
                        const std::size_t ib = block_idx / j_blocks;
                        const std::size_t jb = block_idx % j_blocks;
                        const std::size_t i0 = ib * BLOCK_SIZE;
                        const std::size_t j0 = jb * BLOCK_SIZE;
                        const std::size_t i1 = std::min(i0 + BLOCK_SIZE, R);
                        const std::size_t j1 = std::min(j0 + BLOCK_SIZE, C);
                        for (std::size_t i = i0; i < i1; ++i)
                            for (std::size_t j = j0; j < j1; ++j)
                                dst[j * R + i] = src[i * C + j];
                    });
            }
        }

        // ── GEMM 微内核：NI 行 × RB 列寄存器分块 ───────────────────────────
        // 为何这样写（实测依据）：
        //   旧内核每个 (i,j) 用一个标量累加器、每次 FMA 要 2 次 load（受 load port
        //   限制），且编译日志显示内层循环**未被向量化**（-Wpass-failed=transform-
        //   warning）。本微内核一次处理 4 行 × 8 列（一个 AVX2 向量宽度的列块）：
        //     · B 以 k-major 打包（b_pack[kk * j_len + j]）→ 固定 kk 时 j 连续，
        //       j 方向是一次连续向量 load；
        //     · 每个 k 只从 A 广播 4 个标量、做 4 次向量 FMA → FMA/load 比大幅改善；
        //     · k 维走**编译期常量** BLOCK_SIZE → 循环可完整向量化/展开。
        //   实测 GFLOPS（3072x768x512 / 768x3072x512 / 1024^3 / 512^3）：
        //   140→401 / 155→407 / 163→354 / 82→281。
        //   累加分组不变（块内 kk 升序、块间 k_start 升序）→ 与旧内核**逐位一致**
        //   （对拍 max_abs_diff = 0）。
        static constexpr std::size_t MK_NI = 4;   // 微内核行数
        static constexpr std::size_t MK_RB = 8;   // 微内核列数

        // ap：A 行块首地址（第 ii 行为 ap + ii * a_stride）；
        // bp：k-major 打包块（步长 j_len）；rp：输出行块首地址（行步长 r_stride）。
        // NI 用**模板参数**（编译期常量）：运行时传 NI 会让 ii 循环无法展开、
        // 累加器可能落内存，实测主路径慢 ~1.7x。
        template <std::size_t NI>
        static void gemm_microkernel_(const element *ap, std::size_t a_stride,
                                      const element *bp, std::size_t j_len,
                                      element *rp, std::size_t r_stride) noexcept
        {
            std::array<std::array<acc_type, MK_RB>, NI> acc{};
            for (std::size_t kk = 0; kk < BLOCK_SIZE; ++kk)
            {
                const element *brow = bp + kk * j_len;
                for (std::size_t ii = 0; ii < NI; ++ii)
                {
                    const acc_type av = static_cast<acc_type>(ap[ii * a_stride + kk]);
                    for (std::size_t jj = 0; jj < MK_RB; ++jj)
                        acc[ii][jj] += av * static_cast<acc_type>(brow[jj]);
                }
            }
            for (std::size_t ii = 0; ii < NI; ++ii)
                for (std::size_t jj = 0; jj < MK_RB; ++jj)
                    rp[ii * r_stride + jj] += static_cast<element>(acc[ii][jj]);
        }

        // ── 定长点积（尾块 / 窄 N 专用）─────────────────────────────────
        // 必须独立成函数：NN_VECTORIZE_PRAGMA（loop vectorize(assume_safety)）
        // 若与其它循环同处一个函数，会让整函数的向量化一起失败（实测同结构
        // 374 -> 95 GFLOPS）。尾块在窄 N（如 Linear 的 batch=1，N=1）时是唯一
        // 路径，这一段不隔离就会退化。
        static acc_type dot_span_(const element *x, std::size_t x_stride,
                                  const element *y, std::size_t y_stride,
                                  std::size_t len) noexcept
        {
            acc_type s = acc_type{0};
            NN_VECTORIZE_PRAGMA
            for (std::size_t k = 0; k < len; ++k)
                s += static_cast<acc_type>(x[k * x_stride]) * static_cast<acc_type>(y[k * y_stride]);
            return s;
        }

        // ── 基于 span 的矩阵乘法（零拷贝，供 batched_matmul 等场景使用） ──
        // 从 a/b 的子区间直接计算，无需构造临时 Matrix 拷贝
        static void multiply_to_span(
            std::span<element> r, std::size_t M, std::size_t N,
            std::span<const element> a, std::size_t /*a_rows*/, std::size_t a_cols,
            std::span<const element> b, std::size_t b_rows, std::size_t b_cols)
        {
            NN_ASSERT(a_cols == b_rows, "multiply_to_span: inner dimension mismatch");
            (void)b_rows;  // NN_ASSERT 在 Release 模式下展开为空，参数仅用于断言
            const std::size_t K = a_cols;
            if (M == 0 || N == 0 || K == 0) return;

            // ── 窄 N 专用路径（N < MK_RB）────────────────────────────────
            // 典型是 Linear 的 batch=1（N=1）。主路径此处恒走尾块，且每个
            // (输出块, k 步) 都要把 B 打包进 16KB 栈块（值初始化）——GEMV 的
            // 计算量与之同量级，打包的零填充把内存流量翻倍。窄 N 下 B 很小
            // （N*K），直接按步长读、完全不打包更快。
            if (N < MK_RB)
            {
                const std::size_t nb = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
                nn::parallel_for_samples(nb, [a, b, r, M, N, K, b_cols](std::size_t ib) noexcept
                {
                    const std::size_t i0 = ib * BLOCK_SIZE;
                    const std::size_t i1 = std::min(i0 + BLOCK_SIZE, M);
                    for (std::size_t i = i0; i < i1; ++i)
                        for (std::size_t j = 0; j < N; ++j)
                            r[i * N + j] += dot_span_(a.data() + i * K, 1,
                                                      b.data() + j, b_cols, K);
                });
                return;
            }

            // 直接复用 blocked matmul 内核
            const std::size_t i_blocks = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
            const std::size_t j_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
            const auto n_blocks = i_blocks * j_blocks;

            auto kernel = [a, b, r, M, N, K, a_cols, b_cols, j_blocks](std::size_t block_idx) noexcept
            {
                const std::size_t i_block = block_idx / j_blocks;
                const std::size_t j_block = block_idx % j_blocks;
                const std::size_t i_start = i_block * BLOCK_SIZE;
                const std::size_t i_end = std::min(i_start + BLOCK_SIZE, M);
                const std::size_t j_start = j_block * BLOCK_SIZE;
                const std::size_t j_end = std::min(j_start + BLOCK_SIZE, N);
                for (std::size_t k_start = 0; k_start < K; k_start += BLOCK_SIZE)
                {
                    const std::size_t k_end = std::min(k_start + BLOCK_SIZE, K);
                    const std::size_t k_len = k_end - k_start;
                    const std::size_t j_len = j_end - j_start;
                    // B 打包成 k-major：b_pack[kk * j_len + j]（固定 kk 时 j 连续）
                    std::array<element, BLOCK_SIZE * BLOCK_SIZE> b_pack{};
                    for (std::size_t kk = 0; kk < k_len; ++kk)
                        for (std::size_t j = 0; j < j_len; ++j)
                            b_pack[kk * j_len + j] = b[(k_start + kk) * b_cols + (j_start + j)];

                    for (std::size_t i0 = i_start; i0 < i_end; i0 += MK_NI)
                    {
                        const std::size_t ni = std::min(MK_NI, i_end - i0);
                        const element *ap = a.data() + i0 * a_cols + k_start;
                        for (std::size_t j0 = 0; j0 < j_len; j0 += MK_RB)
                        {
                            const std::size_t nj = std::min(MK_RB, j_len - j0);
                            element *rp = r.data() + i0 * N + j_start + j0;
                            if (ni == MK_NI && nj == MK_RB && k_len == BLOCK_SIZE)
                            {
                                gemm_microkernel_<MK_NI>(ap, a_cols, b_pack.data() + j0,
                                                         j_len, rp, N);
                            }
                            else
                            {
                                // 尾块：kk 放在**最内层** → 归约可向量化（与旧内核同构）。
                                // 对 N 很小（典型是 Linear 的 batch=1，N=1）这是关键路径：
                                // 若把 kk 放外层，jj 只有 1 次迭代，归约无法向量化 → 慢 2x。
                                for (std::size_t ii = 0; ii < ni; ++ii)
                                    for (std::size_t jj = 0; jj < nj; ++jj)
                                    {
                                        acc_type sum = acc_type{0};
                                        NN_VECTORIZE_PRAGMA
                                        for (std::size_t kk = 0; kk < k_len; ++kk)
                                            sum += static_cast<acc_type>(ap[ii * a_cols + kk])
                                                 * static_cast<acc_type>(b_pack[kk * j_len + j0 + jj]);
                                        rp[ii * N + jj] += static_cast<element>(sum);
                                    }
                            }
                        }
                    }
                }
            };

            if (n_blocks <= 1)
                kernel(0);
            else
            {
                auto block_indices = std::views::iota(std::size_t{0}, n_blocks);
                nn::parallel_for_blocks(block_indices.begin(), block_indices.end(), kernel);
            }
        }

        // ── 基于 span 的矩阵乘法（B 转置，零拷贝） ─────────────────────────
        static void multiply_transposed_to_span(
            std::span<element> r, std::size_t M, std::size_t N,
            std::span<const element> a, std::size_t /*a_rows*/, std::size_t a_cols,
            std::span<const element> bt, std::size_t /*bt_rows*/, std::size_t bt_cols)
        {
            NN_ASSERT(a_cols == bt_cols, "multiply_transposed_to_span: inner dimension mismatch");
            (void)bt_cols;  // NN_ASSERT 在 Release 模式下展开为空，参数仅用于断言
            const std::size_t K = a_cols;
            if (M == 0 || N == 0 || K == 0) return;

            // 窄 N 专用路径（见 multiply_to_span 处的说明）。此形态 B^T 存为
            // (N, K)，固定 n 时 k 连续 → 点积两侧都连续。
            if (N < MK_RB)
            {
                const std::size_t nb = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
                nn::parallel_for_samples(nb, [a, bt, r, M, N, K](std::size_t ib) noexcept
                {
                    const std::size_t i0 = ib * BLOCK_SIZE;
                    const std::size_t i1 = std::min(i0 + BLOCK_SIZE, M);
                    for (std::size_t i = i0; i < i1; ++i)
                        for (std::size_t j = 0; j < N; ++j)
                            r[i * N + j] += dot_span_(a.data() + i * K, 1,
                                                      bt.data() + j * K, 1, K);
                });
                return;
            }

            const std::size_t i_blocks = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
            const std::size_t j_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
            const auto n_blocks = i_blocks * j_blocks;

            auto kernel = [a, bt, r, M, N, K, j_blocks](std::size_t block_idx) noexcept
            {
                const std::size_t i_block = block_idx / j_blocks;
                const std::size_t j_block = block_idx % j_blocks;
                const std::size_t i_start = i_block * BLOCK_SIZE;
                const std::size_t i_end = std::min(i_start + BLOCK_SIZE, M);
                const std::size_t j_start = j_block * BLOCK_SIZE;
                const std::size_t j_end = std::min(j_start + BLOCK_SIZE, N);
                for (std::size_t k_start = 0; k_start < K; k_start += BLOCK_SIZE)
                {
                    const std::size_t k_len = std::min(BLOCK_SIZE, K - k_start);
                    const std::size_t j_len = j_end - j_start;
                    // B^T 打包成 k-major（固定 kk 时 j 连续）：
                    //   b_pack[kk * j_len + j] = bt[(j_start + j) * K + (k_start + kk)]
                    std::array<element, BLOCK_SIZE * BLOCK_SIZE> b_pack{};
                    for (std::size_t kk = 0; kk < k_len; ++kk)
                        for (std::size_t j = 0; j < j_len; ++j)
                            b_pack[kk * j_len + j] = bt[(j_start + j) * K + (k_start + kk)];

                    for (std::size_t i0 = i_start; i0 < i_end; i0 += MK_NI)
                    {
                        const std::size_t ni = std::min(MK_NI, i_end - i0);
                        const element *ap = a.data() + i0 * K + k_start;
                        for (std::size_t j0 = 0; j0 < j_len; j0 += MK_RB)
                        {
                            const std::size_t nj = std::min(MK_RB, j_len - j0);
                            element *rp = r.data() + i0 * N + j_start + j0;
                            if (ni == MK_NI && nj == MK_RB && k_len == BLOCK_SIZE)
                            {
                                gemm_microkernel_<MK_NI>(ap, K, b_pack.data() + j0, j_len,
                                                         rp, N);
                            }
                            else
                            {
                                // 尾块：kk 在最内层 → 归约可向量化
                                for (std::size_t ii = 0; ii < ni; ++ii)
                                    for (std::size_t jj = 0; jj < nj; ++jj)
                                    {
                                        acc_type sum = acc_type{0};
                                        NN_VECTORIZE_PRAGMA
                                        for (std::size_t kk = 0; kk < k_len; ++kk)
                                            sum += static_cast<acc_type>(ap[ii * K + kk])
                                                 * static_cast<acc_type>(b_pack[kk * j_len + j0 + jj]);
                                        rp[ii * N + jj] += static_cast<element>(sum);
                                    }
                            }
                        }
                    }
                }
            };

            if (n_blocks <= 1)
                kernel(0);
            else
            {
                auto block_indices = std::views::iota(std::size_t{0}, n_blocks);
                nn::parallel_for_blocks(block_indices.begin(), block_indices.end(), kernel);
            }
        }

        // ── 基于 span 的矩阵乘法（A 转置，零拷贝） ─────────────────────────
        static void transpose_multiply_to_span(
            std::span<element> r, std::size_t M, std::size_t N,
            std::span<const element> a, std::size_t a_rows, std::size_t a_cols,
            std::span<const element> b, std::size_t b_rows, std::size_t b_cols)
        {
            NN_ASSERT(a_rows == b_rows, "transpose_multiply_to_span: inner dimension mismatch");
            (void)b_rows;  // NN_ASSERT 在 Release 模式下展开为空，参数仅用于断言
            const std::size_t K = a_rows;  // a is (K, M) stored
            if (M == 0 || N == 0 || K == 0) return;

            // 窄 N 专用路径（见 multiply_to_span 处的说明）。此形态 A 存为 (K, M)
            // → 固定 m 时以 a_cols 为步长读列（这是本形态的固有代价，无法避免）。
            if (N < MK_RB)
            {
                const std::size_t nb = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
                nn::parallel_for_samples(nb, [a, b, r, M, N, K, a_cols, b_cols](std::size_t ib) noexcept
                {
                    const std::size_t i0 = ib * BLOCK_SIZE;
                    const std::size_t i1 = std::min(i0 + BLOCK_SIZE, M);
                    for (std::size_t i = i0; i < i1; ++i)
                        for (std::size_t j = 0; j < N; ++j)
                            r[i * N + j] += dot_span_(a.data() + i, a_cols,
                                                      b.data() + j, b_cols, K);
                });
                return;
            }

            const std::size_t i_blocks = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
            const std::size_t j_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
            const auto n_blocks = i_blocks * j_blocks;

            auto kernel = [a, b, r, M, N, K, a_cols, b_cols, j_blocks](std::size_t block_idx) noexcept
            {
                const std::size_t i_block = block_idx / j_blocks;
                const std::size_t j_block = block_idx % j_blocks;
                const std::size_t i_start = i_block * BLOCK_SIZE;
                const std::size_t i_end = std::min(i_start + BLOCK_SIZE, M);
                const std::size_t j_start = j_block * BLOCK_SIZE;
                const std::size_t j_end = std::min(j_start + BLOCK_SIZE, N);
                for (std::size_t k_start = 0; k_start < K; k_start += BLOCK_SIZE)
                {
                    const std::size_t k_len = std::min(BLOCK_SIZE, K - k_start);
                    const std::size_t j_len = j_end - j_start;
                    const std::size_t i_len = i_end - i_start;
                    // A 存储为 (K, M)：按 (i, k) 打包成行内 k 连续的块，
                    // 微内核以 k_len 为行步长读它（避免 stride=a_cols 的跨行访问）
                    std::array<element, BLOCK_SIZE * BLOCK_SIZE> a_pack{};
                    for (std::size_t ii = 0; ii < i_len; ++ii)
                        for (std::size_t kk = 0; kk < k_len; ++kk)
                            a_pack[ii * k_len + kk] = a[(k_start + kk) * a_cols + (i_start + ii)];
                    // B 打包成 k-major（固定 kk 时 j 连续）
                    std::array<element, BLOCK_SIZE * BLOCK_SIZE> b_pack{};
                    for (std::size_t kk = 0; kk < k_len; ++kk)
                        for (std::size_t j = 0; j < j_len; ++j)
                            b_pack[kk * j_len + j] = b[(k_start + kk) * b_cols + (j_start + j)];

                    for (std::size_t ii0 = 0; ii0 < i_len; ii0 += MK_NI)
                    {
                        const std::size_t ni = std::min(MK_NI, i_len - ii0);
                        const element *ap = a_pack.data() + ii0 * k_len;
                        element *rbase = r.data() + (i_start + ii0) * N + j_start;
                        for (std::size_t j0 = 0; j0 < j_len; j0 += MK_RB)
                        {
                            const std::size_t nj = std::min(MK_RB, j_len - j0);
                            element *rp = rbase + j0;
                            if (ni == MK_NI && nj == MK_RB && k_len == BLOCK_SIZE)
                            {
                                gemm_microkernel_<MK_NI>(ap, k_len, b_pack.data() + j0, j_len,
                                                         rp, N);
                            }
                            else
                            {
                                // 尾块：kk 在最内层 → 归约可向量化
                                for (std::size_t ii = 0; ii < ni; ++ii)
                                    for (std::size_t jj = 0; jj < nj; ++jj)
                                    {
                                        acc_type sum = acc_type{0};
                                        NN_VECTORIZE_PRAGMA
                                        for (std::size_t kk = 0; kk < k_len; ++kk)
                                            sum += static_cast<acc_type>(ap[ii * k_len + kk])
                                                 * static_cast<acc_type>(b_pack[kk * j_len + j0 + jj]);
                                        rp[ii * N + jj] += static_cast<element>(sum);
                                    }
                            }
                        }
                    }
                }
            };

            if (n_blocks <= 1)
                kernel(0);
            else
            {
                auto block_indices = std::views::iota(std::size_t{0}, n_blocks);
                nn::parallel_for_blocks(block_indices.begin(), block_indices.end(), kernel);
            }
        }

        // ── 矩阵乘法到预分配缓冲区（零分配热路径） ─────────────────────────
        // 使用 std::span（C++20）提供类型安全的非拥有视图
        void multiply_to(MatrixT &result, const MatrixT &other) const
        {
            NN_ASSERT(&result != this, "multiply_to: self-referencing not supported");
            const std::size_t M = rows_;
            const std::size_t N = other.cols_;
            const std::size_t K = cols_;
            result.resize(M, N);
            if (M == 0 || N == 0 || K == 0) return;

            // 清零结果矩阵（使用 RAII 封装的方法）
            result.zero();

            // 委托给 span 版内核：GEMM 内核只保留一份，避免两处实现漂移
            multiply_to_span(result.span(), M, N, span(), M, K, other.span(), K, N);
        }

        // ── 矩阵乘法（B 转置）到预分配缓冲区 ─────────────────────────
        // 计算 result = this * B^T，其中 B 存储为 (N, K) 行主序
        // 无需实际转置 B，直接从 B 的行读取列（节省 O(N*K) 的拷贝）
        // 
        // 维度要求：this=(M,K), b_trans=(N,K) → result=(M,N)
        // 即 C[m][n] = Σ_k A[m][k] * B[n][k]
        // ── 矩阵乘法（B 转置）到预分配缓冲区 ─────────────────────────
        // result = this * B^T，其中 B 存储为 (N, K) 行主序；C[m][n] = Σ_k A[m][k] * B[n][k]。
        // 委托给 span 版内核：GEMM 内核只保留一份，避免两处实现漂移。
        void multiply_transposed_to(MatrixT &result, const MatrixT &b_trans) const
        {
            NN_ASSERT(&result != this && &result != &b_trans, "multiply_transposed_to: self-referencing not supported");
            NN_ASSERT(cols_ == b_trans.cols_, "multiply_transposed_to: inner dimensions mismatch");
            const std::size_t M = rows_;
            const std::size_t K = cols_;
            const std::size_t N = b_trans.rows_;
            result.resize(M, N);
            if (M == 0 || N == 0 || K == 0) return;
            result.zero();
            multiply_transposed_to_span(result.span(), M, N, span(), M, K,
                                        b_trans.span(), N, K);
        }

        // ── 矩阵乘法（A 转置）到预分配缓冲区 ─────────────────────────
        // 计算 result = this^T * B，其中 this 存储为 (K, M) 行主序
        // 无需实际转置 this，直接从 this 的列读取行（节省 O(K*M) 的拷贝）
        //
        // 维度要求：this=(K,M), b=(K,N) → result=(M,N)
        // 即 C[m][n] = Σ_k A[k][m] * B[k][n]
        // 委托给 span 版内核：GEMM 内核只保留一份，避免两处实现漂移。
        void transpose_multiply_to(MatrixT &result, const MatrixT &b) const
        {
            NN_ASSERT(&result != this && &result != &b, "transpose_multiply_to: self-referencing not supported");
            NN_ASSERT(rows_ == b.rows_, "transpose_multiply_to: inner dimensions mismatch");
            const std::size_t K = rows_;  // this is (K, M)
            const std::size_t M = cols_;
            const std::size_t N = b.cols_;
            result.resize(M, N);
            if (M == 0 || N == 0 || K == 0) return;
            result.zero();
            transpose_multiply_to_span(result.span(), M, N, span(), K, cols_,
                                       b.span(), K, b.cols());
        }

        // ── 累加矩阵乘法（A * B^T，结果累加到 result） ─────────────
        // 计算 result += this * B^T
        // 用于梯度累加：grad_w += grad_output * input^T
        void multiply_transposed_add_to(MatrixT &result, const MatrixT &b_trans) const
        {
            NN_ASSERT(&result != this && &result != &b_trans, "multiply_transposed_add_to: self-referencing");
            NN_ASSERT(cols_ == b_trans.cols_, "multiply_transposed_add_to: inner dimensions mismatch");
            NN_ASSERT(result.rows() == rows_ && result.cols() == b_trans.rows_, "multiply_transposed_add_to: result shape mismatch");
            const std::size_t M = rows_;
            const std::size_t K = cols_;
            const std::size_t N = b_trans.rows_;
            if (M == 0 || N == 0 || K == 0) return;

            // span 内核本身就是"累加到 r"的语义（不清零）→ 语义完全一致，
            // 直接委托。注意：本方法全库无调用者（死代码），一并消除第三份重复内核。
            multiply_transposed_to_span(result.span(), M, N, span(), M, K,
                                        b_trans.span(), N, K);
        }

        void scale_inplace(Scalar scalar) noexcept
        {
            auto s = span();
            nn::for_each(s.begin(), s.end(),
                           [scalar](element &value) noexcept { value *= scalar; });
        }

        // 逐元素加法 inplace
        void add_inplace(const MatrixT &other)
        {
            require_same_shape(*this, other, "add_inplace dimension mismatch");
            auto s = span();
            auto o = other.span();
            nn::transform(s.begin(), s.end(), o.begin(),
                           s.begin(), std::plus<>{});
        }

        // 填充零
        void zero() noexcept
        {
            if (size_ != 0)
                std::fill_n(ptr(), size_, element{});
        }

        // ── 归约操作 ────────────────────────────────────────────────────
        // result = reduce_op(init, transform_op(data[0]), transform_op(data[1]), ...)
        template <typename T, typename ReduceOp, typename TransformOp>
        [[nodiscard]] T reduce(T init, ReduceOp&& reduce_op, TransformOp&& transform_op) const
        {
            auto s = span();
            return nn::transform_reduce(s.begin(), s.end(), init,
                std::forward<ReduceOp>(reduce_op), std::forward<TransformOp>(transform_op));
        }

        // ── 按行归约（通用数学原语，不是算法） ──────────────────────────
        // 对每一行独立归约，返回 (rows, 1) 矩阵。
        //   result[r][0] = reduce_op(init, transform_op(this[r][0]), ..., transform_op(this[r][cols-1]))
        // 上层可基于此表达 softmax 行最大值/行求和、按行范数等算法。
        template <typename T, typename ReduceOp, typename TransformOp>
        [[nodiscard]] MatrixT row_reduce(T init, ReduceOp&& reduce_op, TransformOp&& transform_op) const
        {
            MatrixT result(rows_, 1);
            if (rows_ == 0) return result;

            const auto self = span();
            auto out = result.span();
            const std::size_t C = cols_;

            auto process_row = [self, out, C, init,
                                reduce_op = std::forward<ReduceOp>(reduce_op),
                                transform_op = std::forward<TransformOp>(transform_op)](std::size_t r) noexcept {
                const auto row = self.subspan(r * C, C);
                T acc = init;
                for (std::size_t c = 0; c < C; ++c)
                    acc = reduce_op(acc, transform_op(row[c]));
                out[r] = static_cast<element>(acc);
            };

            // 并行门控按**元素数**（R*C），与 broadcast_* 一致。
            // 旧实现用 nn::for_each(row_indices)：它把"行数"当元素数与
            // PARALLEL_THRESHOLD 比较 → 行数永远达不到 512K → **恒定串行**
            // （见 docs/development/11 §R3）。改为按行分片。
            // 每行独立累加、行内顺序不变 → 并行与串行逐字节一致（铁律 8）。
            if (rows_ * cols_ >= PARALLEL_THRESHOLD && rows_ > 1)
                nn::parallel_for_samples(rows_, process_row);
            else
                for (std::size_t r = 0; r < rows_; ++r)
                    process_row(r);
            return result;
        }

        // ── 按列归约（通用数学原语，不是算法） ──────────────────────────
        // 对每一列独立归约，返回 (1, cols) 矩阵。
        //   result[0][c] = reduce_op(init, transform_op(this[0][c]), ..., transform_op(this[rows-1][c]))
        // 上层可基于此表达 LayerNorm 列均值/列方差等算法。
        //
        // 实现策略：cache-friendly blocked + 行块并行。
        // bench_thresholds 实测：blocked 全面优于 naive（按列跨行扫描），
        // 行块并行仅在 R >= 256 且 R*C >= PARALLEL_THRESHOLD 时启用，
        // 详见 bench_thresholds.cpp 测试 2/3。
        template <typename T, typename ReduceOp, typename TransformOp>
        [[nodiscard]] MatrixT col_reduce(T init, ReduceOp&& reduce_op, TransformOp&& transform_op) const
        {
            MatrixT result(1, cols_);
            if (cols_ == 0) return result;

            const auto self = span();
            auto out = result.span();
            const std::size_t R = rows_;
            const std::size_t C = cols_;

            // 极小矩阵直接 naive（避免清零开销）
            if (R * C < 64)
            {
                for (std::size_t c = 0; c < C; ++c)
                {
                    T acc = init;
                    for (std::size_t r = 0; r < R; ++r)
                        acc = reduce_op(acc, transform_op(self[r * C + c]));
                    out[c] = static_cast<element>(acc);
                }
                return result;
            }

            // 行块并行启用条件：R >= COL_REDUCE_PARALLEL_ROWS 且 R*C >= PARALLEL_THRESHOLD。
            // 门槛由 1024 降至 256：R*C >= 512K 时即使 R=256 每线程也有 >=16K 元素
            // 的工作量（32 线程假设），同步开销不占主导；256 覆盖常见 d_model=768 场景。
            constexpr std::size_t COL_REDUCE_PARALLEL_ROWS = 256;      // 行数门槛
            const std::size_t hw_threads = std::thread::hardware_concurrency();
            const std::size_t n_threads = (hw_threads == 0) ? 1 : hw_threads;
            const bool use_parallel =
                R >= COL_REDUCE_PARALLEL_ROWS &&
                R * C >= PARALLEL_THRESHOLD &&
                n_threads > 1;

            if (!use_parallel)
            {
                // ── 单线程行主序扫描 ──
                for (std::size_t c = 0; c < C; ++c)
                    out[c] = static_cast<element>(init);
                for (std::size_t r = 0; r < R; ++r)
                {
                    // 行视图：std::span::subspan 零成本（ptr+len），替代裸指针行起点
                    const auto row = self.subspan(r * C, C);
                    for (std::size_t c = 0; c < C; ++c)
                    {
                        element v = static_cast<element>(transform_op(row[c]));
                        out[c] = static_cast<element>(reduce_op(static_cast<T>(out[c]), v));
                    }
                }
                return result;
            }

            // ── 行块并行路径 ──
            // 分配 n_threads 组本地累加器（连续存储，cache 友好）
            std::vector<T> local_acc(n_threads * C);
            for (std::size_t t = 0; t < n_threads; ++t)
                for (std::size_t c = 0; c < C; ++c)
                    local_acc[t * C + c] = init;

            // 按行分块并行扫描
            auto& pool = global_thread_pool();
            const std::size_t base = R / n_threads;
            const std::size_t rem = R % n_threads;
            auto row_blocks = std::views::iota(std::size_t{0}, n_threads);
            pool.parallel_for_blocks(row_blocks.begin(), row_blocks.end(),
                [self, &local_acc, &reduce_op, &transform_op, C, base, rem](std::size_t t) noexcept {
                    const std::size_t r0 = t * base + std::min(t, rem);
                    const std::size_t r_end = (t + 1) * base + std::min(t + 1, rem);
                    // 本线程累加器行视图（lambda 体内局部，捕获语义不受影响）
                    auto acc = std::span(local_acc).subspan(t * C, C);
                    for (std::size_t r = r0; r < r_end; ++r)
                    {
                        const auto row = self.subspan(r * C, C);
                        for (std::size_t c = 0; c < C; ++c)
                        {
                            element v = static_cast<element>(transform_op(row[c]));
                            acc[c] = reduce_op(acc[c], v);
                        }
                    }
                });

            // 归并阶段：串行合并 n_threads 组累加器到 out[c]
            // 第 0 组直接写入，其余组归并进来（reduce_op 满足结合律，结果与单线程一致）
            for (std::size_t c = 0; c < C; ++c)
                out[c] = static_cast<element>(local_acc[c]);  // 组 0
            for (std::size_t t = 1; t < n_threads; ++t)
            {
                const auto acc = std::span(local_acc).subspan(t * C, C);
                for (std::size_t c = 0; c < C; ++c)
                    out[c] = static_cast<element>(reduce_op(static_cast<T>(out[c]), acc[c]));
            }
            return result;
        }

        // ── 按行广播（通用数学原语，不是算法） ──────────────────────────
        // this[r][c] = op(this[r][c], row_vec[r][0])，row_vec 形状必须为 (rows_, 1)
        // 上层可基于此表达 softmax 减行最大值、除行求和等算法。
        template <typename F>
        void broadcast_row_inplace(const MatrixT& row_vec, F&& op)
        {
            NN_ASSERT(row_vec.rows_ == rows_ && row_vec.cols_ == 1, "row_vec shape mismatch");
            const auto v = row_vec.span();
            const std::size_t R = rows_;
            const std::size_t C = cols_;
            auto d = span();
            // 按行处理：v[r] 每行只取一次（替代逐元素 i/C 除法），行内连续访问可向量化。
            // 并行阈值与旧实现一致（元素数 >= PARALLEL_THRESHOLD），仅并行粒度由元素改为行。
            auto process_row = [&d, &v, C, op = std::forward<F>(op)](std::size_t r) noexcept {
                const element vr = v[r];
                auto row = d.subspan(r * C, C);
                for (std::size_t c = 0; c < C; ++c)
                    row[c] = static_cast<element>(op(row[c], vr));
            };
            if (R * C >= PARALLEL_THRESHOLD && R > 1)
                nn::parallel_for_samples(R, process_row);
            else
                for (std::size_t r = 0; r < R; ++r)
                    process_row(r);
        }

        // ── 按列广播（通用数学原语，不是算法） ──────────────────────────
        // this[r][c] = op(this[r][c], col_vec[0][c])，col_vec 形状必须为 (1, cols_)
        // 上层可基于此表达 LayerNorm 减列均值、乘列标准差等算法。
        template <typename F>
        void broadcast_col_inplace(const MatrixT& col_vec, F&& op)
        {
            NN_ASSERT(col_vec.rows_ == 1 && col_vec.cols_ == cols_, "col_vec shape mismatch");
            const auto v = col_vec.span();
            const std::size_t R = rows_;
            const std::size_t C = cols_;
            auto d = span();
            // 按行处理：行内直接用 v[c]（替代逐元素 i%C 取模），行内连续访问可向量化。
            // 并行阈值与旧实现一致（元素数 >= PARALLEL_THRESHOLD），仅并行粒度由元素改为行。
            auto process_row = [&d, &v, C, op = std::forward<F>(op)](std::size_t r) noexcept {
                auto row = d.subspan(r * C, C);
                for (std::size_t c = 0; c < C; ++c)
                    row[c] = static_cast<element>(op(row[c], v[c]));
            };
            if (R * C >= PARALLEL_THRESHOLD && R > 1)
                nn::parallel_for_samples(R, process_row);
            else
                for (std::size_t r = 0; r < R; ++r)
                    process_row(r);
        }
    };

    // ── 便捷类型别名 ────────────────────────────────────────────────────
    using Matrix = MatrixT<Precision::F32>;
    using MatrixF32 = MatrixT<Precision::F32>;
    using MatrixF16 = MatrixT<Precision::F16>;

    // ═══════════════════════════════════════════════════════════════════════
    // detail 命名空间：逐元素变换的自由函数（原 Matrix 成员方法）
    //
    // 这些函数仅是 nn::transform 的薄包装，放在 detail 命名空间而非 Matrix 类内，
    // 避免 Matrix 接口膨胀。compute_bench 等基准测试直接调用。
    // ═══════════════════════════════════════════════════════════════════════
    namespace detail
    {
        // ── 逐元素一元变换（返回新矩阵） ────────────────────────────────
        // out[i] = func(in[i])，内部自动选择串行/并行。
        template <Precision P, typename F>
        [[nodiscard]] MatrixT<P> apply(const MatrixT<P>& mat, F&& func)
        {
            MatrixT<P> result(mat.rows(), mat.cols());
            auto s = mat.span();
            auto r = result.span();
            nn::transform(s.begin(), s.end(),
                           r.begin(), std::forward<F>(func));
            return result;
        }

        // ── 逐元素二元变换（返回新矩阵） ────────────────────────────────
        // out[i] = func(a[i], b[i])
        template <Precision P, typename F>
        [[nodiscard]] MatrixT<P> binary_apply(const MatrixT<P>& a, const MatrixT<P>& b, F&& func)
        {
            NN_ASSERT(a.rows() == b.rows() && a.cols() == b.cols(),
                       "binary_apply dimension mismatch");
            MatrixT<P> result(a.rows(), a.cols());
            auto s = a.span();
            auto o = b.span();
            auto r = result.span();
            nn::transform(s.begin(), s.end(),
                           o.begin(), r.begin(),
                           std::forward<F>(func));
            return result;
        }

        // ── 逐元素二元变换（就地修改） ──────────────────────────────────
        template <Precision P, typename F>
        void binary_apply_inplace(MatrixT<P>& a, const MatrixT<P>& b, F&& func)
        {
            NN_ASSERT(a.rows() == b.rows() && a.cols() == b.cols(),
                       "binary_apply_inplace dimension mismatch");
            auto s = a.span();
            auto o = b.span();
            nn::transform(s.begin(), s.end(),
                           o.begin(), s.begin(),
                           std::forward<F>(func));
        }
    } // namespace detail
} // namespace nn

// ── GpuTensor 方法实现（需要 Matrix 和 GpuBackend 的完整定义）──────────
#ifdef NN_HAS_VULKAN
#include "backend/compute_gpu_tensor_impl.hpp"
#endif

