#ifndef NN_COMPUTE_LAYER_CONV_HPP
#define NN_COMPUTE_LAYER_CONV_HPP

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "compute_engine.hpp"
#include "compute_tensor.hpp"
#include "model_spec.hpp"
#include "expr_dsl.hpp"

namespace nn
{

class Conv2D final : public Layer
{
private:
    std::size_t in_channels_, out_channels_;
    std::size_t kernel_, stride_, padding_;
    std::size_t in_h_, in_w_;   // 输入空间尺寸（方形外也可用，用于 im2col）
    std::size_t out_h_, out_w_; // 输出空间尺寸（构造时计算）

    Tensor w_;        // (C_out, C_in*k*k)
    Tensor b_;        // (C_out, 1)
    Tensor grad_w_;
    Tensor grad_b_;
    Matrix col_cache_;           // im2col 输出 (C_in*k*k, batch*OH*OW)，供 backward

    inline static thread_local std::mt19937_64 rng_{std::random_device{}()};

    // ── im2col：input (C_in*H*W, batch) → col (C_in*k*k, batch*OH*OW) ──
    static Matrix im2col_(const Matrix& input,
                          std::size_t C_in, std::size_t H_in, std::size_t W_in,
                          std::size_t batch,
                          std::size_t k, std::size_t stride, std::size_t pad,
                          std::size_t H_out, std::size_t W_out)
    {
        Matrix col(C_in * k * k, batch * H_out * W_out);
        for (std::size_t b = 0; b < batch; ++b)
        {
            for (std::size_t oh = 0; oh < H_out; ++oh)
            {
                for (std::size_t ow = 0; ow < W_out; ++ow)
                {
                    for (std::size_t ci = 0; ci < C_in; ++ci)
                    {
                        for (std::size_t kh = 0; kh < k; ++kh)
                        {
                            for (std::size_t kw = 0; kw < k; ++kw)
                            {
                                const long ih = static_cast<long>(oh * stride + kh) - static_cast<long>(pad);
                                const long iw = static_cast<long>(ow * stride + kw) - static_cast<long>(pad);
                                Scalar v = Scalar{0};
                                if (ih >= 0 && iw >= 0 &&
                                    ih < static_cast<long>(H_in) && iw < static_cast<long>(W_in))
                                {
                                    v = input.at_unchecked(ci * H_in * W_in + ih * W_in + iw, b);
                                }
                                const std::size_t r = ci * k * k + kh * k + kw;
                                const std::size_t c = b * H_out * W_out + oh * W_out + ow;
                                col.set_value_unchecked(r, c, v);
                            }
                        }
                    }
                }
            }
        }
        return col;
    }

    // ── col2im：col (C_in*k*k, batch*OH*OW) → (C_in*H*W, batch) ──
    static Matrix col2im_(const Matrix& col,
                          std::size_t C_in, std::size_t H_in, std::size_t W_in,
                          std::size_t batch,
                          std::size_t k, std::size_t stride, std::size_t pad,
                          std::size_t H_out, std::size_t W_out)
    {
        Matrix out(C_in * H_in * W_in, batch, Scalar{0});
        for (std::size_t b = 0; b < batch; ++b)
        {
            for (std::size_t oh = 0; oh < H_out; ++oh)
            {
                for (std::size_t ow = 0; ow < W_out; ++ow)
                {
                    for (std::size_t ci = 0; ci < C_in; ++ci)
                    {
                        for (std::size_t kh = 0; kh < k; ++kh)
                        {
                            for (std::size_t kw = 0; kw < k; ++kw)
                            {
                                const long ih = static_cast<long>(oh * stride + kh) - static_cast<long>(pad);
                                const long iw = static_cast<long>(ow * stride + kw) - static_cast<long>(pad);
                                if (ih < 0 || iw < 0 ||
                                    ih >= static_cast<long>(H_in) || iw >= static_cast<long>(W_in))
                                    continue;
                                const std::size_t r = ci * k * k + kh * k + kw;
                                const std::size_t c = b * H_out * W_out + oh * W_out + ow;
                                const std::size_t orow = ci * H_in * W_in + ih * W_in + iw;
                                out.set_value_unchecked(orow, b,
                                    out.at_unchecked(orow, b) + col.at_unchecked(r, c));
                            }
                        }
                    }
                }
            }
        }
        return out;
    }

    // ── 布局重排：Z (C_out, batch*OH*OW) → out (C_out*OH*OW, batch) ──
    static Matrix cols_to_samples_(const Matrix& Z, std::size_t C_out,
                                   std::size_t OH, std::size_t OW, std::size_t batch)
    {
        Matrix out(C_out * OH * OW, batch);
        for (std::size_t co = 0; co < C_out; ++co)
            for (std::size_t b = 0; b < batch; ++b)
                for (std::size_t p = 0; p < OH * OW; ++p)
                    out.set_value_unchecked(co * OH * OW + p, b,
                        Z.at_unchecked(co, b * OH * OW + p));
        return out;
    }

    // ── 布局重排：out (C_out*OH*OW, batch) → Z (C_out, batch*OH*OW) ──
    static Matrix samples_to_cols_(const Matrix& out, std::size_t C_out,
                                   std::size_t OH, std::size_t OW, std::size_t batch)
    {
        Matrix Z(C_out, batch * OH * OW);
        for (std::size_t co = 0; co < C_out; ++co)
            for (std::size_t b = 0; b < batch; ++b)
                for (std::size_t p = 0; p < OH * OW; ++p)
                    Z.set_value_unchecked(co, b * OH * OW + p,
                        out.at_unchecked(co * OH * OW + p, b));
        return Z;
    }

public:
    Conv2D(std::size_t in_channels, std::size_t out_channels,
           std::size_t kernel, std::size_t stride = 1, std::size_t padding = 0,
           std::size_t in_h = 0, std::size_t in_w = 0)
        : in_channels_(in_channels), out_channels_(out_channels),
          kernel_(kernel), stride_(stride), padding_(padding),
          in_h_(in_h), in_w_(in_w)
    {
        out_h_ = (in_h_ + 2 * padding_ - kernel_) / stride_ + 1;
        out_w_ = (in_w_ + 2 * padding_ - kernel_) / stride_ + 1;
    }

    [[nodiscard]] Result<void> init(ComputeEngine& engine) override
    {
        const std::size_t fan_in = in_channels_ * kernel_ * kernel_;

        // 权重 (C_out, C_in*k*k) — He 风格均匀初始化
        Matrix w_cpu(out_channels_, fan_in);
        const Scalar limit = std::sqrt(6.0 / static_cast<Scalar>(fan_in + out_channels_));
        std::uniform_real_distribution<Scalar> dist(-limit, limit);
        auto w_span = w_cpu.span();
        for (std::size_t i = 0; i < w_cpu.size(); ++i)
            w_span[i] = dist(rng_);

        Matrix b_cpu(out_channels_, 1);  // 零初始化

        auto w_res = engine.from_matrix(w_cpu);
        if (!w_res) return std::unexpected(w_res.error());
        w_ = std::move(*w_res);
        auto b_res = engine.from_matrix(b_cpu);
        if (!b_res) return std::unexpected(b_res.error());
        b_ = std::move(*b_res);

        grad_w_ = engine.create_tensor(out_channels_, fan_in);
        grad_b_ = engine.create_tensor(out_channels_, 1);
        { auto r1 = engine.zero(grad_w_); if (!r1) return std::unexpected(r1.error()); }
        { auto r2 = engine.zero(grad_b_); if (!r2) return std::unexpected(r2.error()); }
        return {};
    }

    std::vector<TensorRef> parameters() override { return {w_, b_}; }
    std::vector<TensorRef> param_gradients() override { return {grad_w_, grad_b_}; }

    void clear_cache() override { col_cache_ = Matrix{}; }

    // ── forward: Z = W × im2col(x) + b，重排回 batch-major ──────────────
    [[nodiscard]] Result<Tensor> forward(
        ComputeEngine& engine, const Tensor& input) override
    {
        if (input.rows() != in_channels_ * in_h_ * in_w_)
            return std::unexpected(Error{"conv forward: input shape mismatch"});
        const std::size_t batch = input.cols();

        // 下载输入 → CPU im2col
        auto in_r = engine.to_matrix(input);
        if (!in_r) return std::unexpected(in_r.error());
        Matrix col = im2col_(*in_r, in_channels_, in_h_, in_w_, batch,
                             kernel_, stride_, padding_, out_h_, out_w_);
        if (!checkpoint_mode_)
            col_cache_ = col;

        auto col_t = engine.from_matrix(col);
        if (!col_t) return std::unexpected(col_t.error());

        // Z = W × col → (C_out, batch*OH*OW)
        auto Z = engine.matmul(w_, *col_t, false, false);
        if (!Z) return std::unexpected(Z.error());
        auto rb = engine.broadcast_row_inplace(*Z, b_, BinaryOp::Add);
        if (!rb) return std::unexpected(rb.error());

        // 重排 → (C_out*OH*OW, batch)
        auto Z_cpu = engine.to_matrix(*Z);
        if (!Z_cpu) return std::unexpected(Z_cpu.error());
        Matrix out_cpu = cols_to_samples_(*Z_cpu, out_channels_, out_h_, out_w_, batch);
        return engine.from_matrix(out_cpu);
    }

    // ── backward ───────────────────────────────────────────────────────
    [[nodiscard]] Result<Tensor> backward(
        ComputeEngine& engine, const Tensor& grad_output) override
    {
        const std::size_t batch = grad_output.cols();

        // grad_output (C_out*OH*OW, batch) → gZ (C_out, batch*OH*OW)
        auto g_cpu = engine.to_matrix(grad_output);
        if (!g_cpu) return std::unexpected(g_cpu.error());
        Matrix gZ_cpu = samples_to_cols_(*g_cpu, out_channels_, out_h_, out_w_, batch);
        auto gZ = engine.from_matrix(gZ_cpu);
        if (!gZ) return std::unexpected(gZ.error());

        // grad_W += gZ × col^T → (C_out, C_in*k*k)
        auto col_t = engine.from_matrix(col_cache_);
        if (!col_t) return std::unexpected(col_t.error());
        auto gw = engine.matmul(*gZ, *col_t, false, true);
        if (!gw) return std::unexpected(gw.error());
        auto r1 = engine.add_inplace(grad_w_, *gw);
        if (!r1) return std::unexpected(r1.error());

        // grad_b += row_reduce_sum(gZ) → (C_out, 1)
        auto gb = engine.row_reduce_sum(*gZ);
        if (!gb) return std::unexpected(gb.error());
        auto r2 = engine.add_inplace(grad_b_, *gb);
        if (!r2) return std::unexpected(r2.error());

        // grad_col = W^T × gZ → (C_in*k*k, batch*OH*OW)
        auto gcol = engine.matmul(w_, *gZ, true, false);
        if (!gcol) return std::unexpected(gcol.error());
        auto gcol_cpu = engine.to_matrix(*gcol);
        if (!gcol_cpu) return std::unexpected(gcol_cpu.error());
        Matrix gin_cpu = col2im_(*gcol_cpu, in_channels_, in_h_, in_w_, batch,
                                 kernel_, stride_, padding_, out_h_, out_w_);
        return engine.from_matrix(gin_cpu);
    }
};

// ══════════════════════════════════════════════════════════════════════════
// MaxPool2D — 二维最大池化（记录 argmax 供 backward）
//
// 布局约定（batch-major 列布局）：
//   输入  Tensor: (C * H * W, batch)
//   输出  Tensor: (C * Hp * Wp, batch)，Hp=(H-pool)/stride+1，Wp 同理
//
// 算法（只在此处，不在 Engine/Shader）：
//   forward:  每个 (pool×pool) 窗口取最大值，记录 argmax 位置
//   backward: 把梯度散射回 argmax 位置（其余位置为 0）
//
// 说明：池化涉及窗口 max + argmax 追踪，在 CPU 端完成（to_matrix/from_matrix），
//       与 Conv2D 的 im2col 策略一致；GPU 融合池化内核留作后续优化。
// ══════════════════════════════════════════════════════════════════════════
class MaxPool2D final : public Layer
{
private:
    std::size_t channels_, in_h_, in_w_;
    std::size_t pool_, stride_;
    std::size_t out_h_, out_w_;
    std::vector<std::size_t> max_indices_;  // (channels*out_h*out_w, batch) 扁平 argmax 行索引

public:
    MaxPool2D(std::size_t channels, std::size_t in_h, std::size_t in_w,
              std::size_t pool = 2, std::size_t stride = 0)
        : channels_(channels), in_h_(in_h), in_w_(in_w),
          pool_(pool), stride_(stride != 0 ? stride : pool)
    {
        out_h_ = (in_h_ - pool_) / stride_ + 1;
        out_w_ = (in_w_ - pool_) / stride_ + 1;
    }

    void clear_cache() override { max_indices_.clear(); }

    // ── forward ─────────────────────────────────────────────────────────
    [[nodiscard]] Result<Tensor> forward(
        ComputeEngine& engine, const Tensor& input) override
    {
        if (input.rows() != channels_ * in_h_ * in_w_)
            return std::unexpected(Error{"maxpool forward: input shape mismatch"});
        const std::size_t batch = input.cols();

        auto in_r = engine.to_matrix(input);
        if (!in_r) return std::unexpected(in_r.error());
        const Matrix& in = *in_r;

        Matrix out(channels_ * out_h_ * out_w_, batch);
        if (!checkpoint_mode_)
            max_indices_.assign(channels_ * out_h_ * out_w_ * batch, 0);

        const std::size_t out_area = out_h_ * out_w_;
        for (std::size_t b = 0; b < batch; ++b)
        {
            for (std::size_t c = 0; c < channels_; ++c)
            {
                for (std::size_t oh = 0; oh < out_h_; ++oh)
                {
                    for (std::size_t ow = 0; ow < out_w_; ++ow)
                    {
                        Scalar best = -std::numeric_limits<Scalar>::infinity();
                        std::size_t best_idx = 0;
                        for (std::size_t dh = 0; dh < pool_; ++dh)
                        {
                            for (std::size_t dw = 0; dw < pool_; ++dw)
                            {
                                const std::size_t ih = oh * stride_ + dh;
                                const std::size_t iw = ow * stride_ + dw;
                                const std::size_t r = c * in_h_ * in_w_ + ih * in_w_ + iw;
                                const Scalar v = in.at_unchecked(r, b);
                                if (v > best) { best = v; best_idx = r; }
                            }
                        }
                        const std::size_t orr = c * out_area + oh * out_w_ + ow;
                        out.set_value_unchecked(orr, b, best);
                        if (!checkpoint_mode_)
                            max_indices_[b * channels_ * out_area + orr] = best_idx;
                    }
                }
            }
        }
        return engine.from_matrix(out);
    }

    // ── backward: 散射梯度到 argmax 位置 ────────────────────────────────
    [[nodiscard]] Result<Tensor> backward(
        ComputeEngine& engine, const Tensor& grad_output) override
    {
        const std::size_t batch = grad_output.cols();
        const std::size_t out_area = out_h_ * out_w_;

        auto g_r = engine.to_matrix(grad_output);
        if (!g_r) return std::unexpected(g_r.error());
        const Matrix& g = *g_r;

        Matrix gin(channels_ * in_h_ * in_w_, batch, Scalar{0});
        for (std::size_t b = 0; b < batch; ++b)
        {
            for (std::size_t c = 0; c < channels_; ++c)
            {
                for (std::size_t oh = 0; oh < out_h_; ++oh)
                {
                    for (std::size_t ow = 0; ow < out_w_; ++ow)
                    {
                        const std::size_t orr = c * out_area + oh * out_w_ + ow;
                        const std::size_t idx = max_indices_[b * channels_ * out_area + orr];
                        gin.set_value_unchecked(idx, b,
                            gin.at_unchecked(idx, b) + g.at_unchecked(orr, b));
                    }
                }
            }
        }
        return engine.from_matrix(gin);
    }
};

// ══════════════════════════════════════════════════════════════════════════
// Softmax — 按行 softmax（用于注意力权重）
//
// 算法（只在此处，不在 Engine/Shader）：
//   forward:  row_max[r] = max_c x[r][c]               (row_reduce_max)
//             shifted[r][c] = x[r][c] - row_max[r]    (broadcast_row Sub)
//             exp_shift[r][c] = exp(shifted[r][c])     (unary Exp)
//             row_sum[r] = Σ_c exp_shift[r][c]         (row_reduce_sum)
//             out[r][c] = exp_shift[r][c] / row_sum[r] (broadcast_row Div)
//   backward: grad_x = out ⊙ (grad_out - row_dot(out ⊙ grad_out))
//             row_dot[r] = Σ_c out[r][c] * grad_out[r][c]
// ══════════════════════════════════════════════════════════════════════════
} // namespace nn

#endif // NN_COMPUTE_LAYER_CONV_HPP
