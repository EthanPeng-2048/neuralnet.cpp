// ── mnist_io.hpp — MNIST 数据加载与评估工具 ─────────────────────────────────
//
// 抽取自 mnist_train/mnist_bench 中重复的 load_csv 与 evaluate 函数。
//
// load_mnist_csv:
//   - 一次性读取整个 CSV 文件到内存，再以 from_chars 快速解析
//   - 文件格式：每行 "label,f1,f2,...,f784"（首列为 0-9 标签，后接 784 个像素值）
//   - 返回 (feat_mat[784, N], label_mat[10, N])，label_mat 为 one-hot 编码
//   - max_samples = 0 表示加载全部样本
//
// evaluate_mnist:
//   - 全量前向后下载到 CPU 做 argmax，计算准确率
//   - GPU 模式下使用 begin_batch/end_batch 包裹 forward，消除 per-primitive 提交开销
//   - eval_samples > 0 时只评估前 N 个样本（Transformer 评估成本较高）
//
// 注：用户原始签名 evaluate_mnist(nn::Layer&, ...) 实际不可行，因为 nn::Model
// 不是 nn::Layer 的派生类，且 Model::forward 自带 engine 绑定（签名不同）。
// 这里改为 nn::Model&，与现有调用点 model.forward(...) 一致，行为完全等价。
// ─────────────────────────────────────────────────────────────────────────

#pragma once

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "neuralnet.cpp/algebra_matrix.hpp"  // nn::Matrix
#include "neuralnet.cpp/compute_engine.hpp"   // nn::ComputeEngine
#include "neuralnet.cpp/core_errors.hpp"      // nn::Result / nn::Error
#include "neuralnet.cpp/domain_mnist.hpp"     // nn::MNIST_NUM_CLASSES
#include "neuralnet.cpp/model_container.hpp"  // nn::Model

namespace nn::cli
{
    // ── 从 CSV 加载 MNIST 数据 ─────────────────────────────────────────────
    // 返回 (feat_mat[feat_dim, N], label_mat[10, N])，label_mat 为 one-hot。
    // max_samples = 0 表示加载全部样本。
    [[nodiscard]] inline nn::Result<std::pair<nn::Matrix, nn::Matrix>>
    load_mnist_csv(const std::string &path, std::size_t max_samples = 0)
    {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file.is_open())
            return std::unexpected(nn::Error{"Cannot open file: " + path});

        const auto file_size = file.tellg();
        file.seekg(0);

        std::string buffer(static_cast<std::size_t>(file_size), '\0');
        file.read(buffer.data(), file_size);
        file.close();

        std::size_t row_count = 0;
        for (char c : buffer)
            if (c == '\n') ++row_count;
        // 末行无换行符时也要计入（否则最后一行样本被静默丢弃）
        if (!buffer.empty() && buffer.back() != '\n')
            ++row_count;
        if (row_count == 0)
            return std::unexpected(nn::Error{"CSV file is empty or malformed: " + path});

        if (max_samples > 0 && max_samples < row_count)
            row_count = max_samples;

        const char *ptr = buffer.data();
        const char *end = buffer.data() + buffer.size();

        // 探测第一行以确定特征维度
        int first_label = 0;
        std::size_t feat_dim = 0;
        {
            const char *p = ptr;
            auto [p1, ec1] = std::from_chars(p, end, first_label);
            p = p1;
            std::size_t cnt = 0;
            while (p < end && *p != '\n' && *p != '\r')
            {
                if (*p == ',')
                {
                    ++cnt;
                    nn::Scalar tmp;
                    auto [p2, ec2] = std::from_chars(p + 1, end, tmp);
                    p = p2;
                }
                else
                    ++p;
            }
            feat_dim = cnt;
        }

        std::vector<nn::Scalar> features(row_count * feat_dim);
        std::vector<int> labels(row_count);

        std::size_t row = 0;
        ptr = buffer.data();

        while (ptr < end && row < row_count)
        {
            auto [p_label, ec_label] = std::from_chars(ptr, end, labels[row]);
            if (ec_label != std::errc{})
                return std::unexpected(
                    nn::Error{"Failed to parse label at row " + std::to_string(row)});
            ptr = p_label;

            for (std::size_t j = 0; j < feat_dim; ++j)
            {
                if (ptr < end && *ptr == ',') ++ptr;
                nn::Scalar val;
                auto [p_feat, ec_feat] = std::from_chars(ptr, end, val);
                if (ec_feat != std::errc{})
                    return std::unexpected(
                        nn::Error{"Failed to parse feature at row " + std::to_string(row)});
                features[row * feat_dim + j] = val;
                ptr = p_feat;
            }

            while (ptr < end && (*ptr == '\r' || *ptr == '\n')) ++ptr;
            ++row;
        }

        const std::size_t N = row;

        nn::Matrix feat_mat(feat_dim, N);
        for (std::size_t i = 0; i < N; ++i)
            for (std::size_t j = 0; j < feat_dim; ++j)
                feat_mat.set_value_unchecked(j, i, features[i * feat_dim + j]);

        nn::Matrix label_mat(10, N);
        for (std::size_t i = 0; i < N; ++i)
        {
            int lbl = labels[i];
            if (lbl < 0 || lbl >= 10)
                return std::unexpected(
                    nn::Error{"Label out of range: " + std::to_string(lbl)});
            label_mat.set_value_unchecked(lbl, i, 1.0);
        }

        return std::pair{std::move(feat_mat), std::move(label_mat)};
    }

    // ── 评估模型准确率 ─────────────────────────────────────────────────────
    // 全量前向后下载到 CPU 做 argmax，计算 top-1 准确率。
    // eval_samples > 0 时只评估前 N 个样本（用于 Transformer 等评估成本高的场景）。
    //
    // 注：原 evaluate 签名 (nn::Layer&, ...) 不可行 —— nn::Model 不派生自 nn::Layer，
    //     且 Model::forward 自带 engine 绑定，签名不同于 Layer::forward(engine, ...)。
    //     此处使用 nn::Model&，与现有调用点 model.forward(...) 一致。
    [[nodiscard]] inline nn::Result<nn::Scalar>
    evaluate_mnist(nn::Model &model, nn::ComputeEngine &engine,
                   const nn::Matrix &x, const nn::Matrix &y_onehot,
                   std::size_t eval_samples = 0)
    {
        const std::size_t N =
            (eval_samples > 0) ? std::min(x.cols(), eval_samples) : x.cols();
        if (N == 0)
            return std::unexpected(nn::Error{"evaluate_mnist: empty dataset"});

        // 若截取子集，则拷贝前 N 列；全量评估直接引用原矩阵（避免无谓拷贝）
        nn::Matrix x_sub, y_sub;
        const nn::Matrix *xp = &x;
        const nn::Matrix *yp = &y_onehot;
        if (N < x.cols())
        {
            x_sub = nn::Matrix(x.rows(), N);
            y_sub = nn::Matrix(y_onehot.rows(), N);
            for (std::size_t i = 0; i < N; ++i)
            {
                for (std::size_t r = 0; r < x.rows(); ++r)
                    x_sub.set_value_unchecked(r, i, x.at_unchecked(r, i));
                for (std::size_t r = 0; r < y_onehot.rows(); ++r)
                    y_sub.set_value_unchecked(r, i, y_onehot.at_unchecked(r, i));
            }
            xp = &x_sub;
            yp = &y_sub;
        }

        auto x_tensor_r = engine.from_matrix(*xp);
        if (!x_tensor_r) return std::unexpected(std::move(x_tensor_r).error());

        // 评估一律用推理模式：BatchNorm 使用 running 统计量而非 batch 统计量。
        // 评估结束后恢复训练模式（调用方默认为训练场景）。
        model.set_training(false);

        // batch 模式加速 forward（GPU 下消除 per-primitive 提交开销）
        auto bb = engine.begin_batch();
        if (!bb) return std::unexpected(bb.error());

        auto out_tensor_r = model.forward(*x_tensor_r);
        if (!out_tensor_r) return std::unexpected(std::move(out_tensor_r).error());

        auto eb = engine.end_batch();
        if (!eb) return std::unexpected(eb.error());

        model.set_training(true);

        auto out_r = engine.to_matrix(*out_tensor_r);
        if (!out_r) return std::unexpected(std::move(out_r).error());
        const auto &out = *out_r;

        int correct = 0;
        for (std::size_t i = 0; i < N; ++i)
        {
            nn::Scalar max_val = out.at_unchecked(0, i);
            int pred = 0;
            for (int j = 1; j < static_cast<int>(nn::MNIST_NUM_CLASSES); ++j)
            {
                nn::Scalar val = out.at_unchecked(j, i);
                if (val > max_val)
                {
                    max_val = val;
                    pred = j;
                }
            }
            int true_label = -1;
            for (int j = 0; j < static_cast<int>(nn::MNIST_NUM_CLASSES); ++j)
            {
                if (yp->at_unchecked(j, i) == 1.0)
                {
                    true_label = j;
                    break;
                }
            }
            if (pred == true_label)
                ++correct;
        }
        return static_cast<nn::Scalar>(correct) / static_cast<nn::Scalar>(N);
    }
} // namespace nn::cli
