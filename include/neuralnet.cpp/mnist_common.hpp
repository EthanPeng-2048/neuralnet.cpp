#ifndef MNIST_COMMON_HPP
#define MNIST_COMMON_HPP

#include <cstddef>
#include <vector>

#include <neuralnet.cpp/nn.hpp>

namespace nn {

// ── MNIST 常量 ──────────────────────────────────────────────────────────────
inline constexpr std::size_t MNIST_INPUT_DIM = 784;
inline constexpr std::size_t MNIST_NUM_CLASSES = 10;

// 默认网络架构：输入层 -> 隐藏层1 -> 隐藏层2 -> 隐藏层3 -> 输出层
inline const std::vector<std::size_t> MNIST_LAYER_DIMS = {
    MNIST_INPUT_DIM, 256, 128, 64, MNIST_NUM_CLASSES
};

// ── 构建 MNIST 模型 ────────────────────────────────────────────────────────
// 根据 MNIST_LAYER_DIMS 自动构建网络
[[nodiscard]] inline Model build_mnist_model()
{
    Model model;
    for (std::size_t i = 0; i < MNIST_LAYER_DIMS.size() - 1; ++i)
    {
        std::size_t in_dim = MNIST_LAYER_DIMS[i];
        std::size_t out_dim = MNIST_LAYER_DIMS[i + 1];
        
        // 添加线性层
        model.add<Linear>(in_dim, out_dim);
        
        // 如果不是最后一层（输出层），则添加 LayerNorm 和 GeLU 激活函数
        if (i < MNIST_LAYER_DIMS.size() - 2)
        {
            model.add<LayerNorm>(out_dim)
                 .add<GeLU>();
        }
    }
    return model;
}

} // namespace nn

#endif // MNIST_COMMON_HPP