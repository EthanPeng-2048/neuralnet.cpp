# w2-domain

## evaluate_mnist 原始签名否决（cli/cli_mnist_io.hpp:16-18 头注释与 evaluate_mnist 函数注释，整改前行号）
- 类型：否决方案
- 内容：曾按「用户原始签名 evaluate_mnist(nn::Layer&, ...)」设计，实际不可行——nn::Model 不是 nn::Layer 的派生类，且 Model::forward 自带 engine 绑定（签名不同于 Layer::forward(engine, ...)）；最终改用 nn::Model&，原注释还附带「与原签名行为完全等价」的等价性论证。技术理由已改写为当前注释（签名为何取 nn::Model&），原始签名叙事与等价性论证移至本条。
