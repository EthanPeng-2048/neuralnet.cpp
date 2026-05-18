#include "neuralnet.cpp/nn.hpp"
#include "neuralnet.cpp/model_io.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>

// -------------------- 数据加载 --------------------
std::pair<nn::Matrix, nn::Matrix> load_csv(const std::string& filename, int max_samples = -1) {
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Cannot open file: " + filename);

    std::vector<double> features;
    std::vector<int> labels;
    std::string line;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        int label;
        std::getline(ss, token, ',');
        label = std::stoi(token);
        labels.push_back(label);

        while (std::getline(ss, token, ',')) {
            features.push_back(std::stod(token));  // 数据已由 ToTensor() 归一化到 [0,1]
        }
    }

    std::size_t N = labels.size();
    std::size_t feat_dim = features.size() / N;  // 784

    if (max_samples > 0 && static_cast<std::size_t>(max_samples) < N) {
        N = max_samples;
        features.resize(N * feat_dim);
        labels.resize(N);
    }

    // 特征矩阵：形状 (feat_dim, N)
    nn::Matrix feat_mat(feat_dim, N);
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < feat_dim; ++j) {
            feat_mat.set_value_unchecked(j, i, features[i * feat_dim + j]);
        }
    }

    // One‑hot 标签矩阵：形状 (10, N)
    nn::Matrix label_mat(10, N);
    for (std::size_t i = 0; i < N; ++i) {
        int lbl = labels[i];
        if (lbl < 0 || lbl >= 10) throw std::out_of_range("Label out of range");
        label_mat.set_value_unchecked(lbl, i, 1.0);
    }

    return {feat_mat, label_mat};
}

// -------------------- 评估函数 --------------------
double evaluate(nn::Linear& l1, nn::ReLU& r1, nn::Linear& l2, nn::ReLU& r2,
                nn::Linear& l3, nn::ReLU& r3, nn::Linear& l4,
                const nn::Matrix& x, const nn::Matrix& y_onehot) {
    std::size_t N = x.cols();
    auto out = l4.forward(r3.forward(l3.forward(r2.forward(l2.forward(r1.forward(l1.forward(x)))))));
    int correct = 0;
    for (std::size_t i = 0; i < N; ++i) {
        // 预测类别：logits 最大值索引
        double max_val = out.at_unchecked(0, i);
        int pred = 0;
        for (int j = 1; j < 10; ++j) {
            double val = out.at_unchecked(j, i);
            if (val > max_val) {
                max_val = val;
                pred = j;
            }
        }
        // 真实类别
        int true_label = -1;
        for (int j = 0; j < 10; ++j) {
            if (y_onehot.at_unchecked(j, i) == 1.0) {
                true_label = j;
                break;
            }
        }
        if (pred == true_label) ++correct;
    }
    return static_cast<double>(correct) / N;
}

// -------------------- 主函数 --------------------
int main(int argc, char* argv[]) {
    // 可选命令行参数：--load model.bin 或 --save model.bin
    std::string model_path = "mnist_model.bin";
    bool load_existing = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--load" && i+1 < argc) {
            model_path = argv[++i];
            load_existing = true;
        } else if (arg == "--save" && i+1 < argc) {
            model_path = argv[++i];
            // 仅指定保存路径，不加载
        }
    }

    // 加载数据
    std::cout << "Loading MNIST data..." << std::endl;
    auto [train_x, train_y] = load_csv("./mnist_data/train.csv");
    auto [test_x, test_y]   = load_csv("./mnist_data/test.csv");
    std::cout << "Train: " << train_x.cols() << " samples, features " << train_x.rows()
              << " x " << train_x.cols() << std::endl;

    // 构建网络
    nn::Linear l1(784, 64);
    nn::ReLU   r1;
    nn::Linear l2(64, 64);
    nn::ReLU   r2;
    nn::Linear l3(64, 64);
    nn::ReLU   r3;
    nn::Linear l4(64, 10);

    // 如果指定加载，则载入已有参数
    if (load_existing) {
        try {
            nn::load_model(model_path, l1, l2, l3, l4);
        } catch (const std::exception& e) {
            std::cerr << "Failed to load model: " << e.what() << ", starting from scratch.\n";
        }
    }

    // 收集参数和梯度（同前）
    std::vector<std::reference_wrapper<nn::Matrix>> params, grads;
    for (auto& p : l1.parameters()) params.push_back(p);
    for (auto& p : l2.parameters()) params.push_back(p);
    for (auto& p : l3.parameters()) params.push_back(p);
    for (auto& p : l4.parameters()) params.push_back(p);
    for (auto& g : l1.param_gradients()) grads.push_back(g);
    for (auto& g : l2.param_gradients()) grads.push_back(g);
    for (auto& g : l3.param_gradients()) grads.push_back(g);
    for (auto& g : l4.param_gradients()) grads.push_back(g);

    nn::SGD optimizer(params, grads, 0.01);
    nn::CrossEntropyLoss ce_loss;

    const std::size_t batch_size = 64;
    const std::size_t num_batches = train_x.cols() / batch_size;

    // 训练
    const int epochs = 5;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0;

        for (std::size_t batch = 0; batch < num_batches; ++batch) {
            // 提取当前 batch
            std::size_t start = batch * batch_size;
            nn::Matrix x_batch(train_x.rows(), batch_size);
            nn::Matrix y_batch(train_y.rows(), batch_size);
            for (std::size_t i = 0; i < batch_size; ++i) {
                for (std::size_t r = 0; r < train_x.rows(); ++r) {
                    x_batch.set_value_unchecked(r, i, train_x.at_unchecked(r, start + i));
                }
                for (std::size_t r = 0; r < train_y.rows(); ++r) {
                    y_batch.set_value_unchecked(r, i, train_y.at_unchecked(r, start + i));
                }
            }

            // 前向
            auto out = l4.forward(r3.forward(l3.forward(r2.forward(l2.forward(r1.forward(l1.forward(x_batch)))))));
            double loss = ce_loss.forward(out, y_batch);
            total_loss += loss;

            // 反向
            auto grad = ce_loss.backward();
            grad = l4.backward(grad);
            grad = r3.backward(grad);
            grad = l3.backward(grad);
            grad = r2.backward(grad);
            grad = l2.backward(grad);
            grad = r1.backward(grad);
            grad = l1.backward(grad);

            // 更新参数
            optimizer.step();
            optimizer.zero_grad();
        }

        double avg_loss = total_loss / num_batches;
        double train_acc = evaluate(l1, r1, l2, r2, l3, r3, l4, train_x, train_y);
        double test_acc  = evaluate(l1, r1, l2, r2, l3, r3, l4, test_x, test_y);
        std::cout << "Epoch " << epoch+1 << "/" << epochs
                  << "  loss = " << avg_loss
                  << "  train acc = " << train_acc
                  << "  test acc = " << test_acc << std::endl;
    }

    // 训练结束后保存模型（也可每个 epoch 保存一次）
    nn::save_model(model_path, l1, l2, l3, l4);
    return 0;
}
