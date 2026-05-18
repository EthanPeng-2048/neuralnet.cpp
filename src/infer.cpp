#include "neuralnet.cpp/nn.hpp"
#include "neuralnet.cpp/model_io.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdint>

// -------------------- 从 CSV 字符串读取单样本 --------------------
nn::Matrix load_image_from_csv(const std::string& csv_line) {
    std::vector<double> pixels;
    std::stringstream ss(csv_line);
    std::string token;
    while (std::getline(ss, token, ',')) {
        pixels.push_back(std::stod(token));
    }
    if (pixels.size() != 784)
        throw std::runtime_error("CSV must contain exactly 784 values");

    // 构造形状为 (784, 1) 的矩阵
    nn::Matrix img(784, 1);
    for (std::size_t i = 0; i < 784; ++i)
        img.set_value_unchecked(i, 0, pixels[i]);
    return img;
}

// -------------------- 预测函数 --------------------
int predict(nn::Linear& l1, nn::ReLU& r1, nn::Linear& l2, nn::ReLU& r2,
            nn::Linear& l3, nn::ReLU& r3, nn::Linear& l4,
            const nn::Matrix& img) {
    auto out = l4.forward(r3.forward(l3.forward(r2.forward(l2.forward(r1.forward(l1.forward(img)))))));
    // 找到最大值的索引
    double max_val = out.at_unchecked(0, 0);
    int pred = 0;
    for (int i = 1; i < 10; ++i) {
        double val = out.at_unchecked(i, 0);
        if (val > max_val) {
            max_val = val;
            pred = i;
        }
    }
    return pred;
}

// -------------------- 主函数 --------------------
int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model.bin> <image.csv>" << std::endl;
        std::cerr << "   or: " << argv[0] << " <model.bin> <image.png>  (if compiled with stb_image)" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string img_path   = argv[2];

    // 构建网络（与训练时结构一致）
    nn::Linear l1(784, 64);
    nn::ReLU   r1;
    nn::Linear l2(64, 64);
    nn::ReLU   r2;
    nn::Linear l3(64, 64);
    nn::ReLU   r3;
    nn::Linear l4(64, 10);

    // 加载模型参数
    try {
        nn::load_model(model_path, l1, l2, l3, l4);
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return 1;
    }

    // 读取图片
    nn::Matrix img;
    try {
        // 判断文件扩展名（简单处理）
        if (img_path.find(".csv") != std::string::npos || img_path.find(".txt") != std::string::npos) {
            std::ifstream file(img_path);
            if (!file) throw std::runtime_error("Cannot open image file");
            std::string line;
            std::getline(file, line);
            img = load_image_from_csv(line);
        } else {
            // 图像文件需要额外依赖，此处仅作示例框架
            std::cerr << "Image file loading not implemented in this example.\n";
            std::cerr << "Please use a CSV file with 784 pixel values.\n";
            return 1;
            /* 若使用 stb_image，可添加以下代码：
            #define STB_IMAGE_IMPLEMENTATION
            #include "stb_image.h"
            int w, h, channels;
            unsigned char* data = stbi_load(img_path.c_str(), &w, &h, &channels, 1); // 强制灰度
            if (!data) throw std::runtime_error("Failed to load image");
            if (w != 28 || h != 28) {
                // 需要缩放到 28x28，此处略
            }
            img = nn::Matrix(784, 1);
            for (int i = 0; i < 784; ++i)
                img.set_value_unchecked(i, 0, data[i] / 255.0);
            stbi_image_free(data);
            */
        }
    } catch (const std::exception& e) {
        std::cerr << "Error loading image: " << e.what() << std::endl;
        return 1;
    }

    int result = predict(l1, r1, l2, r2, l3, r3, l4, img);
    std::cout << "Predicted digit: " << result << std::endl;
    return 0;
}