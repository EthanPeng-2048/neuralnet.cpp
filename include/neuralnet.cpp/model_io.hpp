#ifndef MODEL_IO_HPP
#define MODEL_IO_HPP

#include "layer.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace nn {

// 保存模型参数到二进制文件
// 注意：网络结构（各层维度）必须与加载时一致
inline void save_model(const std::string &filename,
                       Linear &l1, Linear &l2, Linear &l3, Linear &l4) {
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) {
        throw std::runtime_error("Cannot write model file: " + filename);
    }

    const uint32_t magic = 0x4E4E4E4E;
    const uint32_t version = 1;
    ofs.write(reinterpret_cast<const char *>(&magic), sizeof(magic));
    ofs.write(reinterpret_cast<const char *>(&version), sizeof(version));

    auto write_matrix = [&](const Matrix &m) {
        std::size_t rows = m.rows(), cols = m.cols();
        ofs.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
        ofs.write(reinterpret_cast<const char *>(&cols), sizeof(cols));
        const auto &data = m.data();
        ofs.write(reinterpret_cast<const char *>(data.data()),
                  data.size() * sizeof(double));
    };

    // parameters() 返回顺序为 [W, b]
    auto params1 = l1.parameters();
    auto params2 = l2.parameters();
    auto params3 = l3.parameters();
    auto params4 = l4.parameters();
    write_matrix(params1[0].get()); // W1
    write_matrix(params1[1].get()); // b1
    write_matrix(params2[0].get()); // W2
    write_matrix(params2[1].get()); // b2
    write_matrix(params3[0].get()); // W3
    write_matrix(params3[1].get()); // b3
    write_matrix(params4[0].get()); // W4
    write_matrix(params4[1].get()); // b4

    std::cout << "Model saved to " << filename << std::endl;
}

// 从二进制文件加载模型参数
inline void load_model(const std::string &filename,
                       Linear &l1, Linear &l2, Linear &l3, Linear &l4) {
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("Cannot read model file: " + filename);
    }

    uint32_t magic, version;
    ifs.read(reinterpret_cast<char *>(&magic), sizeof(magic));
    ifs.read(reinterpret_cast<char *>(&version), sizeof(version));
    if (magic != 0x4E4E4E4E || version != 1) {
        throw std::runtime_error("Invalid model file format");
    }

    auto read_matrix = [&](Matrix &m) {
        std::size_t rows, cols;
        ifs.read(reinterpret_cast<char *>(&rows), sizeof(rows));
        ifs.read(reinterpret_cast<char *>(&cols), sizeof(cols));
        if (rows != m.rows() || cols != m.cols()) {
            throw std::runtime_error("Model shape mismatch");
        }
        auto &data = m.data();
        ifs.read(reinterpret_cast<char *>(data.data()),
                 data.size() * sizeof(double));
    };

    auto params1 = l1.parameters();
    auto params2 = l2.parameters();
    auto params3 = l3.parameters();
    auto params4 = l4.parameters();
    read_matrix(params1[0].get()); // W1
    read_matrix(params1[1].get()); // b1
    read_matrix(params2[0].get()); // W2
    read_matrix(params2[1].get()); // b2
    read_matrix(params3[0].get()); // W3
    read_matrix(params3[1].get()); // b3
    read_matrix(params4[0].get()); // W4
    read_matrix(params4[1].get()); // b4

    std::cout << "Model loaded from " << filename << std::endl;
}

} // namespace nn

#endif // MODEL_IO_HPP
