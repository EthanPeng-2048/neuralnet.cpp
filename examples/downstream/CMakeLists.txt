# ══════════════════════════════════════════════════════════════════════════
#  examples/downstream — 最小下游消费工程（P0-4 验收样例）
#
#  模拟"第三方把 neuralnet.cpp 丢进 3rd_party/ 后只写两行 CMake"的形态：
#
#      add_subdirectory(3rd_party/neuralnet.cpp)
#      target_link_libraries(app PRIVATE neuralnet::nn)
#
#  外加源码侧一行：#include <neuralnet.cpp/nn.hpp>
#
#  本工程是**独立 project**（不被仓库根 CMakeLists 引用），单独配置构建：
#      cmake -S examples/downstream -B <builddir>
#  也由仓库 ctest 的 downstream_smoke 用例驱动（cmake/downstream_smoke.cmake）。
#
#  可调开关（全部沿用库侧命名）：
#      -DNN_ENABLE_GPU=OFF|AUTO|ON   GPU 开关（smoke 用 OFF 保快速且无 Vulkan 依赖）
#      -DNN_DOWNSTREAM_FUSION=ON     额外构建"自研 dsl::compute + nn_enable_gpu_fusion"
#                                    形态的目标（要跑全量结构收集 + glslc，数分钟，默认 OFF）
#      -DNEURALNET_CPP_DIR=<path>    库位置（默认 ../..，可指向 3rd_party/neuralnet.cpp）
# ══════════════════════════════════════════════════════════════════════════
cmake_minimum_required(VERSION 3.30)
project(neuralnet_downstream LANGUAGES CXX)

# ① 嵌入库（真实场景这里是 3rd_party/neuralnet.cpp；路径可覆盖）
set(NEURALNET_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.." CACHE PATH
    "Path to the neuralnet.cpp repository")
add_subdirectory("${NEURALNET_CPP_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/neuralnet_cpp")

# ② 一行链接——C++ 标准 / include / 线程库 / （GPU 时）NN_HAS_VULKAN +
#    SPIR-V 嵌入头目录 + Vulkan 链接 + shader 生成顺序，全部由目标传递。
add_executable(downstream_demo main.cpp)
target_link_libraries(downstream_demo PRIVATE neuralnet::nn)

# ctest 约定：77 = 跳过（本样例在 GPU 编译进但无设备时返回 77）
enable_testing()
add_test(NAME downstream_demo COMMAND downstream_demo)
set_tests_properties(downstream_demo PROPERTIES SKIP_RETURN_CODE 77)

# ③ 可选：自研 dsl::compute + nn_enable_gpu_fusion（"写表达式 + 一行 CMake"）
#    复用仓库的 fusion 验收源，但从**消费方作用域**调用函数——这正是
#    树外可用性的关键路径（函数体在调用方作用域求值）。
option(NN_DOWNSTREAM_FUSION "Build the custom dsl::compute + nn_enable_gpu_fusion demo (slow)" OFF)
if(NN_DOWNSTREAM_FUSION)
    if(NOT COMMAND nn_enable_gpu_fusion)
        message(FATAL_ERROR
            "NN_DOWNSTREAM_FUSION=ON requires the library's GPU build "
            "(Vulkan + glslc found and NN_ENABLE_GPU != OFF)")
    endif()
    add_executable(downstream_fusion_demo
        "${NEURALNET_CPP_DIR}/examples/fusion_custom_layer_main.cpp"
        "${NEURALNET_CPP_DIR}/examples/fusion_custom_layer.cpp")
    target_link_libraries(downstream_fusion_demo PRIVATE neuralnet::nn)
    target_include_directories(downstream_fusion_demo PRIVATE
        "${NEURALNET_CPP_DIR}/examples")
    # 从消费方作用域调用——验证函数体内 NN_LIB_SOURCE_DIR/NN_LIB_BINARY_DIR
    # 跨作用域解析正确（旧实现用 CMAKE_SOURCE_DIR，嵌入时指向本工程，会错）。
    nn_enable_gpu_fusion(downstream_fusion_demo
        MAIN "${NEURALNET_CPP_DIR}/examples/fusion_custom_layer_main.cpp")
    add_test(NAME downstream_fusion_demo COMMAND downstream_fusion_demo)
    set_tests_properties(downstream_fusion_demo PROPERTIES SKIP_RETURN_CODE 77)
endif()
