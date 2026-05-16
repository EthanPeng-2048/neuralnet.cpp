cmake_minimum_required(VERSION 3.20)
project(neuralnet.cpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 为所有目标添加编译标志
add_compile_options(-fexperimental-library)

# 定义可执行文件
add_executable(mnist_train src/train.cpp)
add_executable(mnist_infer src/infer.cpp)

# 为特定目标添加包含目录和链接库
target_include_directories(mnist_train PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(mnist_infer PRIVATE ${CMAKE_SOURCE_DIR}/include)

# 为特定目标链接 TBB 库
target_link_libraries(mnist_train PRIVATE tbb)
target_link_libraries(mnist_infer PRIVATE tbb)

