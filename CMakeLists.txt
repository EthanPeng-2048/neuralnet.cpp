cmake_minimum_required(VERSION 3.20)
project(neuralnet.cpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 为所有目标添加编译标志
add_compile_options(-fexperimental-library -O3 -ffast-math -Wall -Wextra -Wpedantic -Werror -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function)

# 定义可执行文件
add_executable(mnist_train src/train.cpp)
add_executable(mnist_infer src/infer.cpp)

# 为特定目标添加包含目录和链接库
target_include_directories(mnist_train PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(mnist_infer PRIVATE ${CMAKE_SOURCE_DIR}/include)

# 查找并链接 TBB 库
find_package(TBB REQUIRED)
target_link_libraries(mnist_train PRIVATE TBB::tbb)
target_link_libraries(mnist_infer PRIVATE TBB::tbb)

