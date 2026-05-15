cmake_minimum_required(VERSION 3.20)
project(neuralnet.cpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(mnist_train src/train.cpp)
add_executable(mnist_infer src/infer.cpp)

target_include_directories(mnist_train PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(mnist_infer PRIVATE ${CMAKE_SOURCE_DIR}/include)
