# build.cmake —— 纯 CMake + Clang++ 构建脚本，不依赖任何第三方构建工具
cmake_minimum_required(VERSION 3.20)

set(CLANGXX "C:/Program Files/LLVM/bin/clang++.exe")
set(CMAKE_CXX_COMPILER ${CLANGXX})

set(SRC_DIR  "${CMAKE_SOURCE_DIR}/src")
set(INC_DIR  "${CMAKE_SOURCE_DIR}/include")
set(OUT_DIR  "${CMAKE_SOURCE_DIR}/build")
file(MAKE_DIRECTORY "${OUT_DIR}")

# 构建 mnist_train
set(TRAIN_TARGET "${OUT_DIR}/mnist_train.exe")
set(TRAIN_SOURCES "${SRC_DIR}/train.cpp")

# 构建 mnist_infer
set(INFER_TARGET "${OUT_DIR}/mnist_infer.exe")
set(INFER_SOURCES "${SRC_DIR}/infer.cpp")

set(FLAGS
    -std=c++26
    -O3
    -march=native
    -Wall -Wextra
    -fcolor-diagnostics
)

if(WIN32)
    # Windows 下链接必需要素
    list(APPEND FLAGS -fuse-ld=lld)
endif()

# 编译 mnist_train
execute_process(
    COMMAND ${CLANGXX} ${FLAGS} ${TRAIN_SOURCES}
            -I ${INC_DIR}
            -o ${TRAIN_TARGET}
    RESULT_VARIABLE result
    COMMAND_ECHO STDOUT
)

if(NOT result EQUAL 0)
    message(FATAL_ERROR "Compilation failed")
endif()

message(STATUS "Build successful: ${TRAIN_TARGET}")

# 编译 mnist_infer
execute_process(
    COMMAND ${CLANGXX} ${FLAGS} ${INFER_SOURCES}
            -I ${INC_DIR}
            -o ${INFER_TARGET}
    RESULT_VARIABLE result
    COMMAND_ECHO STDOUT
)

if(NOT result EQUAL 0)
    message(FATAL_ERROR "Compilation failed")
endif()

message(STATUS "Build successful: ${INFER_TARGET}")
