cmake_minimum_required(VERSION 3.20)
project(neuralnet.cpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 为所有目标添加编译标志
add_compile_options(-fexperimental-library -O3 -ffast-math -Wall -Wextra -Wpedantic -Werror)

# ── MNIST 目标 ──────────────────────────────────────────────────────────────
add_executable(mnist_train src/mnist_train.cpp)
add_executable(mnist_infer src/mnist_infer.cpp)

target_include_directories(mnist_train PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(mnist_infer PRIVATE ${CMAKE_SOURCE_DIR}/include)

# ── 文本生成目标 ────────────────────────────────────────────────────────────
add_executable(text_train src/text_train.cpp)
add_executable(text_infer src/text_infer.cpp)

target_include_directories(text_train PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(text_infer PRIVATE ${CMAKE_SOURCE_DIR}/include)

# ── 分词器目标 ──────────────────────────────────────────────────────────────
add_executable(tokenizer_train src/tokenizer_train.cpp)
add_executable(tokenizer_infer src/tokenizer_infer.cpp)

target_include_directories(tokenizer_train PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(tokenizer_infer PRIVATE ${CMAKE_SOURCE_DIR}/include)

# ── GPU 测试目标 ──────────────────────────────────────────────────────────
add_executable(gpu_test src/gpu_test.cpp)
target_include_directories(gpu_test PRIVATE ${CMAKE_SOURCE_DIR}/include)

# ── Vulkan GPU 加速（可选） ────────────────────────────────────────────────
find_package(Vulkan QUIET)
if(Vulkan_FOUND)
    message(STATUS "Vulkan found: ${Vulkan_LIBRARY}")

    # 查找 SPIR-V 编译器（Vulkan SDK 自带）
    find_program(GLSLANG_VALIDATOR glslangValidator HINTS ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE})
    find_program(GLSLC glslc HINTS ${Vulkan_GLSLC_EXECUTABLE})

    set(SHADER_SRC ${CMAKE_SOURCE_DIR}/shaders/matmul.comp)
    set(SHADER_SPV ${CMAKE_BINARY_DIR}/matmul.spv)
    set(SHADER_HPP ${CMAKE_BINARY_DIR}/generated/matmul_spv.hpp)

    if(GLSLC)
        message(STATUS "Found glslc: ${GLSLC}")
        add_custom_command(
            OUTPUT ${SHADER_SPV}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/generated
            COMMAND ${GLSLC} -fshader-stage=compute ${SHADER_SRC} -o ${SHADER_SPV}
            DEPENDS ${SHADER_SRC}
            COMMENT "[GPU] Compiling matmul.comp → SPIR-V (glslc)"
        )
    elseif(GLSLANG_VALIDATOR)
        message(STATUS "Found glslangValidator: ${GLSLANG_VALIDATOR}")
        add_custom_command(
            OUTPUT ${SHADER_SPV}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/generated
            COMMAND ${GLSLANG_VALIDATOR} -V ${SHADER_SRC} -o ${SHADER_SPV}
            DEPENDS ${SHADER_SRC}
            COMMENT "[GPU] Compiling matmul.comp → SPIR-V (glslangValidator)"
        )
    else()
        message(WARNING "No SPIR-V compiler found (glslc or glslangValidator). "
                        "GPU acceleration disabled. Install Vulkan SDK.")
        set(Vulkan_FOUND OFF)
    endif()

    if(Vulkan_FOUND)
        # SPIR-V → C++ 头文件（嵌入为 constexpr 数组）
        add_custom_command(
            OUTPUT ${SHADER_HPP}
            COMMAND ${CMAKE_COMMAND}
                -DSPV_INPUT=${SHADER_SPV}
                -DHPP_OUTPUT=${SHADER_HPP}
                -P ${CMAKE_SOURCE_DIR}/cmake/embed_spirv.cmake
            DEPENDS ${SHADER_SPV} ${CMAKE_SOURCE_DIR}/cmake/embed_spirv.cmake
            COMMENT "[GPU] Embedding SPIR-V into C++ header"
        )

        add_custom_target(compile_gpu_shaders DEPENDS ${SHADER_HPP})

        # 为所有目标启用 Vulkan
        set(NN_GPU_SOURCES
            mnist_train mnist_infer
            text_train text_infer
            tokenizer_train tokenizer_infer
            gpu_test
        )
        foreach(TARGET ${NN_GPU_SOURCES})
            target_compile_definitions(${TARGET} PRIVATE NN_HAS_VULKAN)
            target_include_directories(${TARGET} PRIVATE ${CMAKE_BINARY_DIR}/generated)
            target_link_libraries(${TARGET} PRIVATE Vulkan::Vulkan)
            add_dependencies(${TARGET} compile_gpu_shaders)
        endforeach()

        message(STATUS "GPU acceleration enabled (Vulkan compute)")
    endif()
else()
    message(STATUS "Vulkan not found. GPU acceleration disabled.")
endif()
