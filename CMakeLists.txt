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

    # 定义着色器列表：名称 → 源文件（无后缀）
    set(GPU_SHADERS matmul matmul_tiled elementwise)
    set(GPU_SHADER_SPVS "")
    set(GPU_SHADER_HPPS "")

    foreach(SHADER_NAME ${GPU_SHADERS})
        set(SRC ${CMAKE_SOURCE_DIR}/shaders/${SHADER_NAME}.comp)
        set(SPV ${CMAKE_BINARY_DIR}/${SHADER_NAME}.spv)
        set(HPP ${CMAKE_BINARY_DIR}/generated/${SHADER_NAME}_spv.hpp)
        set(FUNC "nn_${SHADER_NAME}_spirv_bytecode")
        list(APPEND GPU_SHADER_SPVS ${SPV})
        list(APPEND GPU_SHADER_HPPS ${HPP})

        if(GLSLC)
            add_custom_command(
                OUTPUT ${SPV}
                COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/generated
                COMMAND ${GLSLC} -fshader-stage=compute ${SRC} -o ${SPV}
                DEPENDS ${SRC}
                COMMENT "[GPU] Compiling ${SHADER_NAME}.comp → SPIR-V"
            )
        elseif(GLSLANG_VALIDATOR)
            add_custom_command(
                OUTPUT ${SPV}
                COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/generated
                COMMAND ${GLSLANG_VALIDATOR} -V ${SRC} -o ${SPV}
                DEPENDS ${SRC}
                COMMENT "[GPU] Compiling ${SHADER_NAME}.comp → SPIR-V"
            )
        endif()

        add_custom_command(
            OUTPUT ${HPP}
            COMMAND ${CMAKE_COMMAND}
                -DSPV_INPUT=${SPV}
                -DHPP_OUTPUT=${HPP}
                -DFUNC_NAME=${FUNC}
                -P ${CMAKE_SOURCE_DIR}/cmake/embed_spirv.cmake
            DEPENDS ${SPV} ${CMAKE_SOURCE_DIR}/cmake/embed_spirv.cmake
            COMMENT "[GPU] Embedding ${SHADER_NAME} SPIR-V into C++ header"
        )
    endforeach()

    if(NOT GLSLC AND NOT GLSLANG_VALIDATOR)
        message(WARNING "No SPIR-V compiler found (glslc or glslangValidator). "
                        "GPU acceleration disabled. Install Vulkan SDK.")
        set(Vulkan_FOUND OFF)
    endif()

    if(Vulkan_FOUND)
        add_custom_target(compile_gpu_shaders DEPENDS ${GPU_SHADER_HPPS})

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
