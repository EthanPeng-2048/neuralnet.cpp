# ── cuda/CMakeLists.txt — CUDA 计算后端（独立 nvcc 编译）────────────────────
#
# nvcc 在 Windows 上使用 MSVC 作为 host 编译器。
# 使用 VS 2022 BuildTools (MSVC 14.44) 以获得最佳兼容性。
# .cu 文件由 nvcc 编译为 .obj，链接到主 Clang 项目。
#
# 构建流程：
#   1. nvcc -c cuda_kernels.cu → cuda_kernels.obj
#   2. 主项目 Clang 编译 CudaBackend/CudaEngine（调用内核函数）
#   3. 链接 cuda_kernels.obj + cudart + nvrtc
# ─────────────────────────────────────────────────────────────────────────

# nvcc 的 host compiler 路径（VS 2022 BuildTools）
set(NN_CUDA_HOST_CL "C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe")
set(NN_CUDA_HOST_VCVARS "C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvars64.bat")

# 查找 CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# 设置 SM 目标架构（V100 = sm_70）
if(NOT DEFINED NN_CUDA_ARCH)
    set(NN_CUDA_ARCH "70" CACHE STRING "CUDA SM architecture (e.g. 70, 80, 86, 89)")
endif()

# ── nvcc 编译脚本（在 vcvars 环境中运行）──────────────────────────────────
# 由于 CMake 的 add_custom_command 无法直接设置 vcvars 环境，
# 生成一个临时 .bat 脚本来包装编译命令。
# Debug 模式使用 /MDd（动态调试 CRT），Release 模式使用 /MD（动态发布 CRT）
# 同时对齐 _ITERATOR_DEBUG_LEVEL（主程序 Debug 为 2，Release 为 0），
# 否则 lld-link 会报 /failifmismatch。
if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR (DEFINED CMAKE_CONFIGURATION_TYPES AND CMAKE_BUILD_TYPE MATCHES "Debug"))
    set(NN_CUDA_CRT_FLAG "/MDd")
    set(NN_CUDA_ITDL "2")
else()
    set(NN_CUDA_CRT_FLAG "/MD")
    set(NN_CUDA_ITDL "0")
endif()
set(NN_CUDA_COMPILE_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/compile_cuda_kernels.bat")
# 启用 cuBLAS：定义 NN_HAS_CUBLAS 宏，链接 cublas
file(WRITE "${NN_CUDA_COMPILE_SCRIPT}"
    "@echo off\r\n"
    "call \"${NN_CUDA_HOST_VCVARS}\" >nul 2>&1\r\n"
    "\"${CUDAToolkit_BIN_DIR}/nvcc.exe\" -c -O3 --use_fast_math -allow-unsupported-compiler"
    " -Xcompiler ${NN_CUDA_CRT_FLAG}"
    " -Xcompiler /D_ITERATOR_DEBUG_LEVEL=${NN_CUDA_ITDL}"
    " -arch=sm_${NN_CUDA_ARCH} -DNN_HAS_CUBLAS"
    " -I\"${CMAKE_CURRENT_SOURCE_DIR}/../include\""
    " -I\"${CUDAToolkit_INCLUDE_DIRS}\""
    " \"%~1\" -o \"%~2\"\r\n"
    "exit /b %errorlevel%\r\n"
)

# ── CUDA 内核 .obj ──────────────────────────────────────────────────────
set(NN_CUDA_KERNELS_OBJ "${CMAKE_CURRENT_BINARY_DIR}/cuda_kernels.obj")

add_custom_command(
    OUTPUT ${NN_CUDA_KERNELS_OBJ}
    COMMAND cmd /c "${NN_CUDA_COMPILE_SCRIPT}" "${CMAKE_CURRENT_SOURCE_DIR}/cuda_kernels.cu" "${NN_CUDA_KERNELS_OBJ}"
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cuda_kernels.cu
    COMMENT "[CUDA] Compiling cuda_kernels.cu with nvcc (sm_${NN_CUDA_ARCH})"
)

add_custom_target(compile_cuda_kernels DEPENDS ${NN_CUDA_KERNELS_OBJ})

# ── 导出变量给父项目 ────────────────────────────────────────────────────
set(NN_CUDA_KERNELS_OBJ ${NN_CUDA_KERNELS_OBJ} PARENT_SCOPE)
set(NN_CUDA_FOUND TRUE PARENT_SCOPE)
set(NN_CUDA_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)
