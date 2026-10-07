# CMakeLists.txt - Qwen3-30B-A3B FP4 Inference on RTX 5060
cmake_minimum_required(VERSION 3.20)

# 必须在 project() 前关闭 CMake 自动 arch 标志：
# CMake 3.20 不认识 sm_120，native 探测会回退注入 sm_75，与下面的 120a gencode 冲突
set(CMAKE_CUDA_ARCHITECTURES OFF)

project(moe_fused_inference LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# ------------------------------------------------------------------
# 目标硬件：RTX 5060，probe kernel 实测为 sm_120a 变体
#   - FP4 block-scale MMA（MXFP4/NVFP4）需要 120a，不能只写 sm_120
#   - 变体（a/f）nvcc 无法通过 arch 列表区分，用 -gencode 显式指定
#   - 同时保留 compute_120a PTX 作为驱动 JIT 兜底
# ------------------------------------------------------------------
set(CMAKE_CUDA_FLAGS
    "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_120a,code=sm_120a -gencode=arch=compute_120a,code=compute_120a")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")

# ------------------------------------------------------------------
# P0 硬件探针：FP8 / MXFP4 / NVFP4 mma.sync 存在性 + 硬件档案打印
# ------------------------------------------------------------------
add_executable(probe_fp4_mma quant/probe_mxfp4.cu)

# ------------------------------------------------------------------
# 后续目标（随对应 Package 实现取消注释）
# ------------------------------------------------------------------
# P0 极简正确性 MXFP4 MMA 算子扩展（torch 绑定部分在 Python 侧）：
# add_library(mxfp4_linear STATIC src/ext/mxfp4_linear.cu)
#
# P2 高性能算子：
# add_library(moe_kernels STATIC
#     src/kernels/k1_router_topk_sort.cu
#     src/kernels/k2_mxfp4_grouped_gemm.cu
#     src/kernels/k3_deterministic_combine.cu)
# target_include_directories(moe_kernels PUBLIC src/kernels)
