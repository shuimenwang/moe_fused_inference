#!/usr/bin/env python3
# fp4_ext.py - 编译/加载自研 MXFP4 MMA torch 扩展（sm_120a）
#
# 双源架构：
#   mxfp4_linear.cu   —— nvcc 编译的纯设备 kernel + C-ABI launch（不含 torch 头）
#   mxfp4_binding.cpp —— 宿主 g++12 直接编译的 torch/pybind 绑定
# torch/extension.h 经 nvcc 中转会触发本机 g++11 的 decltype 解析缺陷，故拆分。

import os

from torch.utils.cpp_extension import load

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CU = os.path.join(_ROOT, "src", "ext", "mxfp4_linear.cu")
_CPP = os.path.join(_ROOT, "src", "ext", "mxfp4_binding.cpp")
_BUILD = os.path.join(_ROOT, "build", "torch_ext")
# 独立 conda gcc12 工具链（系统 g++11 有缺陷且无 sudo）。
_GCC_BIN = "/home/hngs/miniconda3/envs/gcc12/bin/x86_64-conda-linux-gnu-g++"


def get_ext():
    os.makedirs(_BUILD, exist_ok=True)
    # 必须 120a 变体：默认探测仅得 capability(12,0) 会发 plain sm_120，
    # block_scale MMA PTX 在 plain compute_120 下编译失败（仅 sm_120a 支持）。
    os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0a+PTX"
    # CXX 决定 .cpp 的编译器与最终链接；不设 CC 以免 cpp_extension 注入第二个 -ccbin。
    os.environ["CXX"] = _GCC_BIN
    os.environ.pop("CC", None)
    return load(
        name="mxfp4_linear_ext",
        sources=[_CU, _CPP],
        extra_cuda_cflags=[
            "-O3",
            f"-ccbin={_GCC_BIN}",
        ],
        extra_cflags=["-O3"],
        build_directory=_BUILD,
        verbose=os.environ.get("FP4_EXT_VERBOSE", "0") == "1",
    )


_CU_D = os.path.join(_ROOT, "src", "ext", "mxfp4_decode.cu")
_CPP_D = os.path.join(_ROOT, "src", "ext", "mxfp4_decode_binding.cpp")


def get_decode_ext():
    """P2.2 decode 专用 kernel（prequant/lin_pq/grouped MoE），独立模块。"""
    os.makedirs(_BUILD, exist_ok=True)
    os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0a+PTX"
    os.environ["CXX"] = _GCC_BIN
    os.environ.pop("CC", None)
    return load(
        name="mxfp4_decode_ext",
        sources=[_CU_D, _CPP_D],
        extra_cuda_cflags=[
            "-O3",
            f"-ccbin={_GCC_BIN}",
        ],
        extra_cflags=["-O3"],
        build_directory=_BUILD,
        verbose=os.environ.get("FP4_EXT_VERBOSE", "0") == "1",
    )
