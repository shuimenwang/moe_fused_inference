// mxfp4_binding.cpp - MXFP4 linear 的 torch/pybind 绑定（宿主编译器直接编译）
//
// 说明：torch/extension.h 中 List_inl.h 的 `typename decltype(...)::type`
// 经 nvcc 中转时会被误判（本机宿主 g++11 存在该解析缺陷），因此把绑定从
// .cu（nvcc）中剥离，由宿主 g++12 直接编译本文件；.cu 仅保留纯设备 kernel。

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>

#include <cstdint>

namespace mxfp4 {
// 定义在 mxfp4_linear.cu，纯 C-ABI 风格的 launch 包装。
void launch_mxfp4_linear(const __half* x, const uint8_t* wp, const uint8_t* wsc,
                         float* y, int M, int N, int K, int Kp,
                         cudaStream_t stream);
}  // namespace mxfp4

torch::Tensor mxfp4_linear_forward(torch::Tensor x,    // (M,K) half/bf16
                                   torch::Tensor wp,   // (N,Kp/2) uint8
                                   torch::Tensor wsc)  // (N,Kp/32) uint8
{
    TORCH_CHECK(x.is_cuda(), "x must be cuda");
    TORCH_CHECK(wp.is_cuda() && wsc.is_cuda(), "weights must be cuda");
    TORCH_CHECK(wp.scalar_type() == torch::kUInt8 && wsc.scalar_type() == torch::kUInt8,
                "weight tensors must be uint8");

    // 统一转 half 供 kernel 读取（bf16 输入需先转 half）
    auto xh = x.scalar_type() == torch::kHalf ? x : x.to(torch::kHalf);
    xh = xh.contiguous();
    wp = wp.contiguous();
    wsc = wsc.contiguous();

    int M = x.size(0), K = x.size(1), N = wp.size(0), Kp = wp.size(1) * 2;
    auto y = torch::empty({M, N}, x.options().dtype(torch::kFloat32));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    mxfp4::launch_mxfp4_linear(
        reinterpret_cast<const __half*>(xh.data_ptr()),
        wp.data_ptr<uint8_t>(), wsc.data_ptr<uint8_t>(),
        y.data_ptr<float>(), M, N, K, Kp, stream);
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &mxfp4_linear_forward,
          "MXFP4 block-scale MMA linear (sm_120a): x(M,K) -> (M,N)");
}
