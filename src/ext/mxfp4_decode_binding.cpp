// mxfp4_decode_binding.cpp - P2.2 decode kernel 的 torch/pybind 绑定
//
// 与 mxfp4_binding.cpp 同样的双源策略：torch/pybind 只出现在宿主 g++12
// 直接编译的 .cpp；mxfp4_decode.cu 仅含纯设备 kernel + C-ABI launch。

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>

#include <cstdint>
#include <vector>

namespace mxfp4v2 {
void launch_prequant_v2(const void* x, uint8_t* xq, uint8_t* xsc,
                        int M, int K, int Kp, cudaStream_t stream);
void launch_lin_pq_v2(const uint8_t* xq, const uint8_t* xsc,
                      const uint8_t* wp, const uint8_t* wsc, float* y,
                      int M, int N, int K, int Kp, cudaStream_t stream);
void launch_gateup_v2(const uint8_t* xq, const uint8_t* xsc,
                      const uint8_t* const* wp_arr,
                      const uint8_t* const* wsc_arr,
                      float* gu, int E, int M, int N, int K, int Kp,
                      cudaStream_t stream);
void launch_down_silu_v2(const float* gu,
                         const uint8_t* const* wp_arr,
                         const uint8_t* const* wsc_arr,
                         float* yd, int E, int M, int N, int K, int Kp,
                         cudaStream_t stream);
void launch_combine_v2(const float* yd, int ystride, const float* w, int E,
                       float* out, int M, int N, cudaStream_t stream);
void launch_attn_decode_v2(const void* q, const uint8_t* kvk, const uint8_t* kvv,
                           float* sc, float* o, const int* dpos,
                           int Hq, int Hkv, int D, int max_pos,
                           int M, cudaStream_t stream);
void launch_kvwrite_v2(const void* k, const void* v, uint8_t* kvk, uint8_t* kvv,
                       const int* dpos, int M, int Hkv, int D, int max_pos,
                       cudaStream_t stream);
}  // namespace mxfp4v2

namespace {

// 把一组 cuda uint8 tensor 的 data_ptr 组装成设备上的指针表（int64 承载指针）。
torch::Tensor make_ptr_table(const std::vector<torch::Tensor>& ts) {
    TORCH_CHECK(ts.size() > 0, "empty pointer table");
    std::vector<int64_t> host;
    host.reserve(ts.size());
    for (const auto& t : ts) {
        TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kUInt8,
                    "ptr table entries must be cuda uint8");
        host.push_back(reinterpret_cast<int64_t>(t.data_ptr<uint8_t>()));
    }
    auto opts = torch::TensorOptions().dtype(torch::kInt64).device(ts[0].device());
    auto dev = torch::empty({static_cast<long>(host.size())}, opts);
    auto cpu = torch::from_blob(host.data(),
                                {static_cast<long>(host.size())},
                                torch::TensorOptions().dtype(torch::kInt64));
    dev.copy_(cpu, /*non_blocking=*/false);
    return dev;
}

}  // namespace

// x: (M,K) bf16 cuda -> [xq (M,Kp/2) u8, xsc (M,Kp/32) u8]，Kp = ceil32(K)
std::vector<torch::Tensor> prequant(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be cuda");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bfloat16");
    x = x.contiguous();
    int M = static_cast<int>(x.size(0));
    int K = static_cast<int>(x.size(1));
    int Kp = (K + 31) / 32 * 32;
    auto opts = x.options().dtype(torch::kUInt8);
    auto xq = torch::zeros({M, Kp / 2}, opts);  // 尾部 padding 字节必须为 0 nibble
    auto xsc = torch::empty({M, Kp / 32}, opts);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    mxfp4v2::launch_prequant_v2(
        reinterpret_cast<const void*>(x.data_ptr()),
        xq.data_ptr<uint8_t>(), xsc.data_ptr<uint8_t>(),
        M, K, Kp, stream);
    return {xq, xsc};
}

// 静态投影（xq/xsc 预量化）：y (M,N) f32
torch::Tensor lin_pq(torch::Tensor xq, torch::Tensor xsc,
                     torch::Tensor wp, torch::Tensor wsc,
                     int64_t M_, int64_t N_, int64_t K_) {
    xq = xq.contiguous(); xsc = xsc.contiguous();
    wp = wp.contiguous(); wsc = wsc.contiguous();
    int M = static_cast<int>(M_), N = static_cast<int>(N_), K = static_cast<int>(K_);
    int Kp = static_cast<int>(wp.size(1)) * 2;
    auto y = torch::empty({M, N}, xq.options().dtype(torch::kFloat32));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    mxfp4v2::launch_lin_pq_v2(
        xq.data_ptr<uint8_t>(), xsc.data_ptr<uint8_t>(),
        wp.data_ptr<uint8_t>(), wsc.data_ptr<uint8_t>(),
        y.data_ptr<float>(), M, N, K, Kp, stream);
    return y;
}

// grouped gate/up：gu (E,2N) f32。wps/wscs 长度 2E：[gate_e0..eE, up_e0..eE]
torch::Tensor moe_gateup(torch::Tensor xq, torch::Tensor xsc,
                         const std::vector<torch::Tensor>& wps,
                         const std::vector<torch::Tensor>& wscs,
                         int64_t E_, int64_t M_, int64_t N_, int64_t K_) {
    int E = static_cast<int>(E_), M = static_cast<int>(M_);
    TORCH_CHECK(static_cast<int64_t>(wps.size()) == 2 * E_ * M_, "wps must be 2E*M");
    TORCH_CHECK(static_cast<int64_t>(wscs.size()) == 2 * E_ * M_, "wscs must be 2E*M");
    int N = static_cast<int>(N_), K = static_cast<int>(K_);
    int Kp = static_cast<int>(wps[0].size(1)) * 2;
    auto wp_tab = make_ptr_table(wps);
    auto wsc_tab = make_ptr_table(wscs);
    auto gu = torch::empty({(int64_t)M * E, 2 * N}, xq.options().dtype(torch::kFloat32));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    mxfp4v2::launch_gateup_v2(
        xq.data_ptr<uint8_t>(), xsc.data_ptr<uint8_t>(),
        reinterpret_cast<const uint8_t* const*>(wp_tab.data_ptr<int64_t>()),
        reinterpret_cast<const uint8_t* const*>(wsc_tab.data_ptr<int64_t>()),
        gu.data_ptr<float>(), E, M, N, K, Kp, stream);
    return gu;
}

// fused silu+down：yd (E,N) f32
torch::Tensor moe_down(torch::Tensor gu,
                       const std::vector<torch::Tensor>& wps,
                       const std::vector<torch::Tensor>& wscs,
                       int64_t E_, int64_t M_, int64_t N_, int64_t K_) {
    int E = static_cast<int>(E_), M = static_cast<int>(M_),
        N = static_cast<int>(N_), K = static_cast<int>(K_);
    TORCH_CHECK(static_cast<int64_t>(wps.size()) == E_ * M_, "wps must be E*M");
    int Kp = static_cast<int>(wps[0].size(1)) * 2;
    TORCH_CHECK(gu.size(0) == (int64_t)M * E && gu.size(1) == 2 * K, "gu (M*E,2K)");
    auto wp_tab = make_ptr_table(wps);
    auto wsc_tab = make_ptr_table(wscs);
    auto yd = torch::empty({(int64_t)M * E, N}, gu.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    mxfp4v2::launch_down_silu_v2(
        gu.data_ptr<float>(),
        reinterpret_cast<const uint8_t* const*>(wp_tab.data_ptr<int64_t>()),
        reinterpret_cast<const uint8_t* const*>(wsc_tab.data_ptr<int64_t>()),
        yd.data_ptr<float>(), E, M, N, K, Kp, stream);
    return yd;
}

// combine：out[c] = sum_e w_e * yd[e,c]
torch::Tensor moe_combine(torch::Tensor yd, torch::Tensor w, int64_t M_) {
    yd = yd.contiguous(); w = w.contiguous();
    int M = static_cast<int>(M_);
    int E = static_cast<int>(w.numel()) / M;
    int N = static_cast<int>(yd.size(1));
    TORCH_CHECK((int64_t)M * E == w.numel(), "w (M,E)");
    TORCH_CHECK(yd.size(0) == (int64_t)M * E, "yd (M*E,N)");
    auto out = torch::empty({M, N}, yd.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    mxfp4v2::launch_combine_v2(
        yd.data_ptr<float>(), N, w.data_ptr<float>(), E,
        out.data_ptr<float>(), M, N, stream);
    return out;
}

// graph 友好 KV 写入：k/v (Hkv,D) bf16 stream；kvk/kvv 为某层 slice buffer
void kv_write(torch::Tensor k, torch::Tensor v, torch::Tensor kvk, torch::Tensor kvv,
              torch::Tensor dpos, int64_t M_) {
    int M = (int)M_;
    int Hkv = (int)kvk.size(1), D = (int)kvk.size(2);
    int max_pos = (int)kvk.size(0);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    mxfp4v2::launch_kvwrite_v2(
        reinterpret_cast<const void*>(k.data_ptr()),
        reinterpret_cast<const void*>(v.data_ptr()),
        kvk.data_ptr<uint8_t>(), kvv.data_ptr<uint8_t>(),
        reinterpret_cast<const int*>(dpos.data_ptr()),
        M, Hkv, D, max_pos, stream);
}

// ---- graph 友好版：指针设备表内容随 replay 更新（不重建 python 列表）----
// wps_tab/wscs_tab: (2E,) / (E,) int64 设备表，内容由 host 在 graph 外更新
torch::Tensor moe_gateup_tab(torch::Tensor xq, torch::Tensor xsc,
                             torch::Tensor wps_tab, torch::Tensor wscs_tab,
                             int64_t E_, int64_t M_, int64_t N_, int64_t K_) {
    int E = (int)E_, M = (int)M_, N = (int)N_, K = (int)K_;
    int64_t* wpt = wps_tab.data_ptr<int64_t>();
    int64_t* wst = wscs_tab.data_ptr<int64_t>();
    int Kp = (int)xq.size(1) * 2;      // gate/up K=hidden → Kp=1944 取整后的偶数尺寸
    auto gu = torch::empty({(int64_t)M * E, 2 * N}, xq.options().dtype(torch::kFloat32));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    mxfp4v2::launch_gateup_v2(
        xq.data_ptr<uint8_t>(), xsc.data_ptr<uint8_t>(),
        reinterpret_cast<const uint8_t* const*>(wpt),
        reinterpret_cast<const uint8_t* const*>(wst),
        gu.data_ptr<float>(), E, M, N, K, Kp, stream);
    return gu;
}

torch::Tensor moe_down_tab(torch::Tensor gu,
                           torch::Tensor wps_tab, torch::Tensor wscs_tab,
                           int64_t E_, int64_t M_, int64_t N_, int64_t K_) {
    int E = (int)E_, M = (int)M_, N = (int)N_, K = (int)K_;
    int64_t* wpt = wps_tab.data_ptr<int64_t>();
    int64_t* wst = wscs_tab.data_ptr<int64_t>();
    int Kp = ((gu.size(1) / 2) + 31) / 32 * 32;  // down K=Mi
    auto yd = torch::empty({(int64_t)M * E, N}, gu.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    mxfp4v2::launch_down_silu_v2(
        gu.data_ptr<float>(),
        reinterpret_cast<const uint8_t* const*>(wpt),
        reinterpret_cast<const uint8_t* const*>(wst),
        yd.data_ptr<float>(), E, M, N, K, Kp, stream);
    return yd;
}

// decode attention：q (Hq,D) bf16 roped；kvk/kvv (max_pos,Hkv,D) fp8；
// dpos (1,) int32（内容可在 graph replay 间更新）；sc (Hq,max_pos) f32 复用 buffer
torch::Tensor attn_decode(torch::Tensor q, torch::Tensor kvk, torch::Tensor kvv,
                          torch::Tensor dpos, torch::Tensor sc, int64_t M_) {
    // q (M,Hq,D) bf16；kvk/kvv (max_pos,Hkv,D) fp8；dpos=base；sc (M,Hq,max_pos)
    q = q.contiguous();
    int M = (int)M_;
    int Hq = static_cast<int>(q.size(1));
    int Hkv = static_cast<int>(kvk.size(1));
    int D = static_cast<int>(q.size(2));
    int max_pos = static_cast<int>(kvk.size(0));
    auto o = torch::empty({M, Hq, D}, q.options().dtype(torch::kFloat32));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    mxfp4v2::launch_attn_decode_v2(
        reinterpret_cast<const void*>(q.data_ptr()), kvk.data_ptr<uint8_t>(),
        kvv.data_ptr<uint8_t>(), sc.data_ptr<float>(), o.data_ptr<float>(),
        reinterpret_cast<const int*>(dpos.data_ptr()),
        Hq, Hkv, D, max_pos, M, stream);
    return o;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("prequant", &prequant,
          "bf16 (M,K) -> [xq (M,Kp/2) u8, xsc (M,Kp/32) u8]");
    m.def("lin_pq", &lin_pq, "prequantized static MXFP4 linear -> f32 (M,N)");
    m.def("moe_gateup", &moe_gateup, "grouped gate+up (2E ptrs) -> f32 (E,2N)");
    m.def("moe_down", &moe_down, "fused silu(gate)*up then down -> f32 (M*E,N)");
    m.def("moe_combine", &moe_combine, "out[m,c]=sum_e w[m,e]*yd[m,e,c]");
    m.def("attn_decode", &attn_decode, "M=1 graph-friendly decode attention (fp8 kv)");
    m.def("moe_gateup_tab", &moe_gateup_tab, "grouped gate+up from device ptr table");
    m.def("moe_down_tab", &moe_down_tab, "fused silu-down from device ptr table");
    m.def("kv_write", &kv_write, "fp8 KV write M tokens（读 dev dpos base）");
}
