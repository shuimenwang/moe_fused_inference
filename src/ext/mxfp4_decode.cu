// mxfp4_decode.cu - P2.2 decode 专用 MXFP4 kernel（sm_120a）
//
// 相对 P0 朴素 kernel 的结构性改动：
//   1. 激活预量化外提（prequant）：q/k/v 共享 hp；router+8 专家 gate/up 共享 hp2
//   2. 权重走 cp.async 16B 批量 GMEM→SMEM（朴素版逐 nibble 标量 gather）
//   3. grouped MoE：gate+up 单 launch / fused silu+down 单 launch / combine 单 launch
//      （朴素版 8 专家 ×3 投影 ×48 层 = 1152 launch + 16× M 行浪费）
//
// decode 分组 kernel 假设 M=1（m16 tile 仅 row0 有效，pad 行清 0；权重 B 一次加载
// 仍服务 16 行，浪费的是 MMA 发射槽而非权重带宽——GEMV 场景由权重带宽决定）。

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "mxfp4_common.cuh"

namespace mxfp4v2 {

// =================================================================
// 1) 激活预量化：x (M,K) bf16 → xq (M,Kp/2) u8 + xsc (M,Kp/32) u8
//    grid (M, Kp/32)，block=32（一个 warp 处理一个 1×32 block）
// =================================================================
__global__ void prequant_kernel(const __nv_bfloat16* __restrict__ x,
                                uint8_t* __restrict__ xq,
                                uint8_t* __restrict__ xsc,
                                int K, int Kp)
{
    const int row = blockIdx.x;
    const int kb = blockIdx.y;
    const int t = threadIdx.x;
    const int k0 = kb * BLOCK_K;

    // 每 block 处理一个 1×32 block：每线程 1 元素，偶线程与相邻奇线程打包成 nibble
    float v = 0.0f;
    if (k0 + t < K)
        v = __bfloat162float(x[(long)row * K + k0 + t]);
    float am = fabsf(v);
    #pragma unroll
    for (int off = 16; off >= 1; off >>= 1)
        am = fmaxf(am, __shfl_xor_sync(0xFFFFFFFF, am, off));
    uint8_t code = e8m0_code(am);
    float sv = scale_value(code);
    uint32_t nib = quantize_nibble(v, sv);
    uint32_t nib1 = __shfl_xor_sync(0xFFFFFFFF, nib, 1);  // 相邻的下一元素
    if ((t & 1) == 0 && k0 + t + 1 <= K)
        xq[(long)row * (Kp / 2) + kb * 16 + (t >> 1)] = (uint8_t)(nib | (nib1 << 4));
    if (t == 0)
        xsc[(long)row * (Kp / BLOCK_K) + kb] = code;
}

// =================================================================
// SMEM 布局（每 block，~6.6KB）
// =================================================================
struct TileSmem {
    uint8_t sA[BM * 32];      // [16][32B] A k64 chunk
    uint8_t sAsc[BM * 2];     // [16][2]（每行 kb0/kb1，mma_tile 按 [r*2+b]）
    uint8_t sB[BN * 32];      // [32][32B] B k64 chunk
    uint8_t sBsc[BN * 2];     // [32][2]（每行 kb0/kb1，mma_tile 按 [n*2+b]）
    float sAg[64];            // down fused：gate 值（row0）
    float sAu[64];            // down fused：up 值（row0）
};

__device__ __forceinline__ void clear_a(TileSmem* s) {
    uint32_t* p = reinterpret_cast<uint32_t*>(s->sA);
    for (int i = threadIdx.x; i < BM * 32 / 4; i += blockDim.x)
        p[i] = 0;
    uint16_t* q = reinterpret_cast<uint16_t*>(s->sAsc);
    for (int i = threadIdx.x; i < BM; i += blockDim.x)
        q[i] = 0x7F7Fu;                       // 每行两 byte=2^0（唯一合法 pad）
}

__device__ __forceinline__ void load_b(TileSmem* s,
                                       const uint8_t* wp, int sp,
                                       const uint8_t* wsc, int nsp,
                                       int n0, int N, int kk)
{
    const int tid = threadIdx.x;
    if (tid < 64) {                            // 32 行 × 2 个 16B
        int row = tid >> 1, half = tid & 1;
        int n = n0 + row;
        void* dst = &s->sB[row * 32 + half * 16];
        if (n < N) {
            cp_async_16(dst, wp + (long)n * sp + kk / 2 + half * 16);
        } else {
            *reinterpret_cast<uint4*>(dst) = uint4{0, 0, 0, 0};
        }
    }
    if (tid < BN) {                            // B scale 标量（行跨 nsp）
        int n = n0 + tid;
        s->sBsc[tid * 2 + 0] = (n < N) ? wsc[(long)n * nsp + kk / 32] : 127;
        s->sBsc[tid * 2 + 1] = (n < N) ? wsc[(long)n * nsp + kk / 32 + 1] : 127;
    }
}

__device__ __forceinline__ void load_a_cp(TileSmem* s,
                                          const uint8_t* xq, int xsp,
                                          const uint8_t* xsc, int xsc_sp,
                                          int m0, int M, int kk)
{
    const int tid = threadIdx.x;
    if (tid >= 64 && tid < 96) {               // 16 行 × 2 个 16B
        int row = (tid - 64) >> 1, half = (tid - 64) & 1;
        if (m0 + row < M)
            cp_async_16(&s->sA[row * 32 + half * 16],
                        xq + (long)(m0 + row) * xsp + kk / 2 + half * 16);
    }
    __syncthreads();                           // 等 clear_a 完成
    if (tid < BM * 2) {
        int row = tid >> 1, b = tid & 1;
        s->sAsc[row * 2 + b] = (m0 + row < M)
            ? xsc[(long)(m0 + row) * xsc_sp + kk / 32 + b] : (uint8_t)127;
    }
}

__device__ __forceinline__ void write_d(const float d[4], float* y,
                                        long ystride, int n0, int n_valid)
{
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2, tig = lane & 3;
    const int warp = threadIdx.x >> 5;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int row = (i < 2) ? g : g + 8;
        int col = tig * 2 + (i & 1);
        int gn = n0 + warp * BN_WARP + col;
        if (row == 0 && gn < n_valid)          // decode M=1：只写 row0
            y[gn] = d[i];
    }
}

// =================================================================
// 2) 单张量线性（静态投影 / prefill 专家）：grid (ceil(M/16), ceil(N/32))
// =================================================================
__global__ void lin_pq_kernel(
        const uint8_t* __restrict__ xq, const uint8_t* __restrict__ xsc,
        const uint8_t* __restrict__ wp, const uint8_t* __restrict__ wsc,
        float* __restrict__ y,
        int M, int N, int K, int Kp)
{
    extern __shared__ unsigned char smem_raw[];
    TileSmem* s = reinterpret_cast<TileSmem*>(smem_raw);
    const int warp = threadIdx.x >> 5;
    const int m0 = blockIdx.x * BM;
    const int n0 = blockIdx.y * BN;
    const int sp = Kp / 2, nsp = Kp / BLOCK_K;

    float d[4] = {0, 0, 0, 0};
    for (int kk = 0; kk < Kp; kk += MMA_K) {
        clear_a(s);
        load_b(s, wp, sp, wsc, nsp, n0, N, kk);
        load_a_cp(s, xq, Kp / 2, xsc, Kp / BLOCK_K, m0, M, kk);
        cp_async_commit();
        cp_async_wait_all();
        __syncthreads();
        mma_tile(s->sA, 32, s->sAsc, s->sB, 32, s->sBsc,
                 warp * BN_WARP, d);
        __syncthreads();
    }
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int row = (i < 2) ? g : g + 8;
        int col = tig * 2 + (i & 1);
        int gm = m0 + row, gn = n0 + warp * BN_WARP + col;
        if (gm < M && gn < N)
            y[(long)gm * N + gn] = d[i];
    }
}

// =================================================================
// 3) grouped gate/up：grid (1, N/32, 2E)
//    z<E  → gate，写 gu[z][0:N]；z>=E → up，写 gu[z-E][N:2N]
// =================================================================
// 支持 M 个 token：grid.z = M*2E；每块 (token m, expert e, proj)，只用 A tile row0
__global__ void gateup_kernel(
        const uint8_t* __restrict__ xq, const uint8_t* __restrict__ xsc,
        const uint8_t* const* __restrict__ wp_arr,
        const uint8_t* const* __restrict__ wsc_arr,
        float* __restrict__ gu,                 // (M,E,2N)
        int E, int M, int N, int K, int Kp)
{
    extern __shared__ unsigned char smem_raw[];
    TileSmem* s = reinterpret_cast<TileSmem*>(smem_raw);
    const int warp = threadIdx.x >> 5;
    const int z = blockIdx.z;
    const int m  = z / (2 * E);
    const int pair = z % (2 * E);
    const int proj = pair / E;
    const int e = pair % E;
    const int n0 = blockIdx.y * BN;
    // 连续布局：表 = [gate_e0..eE, up_e0..eE, 下一token...]（token-major）
    const uint8_t* wp = wp_arr[m * 2 * E + proj * E + e];
    const uint8_t* wsc = wsc_arr[m * 2 * E + proj * E + e];
    const int sp = Kp / 2, nsp = Kp / BLOCK_K;

    float d[4] = {0, 0, 0, 0};
    for (int kk = 0; kk < Kp; kk += MMA_K) {
        clear_a(s);
        load_b(s, wp, sp, wsc, nsp, n0, N, kk);
        // row0 = token m（其余行 pad 0）
        load_a_cp(s, xq, Kp / 2, xsc, Kp / BLOCK_K, m, m + 1, kk);
        cp_async_commit();
        cp_async_wait_all();
        __syncthreads();
        mma_tile(s->sA, 32, s->sAsc, s->sB, 32, s->sBsc,
                 warp * BN_WARP, d);
        __syncthreads();
    }
    write_d(d, gu + ((long)m * E + e) * 2 * N + (long)proj * N, 2 * N, n0, N);
}

// =================================================================
// 4) fused silu + down：grid (1, N/32, E)；A=silu(gate)*up 就地量化
// =================================================================
// 支持 M 个 token：grid.z = M*E；每块 (token m, expert e)
__global__ void down_silu_kernel(
        const float* __restrict__ gu,          // (M,E,2K)
        const uint8_t* const* __restrict__ wp_arr,
        const uint8_t* const* __restrict__ wsc_arr,
        float* __restrict__ yd,                // (M,E,N)
        int E, int N, int K, int Kp)
{
    extern __shared__ unsigned char smem_raw[];
    TileSmem* s = reinterpret_cast<TileSmem*>(smem_raw);
    const int warp = threadIdx.x >> 5;
    const int tid = threadIdx.x;
    const int ze = blockIdx.z;                  // token*E + e
    const int e = ze % E;
    const int m = ze / E;
    const int n0 = blockIdx.y * BN;
    const uint8_t* wp = wp_arr[ze];
    const uint8_t* wsc = wsc_arr[ze];
    const int sp = Kp / 2, nsp = Kp / BLOCK_K;
    const float* grow = gu + ((long)m * E + e) * 2 * K;
    const float* urow = grow + K;

    float d[4] = {0, 0, 0, 0};
    for (int kk = 0; kk < Kp; kk += MMA_K) {
        clear_a(s);
        load_b(s, wp, sp, wsc, nsp, n0, N, kk);
        if (tid < 16)
            cp_async_16(&s->sAg[tid * 4], grow + kk + tid * 4);
        else if (tid < 32)
            cp_async_16(&s->sAu[(tid - 16) * 4], urow + kk + (tid - 16) * 4);
        cp_async_commit();
        cp_async_wait_all();
        __syncthreads();

        // warp0/1 各量化 row0 的一个 k32 block → sA[0] / sAsc[0]
        if (tid < 64) {
            int b = tid >> 5, t = tid & 31;
            int idx = b * 32 + t;
            float hh = 0.0f;
            if (kk + idx < K) {
                float gv = s->sAg[idx];
                hh = (gv / (1.0f + expf(-gv))) * s->sAu[idx];
            }
            float am = fabsf(hh);
            #pragma unroll
            for (int off = 16; off >= 1; off >>= 1)
                am = fmaxf(am, __shfl_xor_sync(0xFFFFFFFF, am, off));
            uint8_t code = e8m0_code(am);
            float sv = scale_value(code);
            uint32_t nib = quantize_nibble(hh, sv);
            uint32_t nib1 = __shfl_xor_sync(0xFFFFFFFF, nib, 1);
            if ((t & 1) == 0 && kk + idx + 1 <= K)
                s->sA[b * 16 + (t >> 1)] = (uint8_t)(nib | (nib1 << 4));
            if (t == 0)
                s->sAsc[b] = code;   // warp0→sAsc[0](kb0), warp1→sAsc[1](kb1)
        }
        __syncthreads();
        mma_tile(s->sA, 32, s->sAsc, s->sB, 32, s->sBsc,
                 warp * BN_WARP, d);
        __syncthreads();
    }
    write_d(d, yd + ((long)m * E + e) * N, N, n0, N);
}

// =================================================================
// 5) combine：out[c] = Σ_e w[e] * yd[e,c]；grid (N/256,)，block 256
// =================================================================
// out (M,N): out[m,c]=Σ_e w[m,e] * yd[(m,e),c]；grid (N/256, M)
__global__ void combine_kernel(const float* __restrict__ yd, int ystride,
                               const float* __restrict__ w, int E,
                               float* __restrict__ out, int M, int N)
{
    int m = blockIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= N) return;
    const float* yd_b = yd + (long)m * E * ystride;
    const float* w_b = w + (long)m * E;
    float acc = 0.0f;
    #pragma unroll
    for (int e = 0; e < 8 && e < E; ++e)
        acc += w_b[e] * yd_b[(long)e * ystride + c];
    out[(long)m * N + c] = acc;
}

// =================================================================
// launch（C-ABI）
// =================================================================
void launch_prequant_v2(const void* x, uint8_t* xq, uint8_t* xsc,
                        int M, int K, int Kp, cudaStream_t stream) {
    dim3 grid(M, Kp / BLOCK_K);
    prequant_kernel<<<grid, 32, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x), xq, xsc, K, Kp);
}

void launch_lin_pq_v2(const uint8_t* xq, const uint8_t* xsc,
                      const uint8_t* wp, const uint8_t* wsc, float* y,
                      int M, int N, int K, int Kp, cudaStream_t stream) {
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
    size_t smem = sizeof(TileSmem);
    lin_pq_kernel<<<grid, WARPS * 32, smem, stream>>>(
        xq, xsc, wp, wsc, y, M, N, K, Kp);
}

void launch_gateup_v2(const uint8_t* xq, const uint8_t* xsc,
                      const uint8_t* const* wp_arr,
                      const uint8_t* const* wsc_arr,
                      float* gu, int E, int M, int N, int K, int Kp,
                      cudaStream_t stream) {
    dim3 grid(1, (N + BN - 1) / BN, 2 * M * E);
    gateup_kernel<<<grid, WARPS * 32, sizeof(TileSmem), stream>>>(
        xq, xsc, wp_arr, wsc_arr, gu, E, M, N, K, Kp);
}

void launch_down_silu_v2(const float* gu,
                         const uint8_t* const* wp_arr,
                         const uint8_t* const* wsc_arr,
                         float* yd, int E, int M, int N, int K, int Kp,
                         cudaStream_t stream) {
    dim3 grid(1, (N + BN - 1) / BN, M * E);
    down_silu_kernel<<<grid, WARPS * 32, sizeof(TileSmem), stream>>>(
        gu, wp_arr, wsc_arr, yd, E, N, K, Kp);
}

void launch_combine_v2(const float* yd, int ystride, const float* w, int E,
                       float* out, int M, int N, cudaStream_t stream) {
    dim3 grid((N + 255) / 256, M);
    combine_kernel<<<grid, 256, 0, stream>>>(yd, ystride, w, E, out, M, N);
}

// =================================================================
// 6) decode attention（graph 友好）：M=1，读 device pos/S，就地 e4m3
//    kvk/kvv: (max_pos, Hkv, D) 每层独立 buffer（graph 内固定地址）
//    sc : (Hq, max_pos) f32 临时存储 scores（graph input 复用）
//    结果需与 torch SDPA(enable_gqa, scale=1/sqrt(D)) 对齐
// =================================================================
__device__ __forceinline__ float e4m3_to_f32(uint8_t b) {
    int sign = (b >> 7) & 1;
    int e = (b >> 3) & 0xF;
    int m = b & 7;
    float v;
    if (e == 0) v = (float)m * exp2f(-6.0f);          // subnormal
    else if (e == 15) v = 448.0f * ((float)m * 0.125f + 1.0f); // 防御，不应出现
    else v = ((float)m * 0.125f + 1.0f) * exp2f((float)(e - 7));
    return sign ? -v : v;
}

// ---- Stage A：score[h][s] = q[h] . kv_k[g][s]（grid.x=Hq, block=32==1 warp）----
__global__ void attn_score_kernel(
        const __nv_bfloat16* __restrict__ q,   // (M,Hq,D)
        const uint8_t* __restrict__ kvk,       // (max_pos,Hkv,D)
        float* __restrict__ sc,                // (M,Hq,max_pos)
        const int* __restrict__ dpos, int max_pos, int Hkv, int D, int Hgroup)
{
    int h = blockIdx.x, m = blockIdx.y, lane = threadIdx.x;
    int g = h / Hgroup;
    int base = *dpos;                          // host 已写 device base
    int pos = base + m;
    int S = pos + 1;
    if (pos < 0 || pos >= max_pos) return;
    const __nv_bfloat16* qrow = q + ((long)m * gridDim.x + h) * D;
    // 每个 lane 负责 c = lane, lane+32, lane+64, lane+96（D=128）
    float qv[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) { int c = lane + j * 32; qv[j] = __bfloat162float(qrow[c]); }
    for (int s = 0; s < S; ++s) {
        const uint8_t* krow = kvk + ((long)s * Hkv + g) * D;
        float dotv = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) dotv += qv[j] * e4m3_to_f32(krow[lane + j * 32]);
        #pragma unroll
        for (int off = 16; off >= 1; off >>= 1) dotv += __shfl_xor_sync(0xFFFFFFFF, dotv, off);
        if (lane == 0) sc[((long)m * gridDim.x + h) * max_pos + s] = dotv;
    }
}

// ---- Stage B：online softmax + 加权 v（grid.x=Hq, block=D(128)）----
__global__ void attn_combine_kernel(
        const float* __restrict__ sc,          // (M,Hq,max_pos)
        const uint8_t* __restrict__ kvv,       // (max_pos,Hkv,D)
        float* __restrict__ o,                 // (M,Hq,D)
        const int* __restrict__ dpos, int max_pos, int Hkv, int D, int Hgroup,
        float invsqrt)
{
    int h = blockIdx.x, m = blockIdx.y, c = threadIdx.x;
    int g = h / Hgroup;
    int base = *dpos;
    int pos = base + m;
    int S = pos + 1;
    if (pos < 0 || pos >= max_pos || c >= D) return;
    const float* srow = sc + ((long)m * gridDim.x + h) * max_pos;
    float mm = -1e30f, l = 0.0f, acc = 0.0f;
    for (int s = 0; s < S; ++s) {
        float x = srow[s] * invsqrt;
        float mnew = fmaxf(mm, x);
        float w = expf(x - mnew);
        float lnew = l * expf(mm - mnew) + w;
        // 读 v[c] @ (s,g)
        float vc = e4m3_to_f32(kvv[((long)s * Hkv + g) * D + c]);
        acc = acc * expf(mm - mnew) + w * vc;
        mm = mnew; l = lnew;
    }
    o[((long)m * gridDim.x + h) * D + c] = acc / l;
}

// C-ABI
void launch_attn_decode_v2(const void* q, const uint8_t* kvk, const uint8_t* kvv,
                           float* sc, float* o, const int* dpos,
                           int Hq, int Hkv, int D, int max_pos,
                           int M, cudaStream_t stream) {
    const int Hgroup = Hq / Hkv;
    const float invsqrt = 1.0f / sqrtf((float)D);
    dim3 g(Hq, M);
    attn_score_kernel<<<g, 32, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q), kvk, sc, dpos,
        max_pos, Hkv, D, Hgroup);
    attn_combine_kernel<<<g, D, 0, stream>>>(
        sc, kvv, o, dpos, max_pos, Hkv, D, Hgroup, invsqrt);
}

// =================================================================
// 7) graph 友好 KV 写入：读 device dpos，bf16 → fp8(e4m3)，写入定点 buffer
//    kvk/kvv: (max_pos, Hkv, D) 每层 slice；k/v: (Hkv,D) bf16
// =================================================================
__device__ __forceinline__ uint8_t f32_to_e4m3(float v) {
    uint8_t sign = (v < 0.0f) ? 0x80 : 0;
    float a = fabsf(v);
    if (a == 0.0f) return sign;
    int e; float m = frexpf(a, &e);            // a = m*2^e, m in [0.5,1)
    // a = (1+q/8) * 2^(eb-7)，取 base=2^(e-1) => eb=e-1+7=e+6
    int eb = e + 6;
    float man = a / ldexpf(1.0f, e - 1);       // man in [1,2)
    int q = (int)lroundf((man - 1.0f) * 8.0f);
    if (q == 8) { q = 0; eb += 1; }
    if (eb <= 0) {
        // subnormal：step=2^-6，mantissa=round(a*2^6)/8*k
        int s = (int)lroundf(a * 64.0f);
        if (s > 7) s = 7; if (s < 0) s = 0;
        return (uint8_t)(sign | s);
    }
    if (eb >= 15) { eb = 15; q = 7; }
    return (uint8_t)(sign | ((eb & 0xF) << 3) | (q & 7));
}

// k/v: (M,Hkv,D) bf16；dpos=base；写位置 base+m (m=0..M-1)
__global__ void kvwrite_kernel(const __nv_bfloat16* __restrict__ k,
                               const __nv_bfloat16* __restrict__ v,
                               uint8_t* __restrict__ kvk,
                               uint8_t* __restrict__ kvv,
                               const int* __restrict__ dpos,
                               int M, int Hkv, int D, int max_pos)
{
    long tot = (long)Hkv * D;
    long idx = blockIdx.x * blockDim.x + threadIdx.x;   // 0..M*tot
    int base = *dpos;
    if (base < 0 || idx >= (long)M * tot) return;
    int m = (int)(idx / tot);
    int t = (int)(idx % tot);
    int pos = base + m;
    if (pos >= max_pos) return;
    long off = (long)pos * tot + t;
    kvk[off] = f32_to_e4m3(__bfloat162float(k[m * tot + t]));
    kvv[off] = f32_to_e4m3(__bfloat162float(v[m * tot + t]));
}

void launch_kvwrite_v2(const void* k, const void* v, uint8_t* kvk, uint8_t* kvv,
                       const int* dpos, int M, int Hkv, int D, int max_pos,
                       cudaStream_t stream) {
    long tot = (long)Hkv * D;
    long n = (long)M * tot;
    kvwrite_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(k),
        reinterpret_cast<const __nv_bfloat16*>(v), kvk, kvv, dpos,
        M, Hkv, D, max_pos);
}

}  // namespace mxfp4v2
