// mxfp4_linear.cu - P0 极简正确性 MXFP4 block-scale MMA 算子（sm_120a）
//
// 计算 y = x @ W^T（F.linear）：
//   - 权重 W 预量化为 MXFP4：packed E2M1 nibble（LSB-first 沿 K）+ E8M0 block scale code
//   - 激活 x 在 kernel 内“在线量化”：逐 1×32 block，E8M0 scale（2的幂）+ E2M1 nibble
//   - 单指令 m16n8k64，FP32 累加（D = (A*sA)*(B*sB) + C）
//
// 正确性优先：简单静态 tiling（每 warp 一个 16×8 输出 tile，直接 GMEM→reg），
// 不用 persistent/SMEM 流水线/ldmatrix——这些留给 P2 K2。
//
// Fragment 布局（PTX ISA 8.7 §9.7.14.5.11，权威公式）：
//   groupID=lane>>2, tig=lane%4
//   A(4×b32, i=0..31): reg=i/8
//       row: i∈[0,8)∪[16,24)→g ; i∈[8,16)∪[24,32)→g+8
//       col: i<16→tig*8+(i&7) ; i>=16→32+tig*8+(i&7)
//   B(2×b32, i=0..15): reg=i/8
//       row: i<8→tig*8+(i&7) ; i>=8→32+tig*8+(i&7) ; col=g
//   D(4×f32, i=0..3): row: i<2→g ; i>=2→g+8 ; col=tig*2+(i&1)
//
// Scale selector（scale_vec::2X，恒取 byte-id=0/thread-id=0）：
//   sA: quad 内 lane%4∈{0,1} 提供 bytes[0,1]；lane%4==0→row g，==1→row g+8
//   sB: lane%4==0 提供 bytes[0,1]（col g）
//   （上述字节到 scale 行的映射由数值单测实证）

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

namespace mxfp4 {

constexpr int BLOCK_K = 32;   // MXFP4 block elements
constexpr int MMA_K = 64;     // K per mma instruction
constexpr int BM = 16;        // M per warp
constexpr int BN_WARP = 8;    // N per warp
constexpr int WARPS_PER_BLOCK = 4;
constexpr int BN = BN_WARP * WARPS_PER_BLOCK;  // 32 N per block

// ----------------------------------------------------------------
// E2M1 量化：|v|∈[0,6]（已由 block scale 归一）→ magnitude code 0..7
// 最近值，ties-up（等距取较大幅值）
// ----------------------------------------------------------------
__device__ __forceinline__ uint32_t e2m1_mag_code(float a) {
    if (a < 0.25f) return 0;          // 0
    if (a < 0.75f) return 1;          // 0.5   (0.25 tie → 0.5)
    if (a < 1.25f) return 2;          // 1.0
    if (a < 1.75f) return 3;          // 1.5
    if (a < 2.50f) return 4;          // 2.0
    if (a < 3.50f) return 5;          // 3.0
    if (a < 5.00f) return 6;          // 4.0
    return 7;                         // 6.0
}

__device__ __forceinline__ uint32_t quantize_nibble(float v, float scale) {
    float x = v / scale;
    uint32_t code = e2m1_mag_code(fabsf(x));
    if (x < 0.0f) code |= 0x8;
    return code;
}

// 块内最大幅值 → E8M0 code（最小 2^e 使 amax/2^e <= 6）
__device__ __forceinline__ uint8_t e8m0_code(float amax) {
    if (amax <= 0.0f) return 127;
    int eexp;
    float m = frexpf(amax / 6.0f, &eexp);
    int e = (m == 0.5f) ? eexp - 1 : eexp;
    if (e < -127) e = -127;
    if (e > 127) e = 127;
    return (uint8_t)(e + 127);
}

__device__ __forceinline__ float scale_value(uint8_t code) {
    return exp2f((float)code - 127.0f);
}

// ----------------------------------------------------------------
// 单条 block-scale MMA
// ----------------------------------------------------------------
__device__ __forceinline__ void mma_mxfp4(float d[4],
                                          uint32_t a[4], uint32_t b[2],
                                          uint32_t sfa, uint32_t sfb) {
    uint16_t bida = 0, tida = 0, bidb = 0, tidb = 0;
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X."
        "f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
        "{%10}, {%11,%12}, {%13}, {%14,%15};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "r"(sfa), "h"(bida), "h"(tida),
          "r"(sfb), "h"(bidb), "h"(tidb));
}

// 读取权重 nibble：wp 每行 Kp/2 packed bytes，LSB-first
__device__ __forceinline__ uint32_t weight_nibble(const uint8_t* __restrict__ wp,
                                                  int row, int k, int Kp) {
    int byte_idx = row * (Kp / 2) + k / 2;
    uint8_t b = wp[byte_idx];
    return (k & 1) ? (b >> 4) : (b & 0xF);
}

// ----------------------------------------------------------------
// Kernel：grid(ceil(M/16), ceil(N/32))；每 block 4 warp × (16×8)
// ----------------------------------------------------------------
__global__ void mxfp4_linear_kernel(
        const __half* __restrict__ x,      // (M,K) fp16/bf16 位模式按 half 读
        const uint8_t* __restrict__ wp,   // (N,Kp) packed
        const uint8_t* __restrict__ wsc,  // (N,Kp/32) E8M0 codes
        float* __restrict__ y,            // (M,N) f32
        int M, int N, int K, int Kp)
{
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int g  = lane >> 2;     // groupID 0..7
    const int tig = lane & 3;

    const int m_base = blockIdx.x * BM;
    const int n_base = blockIdx.y * BN + warp * BN_WARP;

    __shared__ float s_x[BM][MMA_K];
    __shared__ float s_av[BM][2];    // activation scale value
    __shared__ uint8_t s_ac[BM][2]; // activation scale code

    float d[4] = {0,0,0,0};

    for (int kk = 0; kk < Kp; kk += MMA_K) {
        // ---- 加载激活 tile 到 smem（f32），越界/越 K 填 0 ----
        for (int idx = threadIdx.x; idx < BM * MMA_K; idx += blockDim.x) {
            int r = idx / MMA_K, c = idx % MMA_K;
            int gm = m_base + r, gk = kk + c;
            float v = (gm < M && gk < K) ? __half2float(x[gm * K + gk]) : 0.0f;
            s_x[r][c] = v;
        }
        // ---- 每个 (row, chunk) combo 由前 32 线程之一求 amax/scale ----
        if (threadIdx.x < BM * 2) {
            int r = threadIdx.x >> 1, ch = threadIdx.x & 1;
            float amax = 0.0f;
            for (int c = 0; c < BLOCK_K; ++c)
                amax = fmaxf(amax, fabsf(s_x[r][ch * BLOCK_K + c]));
            uint8_t code = e8m0_code(amax);
            s_ac[r][ch] = code;
            s_av[r][ch] = scale_value(code);
        }
        __syncthreads();

        // ---- A fragment：每线程 32 nibble（权威公式 row: i∈[0,8)∪[16,24)→g）----
        uint32_t a[4] = {0,0,0,0};
        for (int i = 0; i < 32; ++i) {
            int reg = i >> 3, pos = i & 7;
            int row = (i < 8 || (i >= 16 && i < 24)) ? g : g + 8;
            int kc = (i < 16) ? tig * 8 + (i & 7) : 32 + tig * 8 + (i & 7);
            uint32_t nib = quantize_nibble(s_x[row][kc], s_av[row][kc / BLOCK_K]);
            a[reg] |= nib << (4 * pos);
        }

        // ---- B fragment：每线程 16 nibble（col n = g）----
        uint32_t b[2] = {0,0};
        for (int i = 0; i < 16; ++i) {
            int reg = i >> 3, pos = i & 7;
            int kc = (i < 8) ? tig * 8 + (i & 7) : 32 + tig * 8 + (i & 7);
            int n = n_base + g;
            uint32_t nib = (n < N) ? weight_nibble(wp, n, kk + kc, Kp) : 0;
            b[reg] |= nib << (4 * pos);
        }

        // ---- scale registers ----
        uint32_t sfa = 0, sfb = 0;
        // 激活 scale A：lane%4==0→row g，lane%4==1→row g+8
        if (tig == 0) {
            sfa = (uint32_t)s_ac[g][0] | ((uint32_t)s_ac[g][1] << 8);
        } else if (tig == 1) {
            sfa = (uint32_t)s_ac[g + 8][0] | ((uint32_t)s_ac[g + 8][1] << 8);
        }
        // 权重 scale B：lane%4==0，col n=g，bytes0/1 = kb 0/1
        if (tig == 0) {
            int n = n_base + g;
            if (n < N) {
                int kb0 = (kk) / BLOCK_K, kb1 = (kk + 32) / BLOCK_K;
                uint8_t c0 = wsc[n * (Kp / BLOCK_K) + kb0];
                uint8_t c1 = wsc[n * (Kp / BLOCK_K) + kb1];
                sfb = (uint32_t)c0 | ((uint32_t)c1 << 8);
            }
        }

        mma_mxfp4(d, a, b, sfa, sfb);
        __syncthreads();
    }

    // ---- epilogue：D fragment 写出 ----
    for (int i = 0; i < 4; ++i) {
        int row = (i < 2) ? g : g + 8;
        int col = tig * 2 + (i & 1);
        int gm = m_base + row, gn = n_base + col;
        if (gm < M && gn < N)
            y[gm * N + gn] = d[i];
    }
}

void launch_mxfp4_linear(const __half* x, const uint8_t* wp, const uint8_t* wsc,
                         float* y, int M, int N, int K, int Kp,
                         cudaStream_t stream) {
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
    dim3 block(WARPS_PER_BLOCK * 32);
    mxfp4_linear_kernel<<<grid, block, 0, stream>>>(x, wp, wsc, y, M, N, K, Kp);
}

}  // namespace mxfp4
