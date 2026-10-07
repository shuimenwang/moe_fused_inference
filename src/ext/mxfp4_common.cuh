// mxfp4_common.cuh - MXFP4 block-scale MMA 公共设备原语（P2.2 decode kernel）
//
// 内容：E2M1/E8M0 量化、block-scale m16n8k64 PTX 内联汇编、fragment 装配辅助。
// fragment 映射与 mxfp4_linear.cu（P0 数值单测实证）完全一致：
//   A(4×b32) row: i∈[0,8)∪[16,24)→g ; i∈[8,16)∪[24,32)→g+8
//               col: i<16→tig*8+(i&7) ; i>=16→32+tig*8+(i&7)
//   B(2×b32) row(K): i<8→tig*8+(i&7) ; i>=8→32+tig*8+(i&7) ; col(N)=g
//   D(4×f32) row: i<2→g ; i>=2→g+8 ; col=tig*2+(i&1)
// scale_vec::2X：sA 由 tig==0/1 提供 row g/g+8；sB 由 tig==0 提供 col g。

#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace mxfp4v2 {

constexpr int BLOCK_K = 32;
constexpr int MMA_K = 64;
constexpr int BM = 16;
constexpr int BN_WARP = 8;
constexpr int WARPS = 4;
constexpr int BN = 32;

__device__ __forceinline__ uint32_t e2m1_mag_code(float a) {
    if (a < 0.25f) return 0;
    if (a < 0.75f) return 1;
    if (a < 1.25f) return 2;
    if (a < 1.75f) return 3;
    if (a < 2.50f) return 4;
    if (a < 3.50f) return 5;
    if (a < 5.00f) return 6;
    return 7;
}

__device__ __forceinline__ uint32_t quantize_nibble(float v, float scale) {
    float x = v / scale;
    uint32_t code = e2m1_mag_code(fabsf(x));
    if (x < 0.0f) code |= 0x8;
    return code;
}

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

// sA: [16][32B]（k64 chunk 的 packed nibble）；sAsc: [16][2]（两块 E8M0）
// sB: [32][32B]；sBsc: [32][2]
// 每 warp 对应 N 行 n_w0..n_w0+7；m_tile 恒 16 行。
__device__ __forceinline__ void mma_tile(
        const uint8_t* __restrict__ sA, int sA_stride,
        const uint8_t* __restrict__ sAsc,
        const uint8_t* __restrict__ sB, int sB_stride,
        const uint8_t* __restrict__ sBsc, int n_w0,
        float d[4])
{
    const int lane = threadIdx.x & 31;
    const int g  = lane >> 2;
    const int tig = lane & 3;

    uint32_t a[4] = {0, 0, 0, 0};
    for (int i = 0; i < 32; ++i) {
        int reg = i >> 3, pos = i & 7;
        int row = (i < 8 || (i >= 16 && i < 24)) ? g : g + 8;
        int kc = (i < 16) ? tig * 8 + (i & 7) : 32 + tig * 8 + (i & 7);
        uint8_t by = sA[row * sA_stride + (kc >> 1)];
        uint32_t nib = (kc & 1) ? (by >> 4) : (by & 0xF);
        a[reg] |= nib << (4 * pos);
    }
    uint32_t b[2] = {0, 0};
    for (int i = 0; i < 16; ++i) {
        int reg = i >> 3, pos = i & 7;
        int kc = (i < 8) ? tig * 8 + (i & 7) : 32 + tig * 8 + (i & 7);
        uint8_t by = sB[(n_w0 + g) * sB_stride + (kc >> 1)];
        uint32_t nib = (kc & 1) ? (by >> 4) : (by & 0xF);
        b[reg] |= nib << (4 * pos);
    }
    uint32_t sfa = 0, sfb = 0;
    if (tig == 0)
        sfa = (uint32_t)sAsc[g * 2] | ((uint32_t)sAsc[g * 2 + 1] << 8);
    else if (tig == 1)
        sfa = (uint32_t)sAsc[(g + 8) * 2]
              | ((uint32_t)sAsc[(g + 8) * 2 + 1] << 8);
    if (tig == 0)
        sfb = (uint32_t)sBsc[(n_w0 + g) * 2]
              | ((uint32_t)sBsc[(n_w0 + g) * 2 + 1] << 8);

    mma_mxfp4(d, a, b, sfa, sfb);
}

// 16 字节 cp.async（global→shared）
__device__ __forceinline__ void cp_async_16(void* smem_dst, const void* gmem_src) {
    uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                 :: "r"(sd), "l"(gmem_src));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::);
}

}  // namespace mxfp4v2
