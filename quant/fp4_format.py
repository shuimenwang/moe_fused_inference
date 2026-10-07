#!/usr/bin/env python3
# fp4_format.py - FP4（MXFP4/NVFP4）格式原语库
#
# 内容：
#   E2M1  元素格式：8 个对称幅值 0/.5/1/1.5/2/3/4/6，nibble bit3=符号
#   E8M0  MXFP4 block scale：仅 2 的幂，bias=127
#   E4M3  NVFP4 block scale：标准 FP8 E4M3（bias=7，max=448）
#   nibble pack/unpack：LSB-first（偶数下标在低 4 位，与 PTX 寄存器布局一致）
#
# 设计原则：纯 numpy、无 GPU 依赖、向量化；所有函数可被 streaming 量化器复用。
# 运行 `python fp4_format.py` 执行自测（全部 assert 通过即格式实现正确）。

import math

import numpy as np

# ------------------------------------------------------------------
# E2M1：2-bit 指数 bias=1 + 1-bit 尾数；全部编码均为有限值（无 NaN/Inf）
# nibble: [sign][mag2][mag1][mag0]
# ------------------------------------------------------------------
E2M1_POSITIVE = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def quantize_e2m1(x):
    """标量/数组 → 最近 E2M1 nibble codes（uint8，0..15）。

    - ties-up：等距时取较大幅值（如 0.25 -> 0.5）
    - 超 6 截断为 6（正常流程由 block scale 保证不溢出，此为兜底）
    """
    a = np.abs(np.asarray(x, dtype=np.float64))
    flat = a.reshape(-1)

    # searchsorted 定位相邻表项
    j = np.searchsorted(E2M1_POSITIVE, flat, side="left")
    j_clip = np.minimum(j, len(E2M1_POSITIVE) - 1)
    upper = E2M1_POSITIVE[j_clip]
    lower = E2M1_POSITIVE[np.maximum(j - 1, 0)]

    pick_upper = (flat - lower) >= (upper - flat)   # tie -> upper
    mag = np.where(pick_upper, upper, lower)

    # searchsorted 对超出表范围的值返回 8：lower=6 upper=6，结果已 clamp 到 6
    idx = np.searchsorted(E2M1_POSITIVE, mag.astype(np.float32)).astype(np.uint8)
    sign = (np.asarray(x, dtype=np.float64).reshape(-1) < 0).astype(np.uint8)
    code = ((sign << 3) | idx).reshape(np.shape(x))
    return code.astype(np.uint8)


def decode_e2m1(code):
    """nibble codes → float32 值。"""
    c = np.asarray(code, dtype=np.uint8) & 0xF
    sign = np.where((c >> 3) == 0, 1.0, -1.0)
    return (sign * E2M1_POSITIVE[c & 0x7]).astype(np.float32)


# ------------------------------------------------------------------
# E8M0（MXFP4 block scale）：8-bit 指数，bias=127，值 = 2^(code-127)
# ------------------------------------------------------------------
E8M0_BIAS = 127


def e8m0_encode_value(value):
    """2 的幂值 → E8M0 code（非 2 的幂输入按最近指数取）。"""
    e = int(math.floor(math.log2(abs(value)) + 0.5))
    e = max(-E8M0_BIAS, min(E8M0_BIAS, e))
    return e + E8M0_BIAS


def e8m0_decode(code):
    return np.power(2.0, np.asarray(code, dtype=np.int64) - E8M0_BIAS).astype(np.float32)


def mxfp4_block_scale(amax, block_max=6.0):
    """块内最大幅值 → (scale 2的幂 float32, e8m0 code)。

    选最小的 2^e 使 amax/scale <= block_max（向上保守，保证无 clipping）。
    """
    if amax <= 0.0:
        return np.float32(1.0), E8M0_BIAS
    target = amax / block_max
    m, e = math.frexp(target)          # target = m * 2^e, m ∈ [0.5, 1)
    # log2(target) = e + log2(m)；m=0.5 → e-1；m>0.5 → log2(m)∈(-1,0)，ceil=e
    if m == 0.5:
        e_pow = e - 1                   # 恰好 2 的幂
    else:
        e_pow = e                       # ceil(log2(target))
    e_pow = max(-E8M0_BIAS, min(E8M0_BIAS, e_pow))
    return np.float32(2.0 ** e_pow), e_pow + E8M0_BIAS


# ------------------------------------------------------------------
# E4M3（NVFP4 block scale）：1 符号 / 4 指数 bias=7 / 3 尾数
#   仅 0x7F (exp=15, frac=7) 为 NaN；最大有限值 448
# ------------------------------------------------------------------
def _build_e4m3_table():
    vals = [0.0]
    for exp in range(0, 16):
        frac_max = 7 if exp < 15 else 6   # exp=15,frac=7 是 NaN
        for frac in range(0, frac_max + 1):
            if exp == 0:
                if frac == 0:
                    continue
                v = (frac / 8.0) * 2 ** -6          # subnormal
            else:
                v = (1.0 + frac / 8.0) * 2 ** (exp - 7)
            vals.append(v)
    return np.array(sorted(vals), dtype=np.float32)


E4M3_POSITIVE = _build_e4m3_table()


def nvfp4_block_scale(amax, block_max=6.0):
    """块内最大幅值 → 最小的 E4M3 scale 使 amax/scale <= 6；返回 float32。"""
    if amax <= 0.0:
        return np.float32(E4M3_POSITIVE[1])
    target = amax / block_max
    i = int(np.searchsorted(E4M3_POSITIVE, target, side="left"))
    i = min(max(i, 1), len(E4M3_POSITIVE) - 1)
    return np.float32(E4M3_POSITIVE[i])


# ------------------------------------------------------------------
# nibble 打包（LSB-first）
# ------------------------------------------------------------------
def pack_nibbles(codes):
    """nibble codes → packed uint8 bytes；返回 (bytes, pad_count)。

    byte = codes[2k]（低 4 位） | codes[2k+1]<<4（高 4 位）
    奇数长度末尾补 0。
    """
    c = np.asarray(codes, dtype=np.uint8).reshape(-1)
    pad = int((-len(c)) % 2)
    if pad:
        c = np.concatenate([c, np.zeros(pad, dtype=np.uint8)])
    packed = (c[0::2] | (c[1::2] << 4)).astype(np.uint8)
    return packed, pad


def unpack_nibbles(packed, n=None):
    """packed bytes → nibble codes；n 指定有效长度（去掉 padding）。"""
    p = np.asarray(packed, dtype=np.uint8).reshape(-1)
    out = np.empty(p.size * 2, dtype=np.uint8)
    out[0::2] = p & 0xF
    out[1::2] = (p >> 4) & 0xF
    if n is not None:
        out = out[:n]
    return out


# ------------------------------------------------------------------
# 自测
# ------------------------------------------------------------------
def _selftest():
    # 1) 全部 16 个 nibble：decode → quantize 必须位级还原
    #    例外：code 8 = -0.0，浮点语义下 -0.0 < 0 为 False，归一为 +0（数值相等）
    all_codes = np.arange(16, dtype=np.uint8)
    vals = decode_e2m1(all_codes)
    rec = quantize_e2m1(vals)
    expected = all_codes.copy()
    expected[8] = 0
    assert np.array_equal(rec, expected), f"E2M1 roundtrip 失败: {rec} vs {expected}"
    assert float(decode_e2m1(np.uint8(8))) == 0.0

    # 2) 已知点
    assert quantize_e2m1(1.7) == quantize_e2m1(1.5)   # .2 < .3 → 1.5
    assert decode_e2m1(quantize_e2m1(1.7)) == 1.5
    assert decode_e2m1(quantize_e2m1(0.25)) == 0.5    # tie-up
    assert decode_e2m1(quantize_e2m1(7.0)) == 6.0     # 超范围 clamp
    assert decode_e2m1(quantize_e2m1(-3.0)) == -3.0
    assert decode_e2m1(0xF) == -6.0 and decode_e2m1(0x1) == 0.5

    # 3) E8M0 block scale
    cases = [(5.0, 1.0), (6.0, 1.0), (6.1, 2.0), (12.0, 2.0), (0.5, 0.125)]
    for amax, expect in cases:
        s, code = mxfp4_block_scale(amax)
        assert float(s) == expect and float(e8m0_decode(code)) == expect, (amax, s, expect)
        assert amax / float(s) <= 6.0 + 1e-6

    # 4) E4M3 表：127 项（含 0）、严格递增、max=448
    assert len(E4M3_POSITIVE) == 127 and E4M3_POSITIVE[-1] == 448.0
    assert np.all(np.diff(E4M3_POSITIVE) > 0)
    s448 = nvfp4_block_scale(448.0 * 6.0)
    assert float(s448) == 448.0 and nvfp4_block_scale(0.0) > 0

    # 5) pack/unpack（含奇数长度 pad）
    codes = np.array([0x0, 0x7, 0xF, 0x1, 0x3], dtype=np.uint8)
    packed, pad = pack_nibbles(codes)
    assert pad == 1 and packed[0] == (0x0 | 0x7 << 4) and packed[1] == (0xF | 0x1 << 4)
    back = unpack_nibbles(packed, n=5)
    assert np.array_equal(back, codes)

    # 6) 向量化量化 + 打包完整链路 sanity
    x = np.linspace(-6, 6, 64, dtype=np.float32)
    q = quantize_e2m1(x)
    p2, _ = pack_nibbles(q)
    q2 = unpack_nibbles(p2, n=64)
    assert np.array_equal(q, q2)

    print("=" * 56)
    print(" fp4_format.py 自测全部通过")
    print("=" * 56)
    print(f" E2M1 幅值表:      {E2M1_POSITIVE.tolist()}")
    print(f" E4M3 正值数量:    {len(E4M3_POSITIVE)}（含0），max={E4M3_POSITIVE[-1]}")
    print(" 验证项: 16 nibble roundtrip / ties-up / clamp /")
    print("         E8M0 5 点 / E4M3 表 / pack LSB-first / 64 元素链路")


if __name__ == "__main__":
    _selftest()
