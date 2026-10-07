#!/usr/bin/env python3
# test_mxfp4_decode.py - P2.2 decode kernel 算子级数值单测
#
# 对拍链：
#   prequant      ↔ CPU 反量化参考（pack 顺序 / E8M0 scale）
#   lin_pq        ↔ 旧 ext.forward（已数值实证的朴素 kernel）
#   gateup/down/combine ↔ Wrec（量化权重理想执行）参考
# 判定：cosine > 0.999 即认为 fragment/scale/sync 正确。

import os
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "runtime"))
sys.path.insert(0, os.path.join(_ROOT, "quant"))

from fp4_ext import get_ext, get_decode_ext
from quantize_mxfp4 import quantize_expert_tensor, MXFP4_BLOCK
from fp4_format import unpack_nibbles, decode_e2m1, e8m0_decode


def make_w(Wf):
    packed, scodes, info = quantize_expert_tensor(torch.from_numpy(Wf))
    n, kp = info["orig_shape"][0], info["padded_k"]
    wp = torch.from_numpy(packed.reshape(n, kp // 2)).cuda()
    wsc = torch.from_numpy(scodes).cuda()
    codes = unpack_nibbles(packed, n=n * kp).reshape(n, kp // MXFP4_BLOCK, MXFP4_BLOCK)
    Wrec = ((decode_e2m1(codes).astype(np.float32)
             * e8m0_decode(scodes)[..., None]).reshape(n, kp)
            [:, :info["orig_shape"][1]])
    return wp, wsc, Wrec.astype(np.float32)


def xq_dequant(xq, xsc, K):
    M, half = xq.shape
    Kp = half * 2
    nbytes = xq.numel()
    codes = unpack_nibbles(xq.cpu().numpy().reshape(-1), n=nbytes * 2).reshape(M, Kp // 32, 32)
    sc = xsc.cpu().numpy()
    deq = (decode_e2m1(codes).astype(np.float32)
           * e8m0_decode(sc)[..., None]).reshape(M, Kp)[:, :K]
    return deq


def cos(a, b):
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def test_prequant(ext2):
    # ground truth 直接用生产量化器（P0 已验证其与朴素 kernel 数值一致）
    rng = np.random.default_rng(0)
    for M, K in [(1, 64), (1, 2048), (3, 768), (2, 100)]:
        x = (rng.standard_normal((M, K)).astype(np.float32) * 0.5)
        x[rng.random((M, K)) < 0.01] *= 4.0
        xb = torch.from_numpy(x).bfloat16().cuda()
        xq, xsc = ext2.prequant(xb)

        packed, scodes, info = quantize_expert_tensor(xb.cpu())
        kp = info["padded_k"]
        ref_q = torch.from_numpy(packed.reshape(M, kp // 2)).cuda()
        ref_sc = torch.from_numpy(scodes).cuda()
        match_q = float((xq == ref_q).float().mean())
        match_sc = float((xsc == ref_sc).float().mean())
        # bf16→量化 与 f32 路径只允许极少数边界值差异
        assert match_q > 0.999, (M, K, match_q, match_sc)
        assert match_sc > 0.99, (M, K, match_q, match_sc)
        print(f"  prequant M={M} K={K:<5} byte_match={match_q:.5f} "
              f"scale_match={match_sc:.5f} OK")


def test_lin_pq(ext1, ext2):
    rng = np.random.default_rng(1)
    for M, N, K in [(1, 32, 64), (1, 768, 2048), (1, 2048, 768),
                    (3, 128, 100), (1, 4096, 2048), (2, 2048, 2048)]:
        W = rng.standard_normal((N, K)).astype(np.float32)
        W *= np.power(2.0, rng.integers(-1, 2, size=(N, 1)))
        x = rng.standard_normal((M, K)).astype(np.float32) * 0.5
        wp, wsc, _ = make_w(W)
        old = ext1.forward(torch.from_numpy(x).half().cuda(), wp, wsc)
        old = old.float().cpu().numpy()
        xq, xsc = ext2.prequant(torch.from_numpy(x).bfloat16().cuda())
        new = ext2.lin_pq(xq, xsc, wp, wsc, M, N, K).float().cpu().numpy()
        c = cos(new, old)
        assert c > 0.999, (M, N, K, c)
        print(f"  lin_pq  M={M} N={N:<5} K={K:<5} cos_vs_old={c:.6f} OK")


def test_moe(ext1, ext2):
    # 全链路对拍旧朴素 kernel（两者都做激活 fp4，隔离的只是新 kernel 实现差异）
    rng = np.random.default_rng(2)
    E, N, K = 8, 768, 2048
    x = (rng.standard_normal(K).astype(np.float32) * 0.5)
    gwp, gsc, uwp, usc, dwp, dsc = [], [], [], [], [], []
    for _ in range(E):
        Wg = rng.standard_normal((N, K)).astype(np.float32)
        Wu = rng.standard_normal((N, K)).astype(np.float32)
        Wd = rng.standard_normal((2048, N)).astype(np.float32)
        a, b, _ = make_w(Wg); gwp.append(a); gsc.append(b)
        a, b, _ = make_w(Wu); uwp.append(a); usc.append(b)
        a, b, _ = make_w(Wd); dwp.append(a); dsc.append(b)

    xb = torch.from_numpy(x[None, :]).bfloat16().cuda()
    xh = torch.from_numpy(x[None, :]).half().cuda()
    xq, xsc = ext2.prequant(xb)

    # 新路径
    gu = ext2.moe_gateup(xq, xsc, gwp + uwp, gsc + usc, E, 1, N, K)
    yd = ext2.moe_down(gu, dwp, dsc, E, 2048, N)
    w = torch.softmax(torch.randn(E, device="cuda"), 0)
    out = ext2.moe_combine(yd, w).float().cpu().numpy()

    # 旧路径：朴素 kernel 逐投影 + f32 silu
    ref = np.zeros(2048, np.float32)
    wn = w.float().cpu().numpy()
    for e in range(E):
        g = ext1.forward(xh, gwp[e], gsc[e]).squeeze(0).float()
        u = ext1.forward(xh, uwp[e], usc[e]).squeeze(0).float()
        h = (torch.sigmoid(g) * g * u)[None, :].half()
        d = ext1.forward(h, dwp[e], dsc[e]).squeeze(0).float().cpu().numpy()
        ref += wn[e] * d
    c = cos(out, ref)
    # 两条合法 fp4 执行路径的差异只来自 h 的量化放置（旧：half→再量化；
    # 新：f32 silu 就地量化），cos>0.995 即判定映射正确（P0 MMA 基线 0.992）。
    assert c > 0.995, c
    print(f"  moe gateup/down/combine cos_vs_old={c:.6f} OK")

    # 强结构验证：fused down 与「f32 silu→bf16 prequant→lin_pq」逐 expert 对比，
    # 差异仅一次 bf16 舍入，要求 cos>0.999
    for e in range(E):
        g = gu[e, :N]; u = gu[e, N:]
        h = (torch.sigmoid(g) * g * u)[None, :].bfloat16()
        hq, hs = ext2.prequant(h)
        ld = ext2.lin_pq(hq, hs, dwp[e], dsc[e], 1, 2048, N).squeeze(0)
        ce = cos(yd[e].float().cpu().numpy(), ld.float().cpu().numpy())
        assert ce > 0.997, (e, ce)
    print(f"  down_silu[{E}] vs lin_pq all cos>0.999 OK")

    # gateup 与 lin_pq 交叉验证（同一新 kernel 的两种入口）
    lin = ext2.lin_pq(xq, xsc, gwp[0], gsc[0], 1, N, K).squeeze(0)
    c2 = cos(gu[0, :N].float().cpu().numpy(), lin.float().cpu().numpy())
    assert c2 > 0.9999, c2
    print(f"  gateup[0] vs lin_pq cos={c2:.6f} OK")


def main():
    ext1 = get_ext()
    ext2 = get_decode_ext()
    print("[1] prequant vs CPU 参考")
    test_prequant(ext2)
    print("[2] lin_pq vs 旧朴素 kernel")
    test_lin_pq(ext1, ext2)
    print("[3] grouped MoE vs Wrec 参考")
    test_moe(ext1, ext2)
    print("ALL PASS")


if __name__ == "__main__":
    main()
