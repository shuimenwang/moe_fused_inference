#!/usr/bin/env python3
# test_mxfp4_mma.py - P0 步骤5：MXFP4 block-scale MMA 算子数值正确性
#
# 对照：
#   ref_rec : x_f @ W_rec.T   —— 量化权重(nibble*scale)的“理想执行”，隔离 MMA/激活量化
#   ref_true: x_f @ W_f.T     —— 量化前真值
# 判定：cosine(ext, ref_rec)≈1 且 rel err 小 → fragment/scale 映射正确

import os
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "runtime"))
sys.path.insert(0, os.path.join(_ROOT, "quant"))

from fp4_ext import get_ext
from quantize_mxfp4 import quantize_expert_tensor, MXFP4_BLOCK
from fp4_format import unpack_nibbles, decode_e2m1, e8m0_decode


def make_mxfp4_weight(Wf):
    """f32 (n,k) → wp(n,kp/2) uint8 cuda, wsc(n,kp/32), Wrec f32, info。"""
    packed, scodes, info = quantize_expert_tensor(torch.from_numpy(Wf))
    n, kp = info["orig_shape"][0], info["padded_k"]
    wp = torch.from_numpy(packed.reshape(n, kp // 2)).cuda()
    wsc = torch.from_numpy(scodes).cuda()
    codes = unpack_nibbles(packed, n=n * kp).reshape(n, kp // MXFP4_BLOCK, MXFP4_BLOCK)
    Wrec = (decode_e2m1(codes).astype(np.float32)
            * e8m0_decode(scodes)[..., None]).reshape(n, kp)[:, :info["orig_shape"][1]]
    return wp, wsc, Wrec.astype(np.float32), info


def stats(a, b):
    a = a.reshape(-1)
    b = b.reshape(-1)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    rel = float(np.abs(a - b).sum() / (np.abs(a).sum() + 1e-12))
    return cos, rel


def run_case(ext, M, N, K, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    # 混合动态范围（不同行 scale），并加少量 outlier
    Wf = rng.standard_normal((N, K)).astype(np.float32)
    Wf *= np.power(2.0, rng.integers(-1, 2, size=(N, 1))).astype(np.float32)
    x = rng.standard_normal((M, K)).astype(np.float32) * 0.5
    x[rng.random((M, K)) < 0.005] *= 4.0

    wp, wsc, Wrec, info = make_mxfp4_weight(Wf)
    xh = torch.from_numpy(x).half().cuda()

    out = ext.forward(xh, wp, wsc).float().cpu().numpy()

    ref_rec = x @ Wrec.T
    ref_true = x @ Wf.T

    cos_rec, rel_rec = stats(out, ref_rec)
    cos_true, rel_true = stats(out, ref_true)
    if verbose:
        print(f"  shape M={M} N={N} K={K} (pad={info['pad']})")
        print(f"    vs ref_rec : cos={cos_rec:.6f} rel_l1={rel_rec:.6f}")
        print(f"    vs ref_true: cos={cos_true:.6f} rel_l1={rel_true:.6f}")
    return cos_rec, rel_rec, cos_true


def main():
    print("编译/加载 MXFP4 MMA 扩展 ...")
    ext = get_ext()

    print("\n[1] 单 tile (16,8,64) —— 验证 fragment/scale 映射")
    c, r, ct = run_case(ext, 16, 8, 64)
    if c < 0.98:
        print("  >>> 映射可能错误（cos vs ref_rec 过低），需诊断 nibble/scale 布局")
    else:
        print("  >>> 单 tile 映射正确")

    print("\n[2] 单 tile 不同 seed")
    for s in (1, 2):
        run_case(ext, 16, 8, 64, seed=s)

    print("\n[3] 非整除 K（pad 到 /32）")
    run_case(ext, 16, 8, 100)
    run_case(ext, 16, 8, 40)

    print("\n[4] decode 实际小 M（M=1）")
    run_case(ext, 1, 8, 64)
    run_case(ext, 1, 32, 64)

    print("\n[5] 多 tile（N/M 跨 block）+ 真实专家形 K=2048")
    run_case(ext, 16, 32, 64)
    run_case(ext, 32, 32, 64)
    run_case(ext, 8, 64, 2048)

    print("\n[6] Qwen3 专家 gate/up 投影 (N=768,K=2048)，decode M=1")
    c, r, ct = run_case(ext, 1, 768, 2048)
    print("\n[7] down 投影 (N=2048,K=768)，多 token M=8")
    c2, r2, ct2 = run_case(ext, 8, 2048, 768)

    ok = c > 0.95 and c2 > 0.95
    print("\n" + "=" * 56)
    print(f" 总体判定：{'PASS — MXFP4 MMA 算子数值正确' if ok else 'FAIL — 需排查'}")
    print("=" * 56)


if __name__ == "__main__":
    main()
