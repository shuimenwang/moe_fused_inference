#!/usr/bin/env python3
# quantize_mxfp4.py - Streaming MXFP4 量化器（P0 主路径）
#
# Qwen3-30B-A3B 无官方 FP4 checkpoint，自量化是主路径。
# 本脚本逐 safetensors 文件、逐 tensor 处理（不整模入内存）：
#   专家权重（experts.{j}.{gate,up,down}_proj.weight）
#     → 沿最后一维 1×32 block → E8M0 2的幂 scale → E2M1 nibble → LSB-first 打包
#   非专家 tensor 默认跳过（runner 直接从 FP8 checkpoint 加载它们）
#
# 输出：
#   {output}/model[-XXXXX-of-XXXXX].safetensors   （fp4_data uint8 + fp4_scale float32）
#   {output}/manifest.json                        （原始 shape/dtype/pad，runner 重建用）
#
# --selftest：合成迷你 checkpoint（含非整除K、fp32/fp8 两种输入）跑完整链路，
#             读回重建并产出误差报告，落盘 quant/selftest/。

import argparse
import json
import os
import re
import shutil
import time

import numpy as np
import torch
from safetensors import safe_open
from safetensors.numpy import save_file

from fp4_format import (
    decode_e2m1, e8m0_decode, mxfp4_block_scale, pack_nibbles,
    quantize_e2m1, unpack_nibbles,
)

DEFAULT_EXPERT_RE = r"experts\.\d+\.(gate|up|down)_proj\.weight$"
MXFP4_BLOCK = 32


# ------------------------------------------------------------------
# Shard 写入：累积 tensor 到阈值落盘，finalize 时 rename + 写 index
# ------------------------------------------------------------------
class ShardWriter:
    def __init__(self, out_dir, max_bytes=2 * 1024 ** 3):
        self.out_dir = out_dir
        self.max_bytes = max_bytes
        os.makedirs(out_dir, exist_ok=True)
        self.buf, self.nbytes = {}, 0
        self.tmp_shards, self.weight_map = [], {}

    def add(self, name, arr):
        self.buf[name] = np.ascontiguousarray(arr)
        self.nbytes += arr.nbytes
        if self.nbytes >= self.max_bytes:
            self._flush()

    def _flush(self):
        if not self.buf:
            return
        idx = len(self.tmp_shards) + 1
        fn = f"tmp-{idx:05d}.safetensors"
        save_file(self.buf, os.path.join(self.out_dir, fn))
        for k in self.buf:
            self.weight_map[k] = fn
        self.tmp_shards.append(fn)
        self.buf, self.nbytes = {}, 0

    def finalize(self):
        self._flush()
        n = len(self.tmp_shards)
        if n == 1:
            final = "model.safetensors"
            os.replace(os.path.join(self.out_dir, self.tmp_shards[0]),
                       os.path.join(self.out_dir, final))
            self.weight_map = {k: final for k in self.weight_map}
            return None
        for i, tmp in enumerate(self.tmp_shards, 1):
            final = f"model-{i:05d}-of-{n:05d}.safetensors"
            os.replace(os.path.join(self.out_dir, tmp), os.path.join(self.out_dir, final))
            self.weight_map = {k: (final if v == tmp else v) for k, v in self.weight_map.items()}
        index = {
            "metadata": {"total_size": int(sum(
                os.path.getsize(os.path.join(self.out_dir, f))
                for f in set(self.weight_map.values())))},
            "weight_map": self.weight_map,
        }
        with open(os.path.join(self.out_dir, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f, indent=2)
        return index


# ------------------------------------------------------------------
# FP8 128×128 block dequant（官方 FP8 checkpoint 带 weight_scale_inv）
#   W_bf16[b,i] = fp8 值 * scale_inv[out_block, in_block]
# ------------------------------------------------------------------
FP8_WEIGHT_BLOCK = 128


def dequant_fp8_block(w8: torch.Tensor, s_inv: torch.Tensor):
    out, inp = w8.shape
    assert out % FP8_WEIGHT_BLOCK == 0 and inp % FP8_WEIGHT_BLOCK == 0, \
        f"FP8 block dequant 要求维度整除128: {w8.shape}"
    w = w8.float().view(out // FP8_WEIGHT_BLOCK, FP8_WEIGHT_BLOCK,
                        inp // FP8_WEIGHT_BLOCK, FP8_WEIGHT_BLOCK)
    w = w * s_inv.view(out // FP8_WEIGHT_BLOCK, 1, inp // FP8_WEIGHT_BLOCK, 1).float()
    return w.reshape(out, inp)


# ------------------------------------------------------------------
# 单 tensor 量化
# ------------------------------------------------------------------
def quantize_expert_tensor(w: torch.Tensor, s_inv: torch.Tensor = None):
    """2D 权重 → (packed uint8, scale_codes uint8[N,K/32] (E8M0), manifest 条目)。

    s_inv 非 None 时先做 FP8 128×128 block dequant 还原真实权重值。
    scale 全程 FP32 参与计算，落盘只存 E8M0 指数 code（=2 的幂，8-bit/block）。
    """
    orig_shape = list(w.shape)
    orig_dtype = str(w.dtype)
    if s_inv is not None:
        wf = dequant_fp8_block(w, s_inv).numpy().astype(np.float32)
    else:
        wf = w.float().numpy().astype(np.float32)
    n, k = wf.shape

    pad = int((-k) % MXFP4_BLOCK)
    if pad:
        wf = np.pad(wf, ((0, 0), (0, pad)))
    kp = wf.shape[1]
    blocks = wf.reshape(n, kp // MXFP4_BLOCK, MXFP4_BLOCK)

    amax = np.max(np.abs(blocks), axis=-1)              # (n, nb)
    # 向量化 E8M0 scale：scale = 2^ceil(log2(amax/6))
    scales = np.ones_like(amax, dtype=np.float32)
    scale_codes = np.full(amax.shape, 127, dtype=np.uint8)  # 全零块 scale=1.0
    nz = amax > 0
    e = np.ceil(np.log2(np.maximum(amax[nz] / 6.0, 1e-30))).astype(np.int64)
    s = np.power(2.0, e).astype(np.float32)
    # 浮点边界兜底：若仍有元素超出 6（log2/ceil 边界误差），再提一档
    overflow = amax[nz] / s > 6.0 + 1e-6
    if np.any(overflow):
        s[overflow] *= 2.0
        e[overflow] += 1
    scales[nz] = s
    scale_codes[nz] = np.clip(e + 127, 0, 255).astype(np.uint8)

    scaled = blocks / scales[..., None]
    codes = quantize_e2m1(scaled)                       # (n, nb, 32)
    packed, _ = pack_nibbles(codes.reshape(-1))

    # 不变量：量化前没有元素超出 6（否则 scale 选择有 bug）
    assert np.all(scaled <= 6.0 + 1e-6), "block scale 未覆盖块内最大值"
    info = {
        "orig_shape": orig_shape,
        "orig_dtype": orig_dtype,
        "padded_k": kp,
        "pad": pad,
        "n_elements": int(n * k),
        "scale_shape": list(scale_codes.shape),
        "scale_dtype": "e8m0_uint8",
    }
    return packed, scale_codes, info


# ------------------------------------------------------------------
# checkpoint 级量化主流程
# ------------------------------------------------------------------
def list_input_files(input_dir):
    idx = os.path.join(input_dir, "model.safetensors.index.json")
    if os.path.exists(idx):
        files = sorted(set(json.load(open(idx))["weight_map"].values()))
    else:
        files = [f for f in os.listdir(input_dir) if f.endswith(".safetensors")]
    return sorted(files)


def run_quantization(input_dir, output_dir, pattern, shard_gb, copy_non_experts):
    rx = re.compile(pattern)
    writer = ShardWriter(output_dir, max_bytes=int(shard_gb * 1024 ** 3))
    manifest = {
        "format": "mxfp4-1x32-e8m0-e2m1", "block_size": MXFP4_BLOCK,
        "scale_type": "e8m0_power_of_2", "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tensors": {},
    }
    stats = {"expert_tensors": 0, "copied_tensors": 0, "skipped_tensors": 0,
             "consumed_scale_inv": 0,
             "input_bytes": 0, "fp4_bytes": 0, "scale_bytes": 0}

    files = list_input_files(input_dir)
    assert files, f"输入目录无 safetensors: {input_dir}"
    for fn in files:
        with safe_open(os.path.join(input_dir, fn), framework="torch") as st:
            all_keys = st.keys()
            for name in all_keys:
                t = st.get_tensor(name)
                stats["input_bytes"] += t.element_size() * t.numel()
                if rx.search(name):
                    scale_key = name + "_scale_inv"
                    s_inv = st.get_tensor(scale_key) if scale_key in all_keys else None
                    packed, scales, info = quantize_expert_tensor(t, s_inv)
                    dk, sk = name + ".fp4_data", name + ".fp4_scale"
                    writer.add(dk, packed)
                    writer.add(sk, scales)
                    info.update({"data_key": dk, "scale_key": sk,
                                 "fp8_scale_inv_present": s_inv is not None})
                    manifest["tensors"][name] = info
                    stats["expert_tensors"] += 1
                    stats["fp4_bytes"] += packed.nbytes
                    stats["scale_bytes"] += scales.nbytes
                    if s_inv is not None:
                        stats["consumed_scale_inv"] += 1
                elif name.endswith("_scale_inv"):
                    pass  # FP8 权重附属 scale tensor（专家/注意力），不计数
                elif copy_non_experts:
                    arr = t.float().numpy()   # 统一 float32（bf16/fp8 无直接 numpy）
                    writer.add(name, arr)
                    stats["copied_tensors"] += 1
                else:
                    stats["skipped_tensors"] += 1

    writer.finalize()
    manifest["stats"] = stats
    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


# ------------------------------------------------------------------
# 读回重建 + 误差报告（selftest 用，也可独立验证任何镜像）
# ------------------------------------------------------------------
def reconstruct_and_report(input_dir, mirror_dir, report_path, label):
    manifest = json.load(open(os.path.join(mirror_dir, "manifest.json")))
    files = list_input_files(mirror_dir)
    handles = {fn: safe_open(os.path.join(mirror_dir, fn), framework="numpy") for fn in files}

    def get(key):
        for h in handles.values():
            if key in h.keys():
                return h.get_tensor(key)
        raise KeyError(key)

    report = {"label": label, "tensors": {}}
    agg_cos, agg_err, agg_rel, total_elem = [], [], [], 0
    for name, info in manifest["tensors"].items():
        data = get(info["data_key"])
        scales = e8m0_decode(get(info["scale_key"]))
        n, kp = info["orig_shape"][0], info["padded_k"]
        codes = unpack_nibbles(data, n=n * kp).reshape(n, kp // MXFP4_BLOCK, MXFP4_BLOCK)
        deq = decode_e2m1(codes).astype(np.float32)
        recon = (deq * scales[..., None]).reshape(n, kp)[:, : info["orig_shape"][1]]

        # 原始值（以输入 checkpoint 实际 dtype decode 后为基准，即只度量 FP4 增量误差）
        src_file = list_input_files(input_dir)
        orig = None
        for fn in src_file:
            with safe_open(os.path.join(input_dir, fn), framework="torch") as st:
                if name in st.keys():
                    orig = st.get_tensor(name).float().numpy()
        assert orig is not None

        err = np.abs(recon - orig)
        cos = float(np.dot(recon.ravel(), orig.ravel()) /
                    (np.linalg.norm(recon) * np.linalg.norm(orig) + 1e-12))
        rel = float(err.sum() / (np.abs(orig).sum() + 1e-12))
        report["tensors"][name] = {
            "shape": info["orig_shape"], "cosine": round(cos, 6),
            "max_abs_err": float(err.max()),
            "mean_abs_err": float(err.mean()),
            "relative_l1": round(rel, 6),
            "max_scale": float(scales.max()), "min_scale": float(scales.min()),
        }
        agg_cos.append(cos)
        total_elem += orig.size
        agg_err.append(err.sum())
        agg_rel.append(np.abs(orig).sum())

    for h in handles.values():
        del h
    report["aggregate"] = {
        "mean_cosine": round(float(np.mean(agg_cos)), 6),
        "min_cosine": round(float(min(agg_cos)), 6),
        "weighted_relative_l1": round(float(sum(agg_err) / (sum(agg_rel) + 1e-12)), 6),
        "tensors": len(manifest["tensors"]),
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    return report


# ------------------------------------------------------------------
# selftest：合成迷你 checkpoint（fp32 / fp8 两种输入 + 非整除 K）
# ------------------------------------------------------------------
def make_mini_checkpoint(ckpt_dir, dtype):
    os.makedirs(ckpt_dir, exist_ok=True)
    rng = np.random.default_rng(42)

    def make(n, k):
        # 混合动态范围：部分行/块带 outlier，迫使 scale 逐块不同
        x = rng.standard_normal((n, k)).astype(np.float32)
        row_gain = np.power(2.0, rng.integers(-2, 3, size=(n, 1))).astype(np.float32)
        x = (x * row_gain).astype(np.float32)
        x[rng.random((n, k)) < 0.01] *= np.float32(rng.choice([8.0, -8.0]))
        t = torch.from_numpy(x)
        if dtype == "fp8":
            t = t.to(torch.float8_e4m3fn)
        return t

    tensors = {}
    for layer in (0, 1):
        for ex in (0, 1):
            pre = f"model.layers.{layer}.mlp.experts.{ex}"
            # 96/128 整除32；down 用 K=100 测 pad（100→128）
            tensors[f"{pre}.gate_proj.weight"] = make(96, 128)
            tensors[f"{pre}.up_proj.weight"] = make(96, 128)
            tensors[f"{pre}.down_proj.weight"] = make(128, 100)
    # 非专家 tensor（验证默认跳过）
    tensors["model.layers.0.input_layernorm.weight"] = torch.ones(128)
    from safetensors.torch import save_file as torch_save_file
    torch_save_file(tensors, os.path.join(ckpt_dir, "model.safetensors"))


def selftest(base_dir):
    out = {"runs": {}}
    for dtype in ("fp32", "fp8"):
        ckpt = os.path.join(base_dir, f"mini_checkpoint_{dtype}")
        mirror = os.path.join(base_dir, f"mxfp4_mirror_{dtype}")
        for d in (ckpt, mirror):
            if os.path.exists(d):
                shutil.rmtree(d)
        make_mini_checkpoint(ckpt, dtype)
        manifest = run_quantization(ckpt, mirror, DEFAULT_EXPERT_RE, shard_gb=2,
                                    copy_non_experts=False)
        report = reconstruct_and_report(ckpt, mirror,
                                        os.path.join(base_dir, f"quant_report_{dtype}.json"),
                                        f"mini-{dtype}")
        out["runs"][dtype] = {"stats": manifest["stats"], "aggregate": report["aggregate"]}

    with open(os.path.join(base_dir, "selftest_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="HF checkpoint 目录")
    ap.add_argument("--output", help="MXFP4 镜像输出目录")
    ap.add_argument("--pattern", default=DEFAULT_EXPERT_RE)
    ap.add_argument("--shard-gb", type=float, default=2.0)
    ap.add_argument("--copy-non-experts", action="store_true")
    ap.add_argument("--selftest", action="store_true",
                    help="合成迷你 checkpoint 验证全链路（落盘 quant/selftest/）")
    ap.add_argument("--verify", action="store_true",
                    help="量化后读回重建并产出误差报告")
    args = ap.parse_args()

    if args.selftest:
        base = os.path.dirname(os.path.abspath(__file__))
        st_dir = os.path.join(base, "selftest")
        os.makedirs(st_dir, exist_ok=True)
        result = selftest(st_dir)
        print("=" * 60)
        print(" MXFP4 QUANTIZER SELFTEST 结论")
        print("=" * 60)
        for dtype, r in result["runs"].items():
            a, s = r["aggregate"], r["stats"]
            print(f" [{dtype}] 专家tensor={s['expert_tensors']} "
                  f"跳过={s['skipped_tensors']}")
            print(f"   fp4={s['fp4_bytes']/1024:.0f}KB "
                  f"scale={s['scale_bytes']/1024:.0f}KB "
                  f"输入={s['input_bytes']/1024:.0f}KB")
            print(f"   mean_cos={a['mean_cosine']} min_cos={a['min_cosine']} "
                  f"rel_l1={a['weighted_relative_l1']}")
        print("=" * 60)
        print(f" 落盘: {st_dir}")
        return

    assert args.input and args.output, "真实量化需 --input 与 --output"
    manifest = run_quantization(args.input, args.output, args.pattern,
                                args.shard_gb, args.copy_non_experts)
    s = manifest["stats"]
    print(f"完成：专家 {s['expert_tensors']} / 跳过 {s['skipped_tensors']} / "
          f"复制 {s['copied_tensors']}")
    print(f"fp4 {s['fp4_bytes']/1024**3:.2f}GB + scale {s['scale_bytes']/1024**3:.2f}GB"
          f"（输入 {s['input_bytes']/1024**3:.2f}GB）")
    if args.verify:
        report = reconstruct_and_report(
            args.input, args.output,
            os.path.join(args.output, "quant_report.json"), "real")
        print(f"mean cosine={report['aggregate']['mean_cosine']} "
              f"rel_l1={report['aggregate']['weighted_relative_l1']}")


if __name__ == "__main__":
    main()
