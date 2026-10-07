#!/usr/bin/env python3
# quantize_static_mxfp4.py - 非专家权重量化为 MXFP4 镜像（P2.1a）
#
# 目的：把 GPU 常驻的 bf16/fp32 静态权重（~2.94GB）压成 MXFP4（~0.85GB），
#       腾出的显存给 expert LRU cache（命中率 500cap 51% → 1500cap 84%）。
#
# 覆盖（2D 权重，走 1×32 block E8M0 + E2M1 nibble）：
#   model.embed_tokens.weight (151936,2048) F32
#   lm_head.weight            (151936,2048) F32
#   layers.L.self_attn.{q,k,v,o}_proj.weight （F8_E4M3 + 128 block scale_inv）
#   layers.L.mlp.gate.weight  (128,2048) F32
# 不覆盖：所有 1D norm（极小，运行时继续从 FP8 目录读）。
#
# 输出格式与主 MXFP4 镜像一致（fp4_data / fp4_scale + manifest.json），
# WeightStore 直接复用同一条 packed 读取路径。

import argparse
import json
import os
import re
import sys

from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantize_mxfp4 import (
    ShardWriter, quantize_expert_tensor, list_input_files,
)

# q/k/v/o 的 K（输入维）均为 2048/4096，整除 32；其余 2D 同样整除。
STATIC_RE = re.compile(
    r"^(?:"
    r"model\.embed_tokens\.weight"
    r"|lm_head\.weight"
    r"|model\.layers\.\d+\.self_attn\.(?:q|k|v|o)_proj\.weight"
    r"|model\.layers\.\d+\.mlp\.gate\.weight"
    r")$"
)


def run(input_dir, output_dir, shard_gb=2.0):
    os.makedirs(output_dir, exist_ok=True)
    writer = ShardWriter(output_dir, max_bytes=int(shard_gb * 1024 ** 3))
    manifest = {
        "format": "mxfp4-1x32-e8m0-e2m1-static", "block_size": 32,
        "scope": "non-expert 2D weights (embed/lm_head/attn/router)",
        "created": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
        "tensors": {},
    }
    n_q = 0
    fp4_bytes = scale_bytes = 0
    for fn in list_input_files(input_dir):
        with safe_open(os.path.join(input_dir, fn), framework="torch") as st:
            keys = list(st.keys())
            for name in keys:
                if not STATIC_RE.match(name):
                    continue
                t = st.get_tensor(name)
                scale_key = name + "_scale_inv"
                s_inv = st.get_tensor(scale_key) if scale_key in keys else None
                packed, scales, info = quantize_expert_tensor(t, s_inv)
                dk, sk = name + ".fp4_data", name + ".fp4_scale"
                writer.add(dk, packed)
                writer.add(sk, scales)
                info.update({"data_key": dk, "scale_key": sk,
                             "fp8_scale_inv_present": s_inv is not None})
                manifest["tensors"][name] = info
                n_q += 1
                fp4_bytes += packed.nbytes
                scale_bytes += scales.nbytes
                print(f"[{n_q:4d}] {name:55s} {list(t.shape)} "
                      f"-> {packed.nbytes/1024**2:.1f}MB")
    writer.finalize()
    manifest["stats"] = {"tensors": n_q,
                         "fp4_gib": fp4_bytes / 1024**3,
                         "scale_gib": scale_bytes / 1024**3}
    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n完成：{n_q} tensors，fp4 {fp4_bytes/1024**3:.3f}GiB + "
          f"scale {scale_bytes/1024**3:.3f}GiB → {output_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/home/hngs/models/Qwen3-30B-A3B-FP8")
    ap.add_argument("--output",
                    default="/home/hngs/models/Qwen3-30B-A3B-MXFP4-static")
    args = ap.parse_args()
    run(args.input, args.output)
