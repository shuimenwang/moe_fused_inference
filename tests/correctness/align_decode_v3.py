#!/usr/bin/env python3
# align_decode_v3.py - 验证 v3 CUDA-graph decode 与 v2 数值一致
#
# 流程：v2 prefill 得到 KV → 拷贝到 v3 → 对同一 token 单步 decode 对比 logits。

import os
import sys

import torch
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "runtime"))
sys.path.insert(0, os.path.join(_ROOT, "quant"))

from weight_store import WeightStore
from model_runner_v2 import ModelRunnerV2
from model_runner_v3 import ModelRunnerV3
from decode_bench import build_cfg

FP8 = "/home/hngs/models/Qwen3-30B-A3B-FP8"
MX = "/home/hngs/models/Qwen3-30B-A3B-MXFP4"
ST = "/home/hngs/models/Qwen3-30B-A3B-MXFP4-static"


def cos(a, b):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-9))


def main():
    from transformers import AutoTokenizer
    cfg = build_cfg(FP8)
    tok = AutoTokenizer.from_pretrained(FP8)
    prompt = ("In a future where artificial minds share the world with us, "
              "the most important lesson is")
    ids = tok(prompt, return_tensors="pt").input_ids[0].cuda()
    max_len = 1024

    store = WeightStore(FP8, MX, expert_cache_cap=3000, static_mxfp4_dir=ST)
    v2 = ModelRunnerV2(store, cfg)
    v2.prepare(max_len)
    v2.forward(ids, 0)            # prefill，KV[pos0..T)
    pos = ids.numel()
    nxt = torch.tensor([ids[-1]], device="cuda")

    # v2 / v3 都用 start_pos=pos，分别写各自 KV[pos]、读 S=pos+1
    ref, _ = v2.forward(nxt, pos)
    v2ref = nxt
    torch.cuda.synchronize()

    # v3：拷贝 v2 的 KV
    v3 = ModelRunnerV3(store, cfg)
    v3.prepare(max_len)
    v3.kv_k.copy_(v2.kv[0])
    v3.kv_v.copy_(v2.kv[1])

    got, _ = v3.forward(nxt, pos)   # start_pos=pos（与 v2 相同）
    torch.cuda.synchronize()

    c = cos(got, ref)
    neq = float(torch.argmax(got) == torch.argmax(ref))
    print(f"v3-vs-v2 decode: cos={c:.6f}  argmax_eq={neq}")
    print(f"  got argmax={tok.decode([torch.argmax(got).item()])!r}"
          f"  ref argmax={tok.decode([torch.argmax(ref).item()])!r}")
    assert c > 0.98, "v3 diverged from v2"


if __name__ == "__main__":
    main()