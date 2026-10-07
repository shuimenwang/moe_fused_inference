#!/usr/bin/env python3
# decode_v2.py - P2 v2 路径 decode 基准（静态 MXFP4 + FP8 预分配 KV + GQA）

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "runtime"))

from weight_store import WeightStore
from model_runner_v2 import ModelRunnerV2
from decode_bench import build_cfg, mib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp8", default="/home/hngs/models/Qwen3-30B-A3B-FP8")
    ap.add_argument("--mxfp4", default="/home/hngs/models/Qwen3-30B-A3B-MXFP4")
    ap.add_argument("--static-mxfp4",
                    default="/home/hngs/models/Qwen3-30B-A3B-MXFP4-static")
    ap.add_argument("--prompt", default="In a future where artificial minds "
                    "share the world with us, the most important lesson is")
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--cache-cap", type=int, default=1500)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    cfg = build_cfg(args.fp8)
    tokenizer = AutoTokenizer.from_pretrained(args.fp8)
    total_mem = torch.cuda.get_device_properties(0).total_memory

    store = WeightStore(args.fp8, args.mxfp4,
                        expert_cache_cap=args.cache_cap,
                        static_mxfp4_dir=args.static_mxfp4)
    runner = ModelRunnerV2(store, cfg)

    sync = torch.cuda.synchronize
    sync(); t0 = time.time()
    runner.prepare(max_len=args.max_len)
    sync(); t_prep = time.time() - t0
    print(f"[prep] {t_prep:.2f}s，allocated={mib(torch.cuda.memory_allocated()):.0f} MiB")

    ids = tokenizer(args.prompt, return_tensors="pt").input_ids[0].cuda()
    pos = 0
    torch.cuda.reset_peak_memory_stats()
    sync(); t0 = time.time()
    logits, _ = runner.forward(ids, pos)
    sync(); ttft = time.time() - t0
    pos += ids.numel()
    nxt = logits[-1].argmax().reshape(1)
    print(f"[TTFT] {ttft*1000:.1f} ms（{ids.numel()} tokens），"
          f"peak={mib(torch.cuda.max_memory_allocated()):.0f} MiB")

    dts, gen = [], []
    for step in range(args.steps):
        sync(); t0 = time.time()
        logits, _ = runner.forward(nxt, pos)
        sync(); dt = time.time() - t0
        dts.append(dt)
        pos += 1
        nxt = logits[-1].argmax().reshape(1)
        tok = int(nxt.item())
        if tok == tokenizer.eos_token_id:
            break
        gen.append(tok)

    dts = np.array(dts) * 1000
    hit = store.cache_hit_rate()
    peak = torch.cuda.max_memory_allocated()
    kv_bytes = cfg["num_hidden_layers"] * 2 * cfg["num_key_value_heads"] \
        * cfg["head_dim"] * pos  # fp8：1 byte/elem
    print("\n" + "=" * 62)
    print(f"decode={len(dts)}，KV={pos}，TPOT mean={dts.mean():.2f} "
          f"p50={np.percentile(dts,50):.2f} p90={np.percentile(dts,90):.2f} "
          f"p99={np.percentile(dts,99):.2f} ms")
    print(f"吞吐 {1000/dts.mean():.1f} tok/s；cache 命中 {hit*100:.1f}% "
          f"({len(store.expert_cache)} cached)")
    print(f"allocated 峰值 {mib(peak):.0f}/{mib(total_mem):.0f} MiB "
          f"({peak/total_mem*100:.1f}%)；KV(fp8) {mib(kv_bytes):.0f} MiB")
    print("生成：", tokenizer.decode(gen)[:300])

    result = {
        "tag": args.tag, "steps": len(dts), "kv_len": pos,
        "ttft_ms": ttft * 1000, "prep_s": t_prep,
        "tpot_mean_ms": float(dts.mean()),
        "tpot_p50_ms": float(np.percentile(dts, 50)),
        "tpot_p90_ms": float(np.percentile(dts, 90)),
        "tpot_p99_ms": float(np.percentile(dts, 99)),
        "tok_per_s": float(1000 / dts.mean()),
        "hit_rate": hit, "cache_size": len(store.expert_cache),
        "cache_cap": args.cache_cap, "max_len": args.max_len,
        "peak_alloc_mib": mib(peak), "peak_pct": peak / total_mem * 100,
        "kv_fp8_mib": mib(kv_bytes),
    }
    out = args.out or os.path.join(_ROOT, "benchmarks", "results",
                                   f"decode_{args.tag}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(result, open(out, "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
