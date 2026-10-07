#!/usr/bin/env python3
# decode_bench.py - P0 步骤6：分层 loader + decode loop + 8GB 显存水位 / TTFT / TPOT
#
# 分层策略（复用 WeightStore + ModelRunner）：
#   非专家权重（embed/lm_head/attn/gate/norm）：FP8 dequant → GPU 常驻
#   128 专家权重：从 MXFP4 镜像按 active 集合逐层读取，算完即释放（不常驻）
#
# 测量：
#   静态权重加载耗时；prefill TTFT；连续 decode（默认 128 步）TPOT p50/p90；
#   torch allocator allocated/reserved 峰值 + 设备级 mem_get_info 水位；
#   KV cache 估算。连续 decode 不 OOM 即通过。

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
from model_runner import ModelRunner


def build_cfg(fp8_dir):
    r = json.load(open(os.path.join(fp8_dir, "config.json")))
    return {
        "num_hidden_layers": r["num_hidden_layers"],
        "hidden_size": r["hidden_size"],
        "intermediate_size": r.get("intermediate_size"),
        "moe_intermediate_size": r.get("moe_intermediate_size", 768),
        "vocab_size": r["vocab_size"],
        "num_attention_heads": r["num_attention_heads"],
        "num_key_value_heads": r["num_key_value_heads"],
        "head_dim": r.get("head_dim", r["hidden_size"] // r["num_attention_heads"]),
        "rms_norm_eps": r["rms_norm_eps"],
        "rope_theta": r.get("rope_theta", 1e6),
        "num_experts": r["num_experts"],
        "num_experts_per_tok": r["num_experts_per_tok"],
        "norm_topk_prob": r.get("norm_topk_prob", True),
    }


def sync():
    torch.cuda.synchronize()


def mib(x):
    return x / (1024 ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp8", default="/home/hngs/models/Qwen3-30B-A3B-FP8")
    ap.add_argument("--mxfp4", default="/home/hngs/models/Qwen3-30B-A3B-MXFP4")
    ap.add_argument("--variant", default="mxfp4_mma",
                    choices=["mxfp4_mma", "mxfp4", "fp8"])
    ap.add_argument("--prompt", default="In a future where artificial minds "
                    "share the world with us, the most important lesson is")
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--cache-cap", type=int, default=500)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    cfg = build_cfg(args.fp8)
    tokenizer = AutoTokenizer.from_pretrained(args.fp8)

    total_mem = torch.cuda.get_device_properties(0).total_memory
    print(f"设备总显存 {mib(total_mem):.0f} MiB；variant={args.variant}")

    store = WeightStore(args.fp8, args.mxfp4, expert_cache_cap=args.cache_cap)
    runner = ModelRunner(store, cfg)

    # ---- 1) 非专家权重常驻加载（不计入 TTFT）----
    sync(); t0 = time.time()
    runner._ensure_static()
    sync(); t_load = time.time() - t0
    nstatic = len(runner.static)
    print(f"[load] 非专家权重 {nstatic} tensor 常驻，耗时 {t_load:.2f}s，"
          f"allocated={mib(torch.cuda.memory_allocated()):.0f} MiB")

    ids = tokenizer(args.prompt, return_tensors="pt").input_ids[0].cuda()
    print(f"[prefill] prompt tokens = {ids.numel()}")

    # ---- 2) prefill → TTFT ----
    torch.cuda.reset_peak_memory_stats()
    sync(); t0 = time.time()
    logits, kvs, _ = runner.forward(ids, args.variant)
    sync(); ttft = time.time() - t0
    nxt = logits[-1].argmax().reshape(1)
    peak_prefill = torch.cuda.max_memory_allocated()
    print(f"[TTFT] {ttft*1000:.1f} ms  ({ttft/ids.numel()*1000:.2f} ms/token)，"
          f"peak allocated={mib(peak_prefill):.0f} MiB")

    # ---- 3) decode loop → TPOT/ITL 分布 ----
    dts, gen = [], []
    free_min = total_mem
    for step in range(args.steps):
        sync(); t0 = time.time()
        logits, kvs, _ = runner.forward(nxt, args.variant, past_kvs=kvs,
                                        need_lm_head=True)
        sync(); dt = time.time() - t0
        dts.append(dt)
        nxt = logits[-1].argmax().reshape(1)
        tok = int(nxt.item())
        if tok == tokenizer.eos_token_id:
            break
        gen.append(tok)
        fi, _ = torch.cuda.mem_get_info()
        free_min = min(free_min, fi)

    dts = np.array(dts)
    tpot_ms = dts * 1000
    peak_all = torch.cuda.max_memory_allocated()
    peak_rsv = torch.cuda.max_memory_reserved()
    hit_rate = store.cache_hit_rate()
    cache_sz = len(store.expert_cache)

    # ---- 4) 汇总 ----
    kv_layers = len(kvs)
    kv_len = kvs[0][0].shape[0]
    kv_bytes = (kv_layers * 2 * kv_len * cfg["num_key_value_heads"]
                * cfg["head_dim"] * 2)  # bf16
    print("\n" + "=" * 62)
    print(f"decode 步数 = {len(dts)}（KV 长度 {kv_len}）")
    print(f"TPOT  mean={tpot_ms.mean():.2f}  p50={np.percentile(tpot_ms,50):.2f}  "
          f"p90={np.percentile(tpot_ms,90):.2f}  p99={np.percentile(tpot_ms,99):.2f} ms，"
          f"吞吐 {1000/tpot_ms.mean():.1f} tok/s  (p99/mean="
          f"{np.percentile(tpot_ms,99)/max(tpot_ms.mean(),1e-6):.2f}×)")
    print(f"TTFT={ttft*1000:.1f} ms；静态加载={t_load:.2f}s")
    print(f"expert cache: 命中率 {hit_rate*100:.1f}%，缓存 {cache_sz} 个 expert")
    print("\n显存水位：")
    print(f"  allocated 峰值 {mib(peak_all):.0f} / {mib(total_mem):.0f} MiB "
          f"({peak_all/total_mem*100:.1f}%)")
    print(f"  reserved  峰值 {mib(peak_rsv):.0f} MiB；"
          f"decode 最低空闲 {mib(free_min):.0f} MiB")
    print(f"  KV cache 估算 {mib(kv_bytes):.0f} MiB")

    print("\n生成文本：")
    print(tokenizer.decode(gen))

    result = {
        "variant": args.variant,
        "prompt_len": int(ids.numel()),
        "static_load_s": t_load,
        "ttft_ms": ttft * 1000,
        "decode_steps": len(dts),
        "kv_len": kv_len,
        "tpot_mean_ms": float(tpot_ms.mean()),
        "tpot_p50_ms": float(np.percentile(tpot_ms, 50)),
        "tpot_p90_ms": float(np.percentile(tpot_ms, 90)),
        "tpot_p99_ms": float(np.percentile(tpot_ms, 99)),
        "tok_per_s": float(1000 / tpot_ms.mean()),
        "expert_cache_hit_rate": hit_rate,
        "expert_cache_size": cache_sz,
        "peak_allocated_mib": mib(peak_all),
        "peak_reserved_mib": mib(peak_rsv),
        "total_mib": mib(total_mem),
        "min_free_mib": mib(free_min),
        "kv_cache_mib": mib(kv_bytes),
        "generated": tokenizer.decode(gen),
    }
    json.dump(result, open(os.path.join(_ROOT, "quant", "decode_bench.json"), "w"),
              indent=2, ensure_ascii=False)

    no_oom = peak_all < total_mem
    print("\n" + "=" * 62)
    print(("PASS — 连续 %d 步 decode 不 OOM，8GB 分层推理跑通" % len(dts))
          if no_oom else "FAIL — 显存超限")
    print("=" * 62)


if __name__ == "__main__":
    main()
