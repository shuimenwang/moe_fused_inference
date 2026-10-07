#!/usr/bin/env python3
# profile_tpot.py - P2 决策 profile：TPOT 分层归因 + 路由轨迹 LRU 模拟 + H2D 微基准
#
# 三类数据：
#  A) 逐层 wall/cuda-event 计时：attention / route(.tolist 同步) / CPU 读盘 / H2D / expert kernel
#  B) 记录每个 decode step 的 (layer, expert) 轨迹 → 离线模拟任意 cache cap 的 LRU 命中率
#  C) H2D 微基准：pageable vs mmap+cudaHostRegister pinned（zero-copy 直通）
#
# 注意：torch GPU launch 必须在沙箱外运行。

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
from model_runner import ModelRunner, rms_norm


def build_cfg(fp8_dir):
    r = json.load(open(os.path.join(fp8_dir, "config.json")))
    return {
        "num_hidden_layers": r["num_hidden_layers"],
        "num_attention_heads": r["num_attention_heads"],
        "num_key_value_heads": r["num_key_value_heads"],
        "head_dim": r.get("head_dim", r["hidden_size"] // r["num_attention_heads"]),
        "rms_norm_eps": r["rms_norm_eps"],
        "rope_theta": r.get("rope_theta", 1e6),
        "num_experts": r["num_experts"],
        "num_experts_per_tok": r["num_experts_per_tok"],
        "norm_topk_prob": r.get("norm_topk_prob", True),
    }


def ev():
    return torch.cuda.Event(enable_timing=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp8", default="/home/hngs/models/Qwen3-30B-A3B-FP8")
    ap.add_argument("--mxfp4", default="/home/hngs/models/Qwen3-30B-A3B-MXFP4")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--cache-cap", type=int, default=500)
    ap.add_argument("--out", default=os.path.join(_ROOT, "benchmarks", "results",
                                                  "tpot_profile.json"))
    args = ap.parse_args()

    from transformers import AutoTokenizer
    cfg = build_cfg(args.fp8)
    tokenizer = AutoTokenizer.from_pretrained(args.fp8)
    L_all = cfg["num_hidden_layers"]

    store = WeightStore(args.fp8, args.mxfp4, expert_cache_cap=args.cache_cap)
    runner = ModelRunner(store, cfg)
    runner._ensure_static()
    torch.cuda.synchronize()

    prompt = ("In a future where artificial minds share the world with us, "
              "the most important lesson is")
    ids = tokenizer(prompt, return_tensors="pt").input_ids[0].cuda()
    logits, kvs, _ = runner.forward(ids, "mxfp4_mma")
    nxt = logits[-1].argmax().reshape(1)

    # 预热 8 步（cache 热身，不计入）
    for _ in range(8):
        logits, kvs, _ = runner.forward(nxt, "mxfp4_mma", past_kvs=kvs)
        nxt = logits[-1].argmax().reshape(1)
    store.cache_stats.update(hits=0, misses=0)

    # ---------- A) 分层计时 ----------
    agg = {"attn": 0.0, "route": 0.0, "cpu_read": 0.0, "h2d": 0.0,
           "kernel": 0.0, "combine": 0.0, "misc": 0.0}
    h2d_bytes = 0
    trace = []          # 每步：list[L] -> list[eid]
    per_step = []

    # 给 CPU 读盘计时：monkey-patch WeightStore._mx_get（numpy safetensors 读取）
    orig_mx_get = store._mx_get
    cpu_read_ns = [0]

    def timed_mx_get(key):
        t0 = time.perf_counter_ns()
        t = orig_mx_get(key)
        cpu_read_ns[0] += time.perf_counter_ns() - t0
        return t
    store._mx_get = timed_mx_get

    # load_layer_experts 整体 wall time；H2D ≈ wall - CPU 读盘
    timed = {"load_wall": [0.0]}
    orig_load = store.load_layer_experts

    def timed_load(layer, variant, active=None):
        t0 = time.perf_counter_ns()
        r = orig_load(layer, variant, active=active)
        timed["load_wall"][0] += time.perf_counter_ns() - t0
        return r
    store.load_layer_experts = timed_load

    # attention / moe 分别计时：patch runner 方法
    orig_attn = runner.attention
    orig_route = runner.moe_route
    orig_compute = runner.moe_compute
    buckets = {"attn": [0.0], "route": [0.0], "compute": [0.0]}

    def t_attn(h_, L, kv):
        s, e = ev(), ev(); s.record()
        r = orig_attn(h_, L, kv)
        e.record(); torch.cuda.synchronize()
        buckets["attn"][0] += s.elapsed_time(e)
        return r

    def t_route(h_, L):
        t0 = time.perf_counter_ns()
        r = orig_route(h_, L)
        buckets["route"][0] += (time.perf_counter_ns() - t0) / 1e6
        return r

    def t_compute(h_, topi, scores, experts, variant):
        s, e = ev(), ev(); s.record()
        r = orig_compute(h_, topi, scores, experts, variant)
        e.record(); torch.cuda.synchronize()
        buckets["compute"][0] += s.elapsed_time(e)
        return r

    runner.attention = t_attn
    runner.moe_route = t_route
    runner.moe_compute = t_compute

    # 为抓路由轨迹，patch store.load_layer_experts 的 active 参数
    trace_load = orig_load

    def trace_load(layer, variant, active=None):
        if trace_step[0] is not None and active is not None:
            trace_step[0][layer] = sorted(active)
        return timed_load(layer, variant, active=active)
    store.load_layer_experts = trace_load
    trace_step = [None]

    for step in range(args.steps):
        trace_step[0] = {}
        t0 = time.perf_counter_ns()
        logits, kvs, _ = runner.forward(nxt, "mxfp4_mma", past_kvs=kvs)
        torch.cuda.synchronize()
        dt = (time.perf_counter_ns() - t0) / 1e6
        per_step.append(dt)
        trace.append(trace_step[0])
        nxt = logits[-1].argmax().reshape(1)

    total_ms = sum(per_step)
    load_ms = timed["load_wall"][0] / 1e6
    cpu_ms = cpu_read_ns[0] / 1e6

    # ---------- B) LRU 离线模拟 ----------
    def sim_lru(cap):
        cache = {}
        from collections import OrderedDict
        lru = OrderedDict()
        hits = tot = 0
        for step in trace:
            for L in range(L_all):
                for e in step[L]:
                    key = (L, e)
                    tot += 1
                    if key in lru:
                        lru.move_to_end(key); hits += 1
                    else:
                        lru[key] = True
                        if len(lru) > cap:
                            lru.popitem(last=False)
        return hits / tot

    caps = [250, 500, 750, 1000, 1500, 2000, 3000, 6144]
    hit_curve = {c: sim_lru(c) for c in caps}
    # 每步平均未命中 expert 数
    miss_experts = []
    lru = __import__("collections").OrderedDict()
    for cap_used in (1300,):
        lru.clear()
        per_s = []
        for step in trace:
            m = 0
            for L in range(L_all):
                for e in step[L]:
                    key = (L, e)
                    if key in lru:
                        lru.move_to_end(key)
                    else:
                        m += 1; lru[key] = True
                        if len(lru) > cap_used:
                            lru.popitem(last=False)
            per_s.append(m)
        miss_experts = per_s

    # ---------- C) H2D 微基准 ----------
    # pageable（numpy）vs cudaHostRegister 固定的 mmap 页
    h2d_bench = {}
    nbytes = 32 * 1024 * 1024
    pageable = np.empty(nbytes, dtype=np.uint8)
    gpu = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    for tag, src in [("pageable", pageable)]:
        for _ in range(3):
            gpu.copy_(torch.from_numpy(src), non_blocking=False)
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        reps = 10
        for _ in range(reps):
            gpu.copy_(torch.from_numpy(src), non_blocking=False)
        torch.cuda.synchronize()
        dt = (time.perf_counter_ns() - t0) / 1e9 / reps
        h2d_bench[tag + "_gbps"] = nbytes / dt / 1e9

    # mmap 文件 + hostRegister
    mm_path = "/tmp/h2d_mm.bin"
    with open(mm_path, "wb") as f:
        f.write(np.random.randint(0, 256, nbytes, dtype=np.uint8).tobytes())
    import mmap
    import ctypes
    mm = mmap.mmap(os.open(mm_path, os.O_RDWR), nbytes,
                   prot=mmap.PROT_READ | mmap.PROT_WRITE)
    addr = ctypes.addressof(ctypes.c_char.from_buffer(mm))
    libcudart = ctypes.CDLL("libcudart.so")
    rc = libcudart.cudaHostRegister(ctypes.c_void_p(addr), ctypes.c_size_t(nbytes), 0)
    h2d_bench["hostregister_rc"] = rc
    if rc == 0:
        for _ in range(3):
            gpu.copy_(torch.from_numpy(np.frombuffer(mm, dtype=np.uint8)),
                      non_blocking=False)
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        for _ in range(reps):
            gpu.copy_(torch.from_numpy(np.frombuffer(mm, dtype=np.uint8)),
                      non_blocking=False)
        torch.cuda.synchronize()
        dt = (time.perf_counter_ns() - t0) / 1e9 / reps
        h2d_bench["mmap_pinned_gbps"] = nbytes / dt / 1e9
        # 小块（单 expert 一投影 816KB）
        small = 816 * 1024
        t0 = time.perf_counter_ns()
        reps2 = 200
        for _ in range(reps2):
            gpu[:small].copy_(torch.from_numpy(
                np.frombuffer(mm, dtype=np.uint8)[:small]), non_blocking=False)
        torch.cuda.synchronize()
        dt = (time.perf_counter_ns() - t0) / 1e9 / reps2
        h2d_bench["pinned_816k_gbps"] = small / dt / 1e9

    result = {
        "steps": args.steps,
        "kv_len": kvs[0][0].shape[0],
        "tpot_ms_mean": total_ms / args.steps,
        "stage_ms_per_token": {
            "attn_gpu": buckets["attn"][0] / args.steps,
            "route_incl_tolist": buckets["route"][0] / args.steps,
            "load_wall_total": load_ms / args.steps,
            "load_cpu_read": cpu_ms / args.steps,
            "load_h2d_est": max(load_ms - cpu_ms, 0) / args.steps,
            "expert_kernel_gpu": buckets["compute"][0] / args.steps,
        },
        "live_cache_hit_rate": store.cache_hit_rate(),
        "lru_sim_hit_by_cap": hit_curve,
        "miss_experts_per_step_cap1300": miss_experts,
        "h2d_bench": h2d_bench,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
