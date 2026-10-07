#!/usr/bin/env python3
# spec_decode_bench.py - prompt-lookup 投机解码：对齐验证 + 吞吐测量（v2 引擎）

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
from spec_decoder import PromptLookup
from decode_bench import build_cfg

FP8 = "/home/hngs/models/Qwen3-30B-A3B-FP8"
MX = "/home/hngs/models/Qwen3-30B-A3B-MXFP4"
ST = "/home/hngs/models/Qwen3-30B-A3B-MXFP4-static"


def greedy_gen(runner, prompt_t, max_new):
    """单步贪心（forward_batch 解码路径，与投机同源）。"""
    runner.forward_batch(prompt_t, 0)        # prefill（丢弃 logits，只用 KV）
    pos = prompt_t.numel()
    hist = [int(x) for x in prompt_t.tolist()]
    out = []
    for _ in range(max_new):
        lg, _ = runner.forward_batch(
            torch.tensor([hist[-1]], dtype=torch.long, device=prompt_t.device), pos - 1)
        tok = int(lg[0].argmax())
        out.append(tok); hist.append(tok); pos += 1
    return out


def spec_gen(runner, prompt_t, max_new, **kw):
    """投机生成。runner 已 prefill。返回 (token list, 统计)。"""
    lg, _ = runner.forward(prompt_t, 0)
    pos = prompt_t.numel()
    pd = PromptLookup(runner, **kw)
    pd.hist = [int(x) for x in prompt_t.tolist()]
    pd.pos = pos
    pd._rebuild_tailidx()
    stats = {"verify": 0, "accepted": 0}
    while len(pd.hist) - len(prompt_t.tolist()) < max_new:
        prev = len(pd.hist)
        pos = pd._verify(pos)
        stats["verify"] += 1
        stats["accepted"] += len(pd.hist) - prev
    return pd.hist[len(prompt_t.tolist()):], stats


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--cache-cap", type=int, default=1500)
    ap.add_argument("--n-gram", type=int, default=7)
    ap.add_argument("--draft", type=int, default=4)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    cfg = build_cfg(FP8)
    tok = AutoTokenizer.from_pretrained(FP8)
    prompt = ("In a future where artificial minds share the world with us, "
              "the most important lesson is")
    ids = tok(prompt, return_tensors="pt").input_ids[0]
    dev = torch.device("cuda")

    store = WeightStore(FP8, MX, expert_cache_cap=args.cache_cap, static_mxfp4_dir=ST)

    # ---- 对齐验证（小步）：投机 vs 单步贪心 ----
    NALIGN = 16
    rg = ModelRunnerV2(store, cfg); rg.prepare(1024)
    gseq = greedy_gen(rg, ids.to(dev), NALIGN)
    del rg; torch.cuda.empty_cache()

    rs = ModelRunnerV2(store, cfg); rs.prepare(1024)
    sseq, _ = spec_gen(rs, ids.to(dev), NALIGN, n_gram=args.n_gram, draft_len=args.draft)

    agree = sum(1 for a, b in zip(gseq, sseq) if a == b)
    print(f"[align] 投机 vs 贪心 {NALIGN} token：一致率 {agree}/{NALIGN} "
          f"({agree / NALIGN * 100:.0f}%)")
    print(f"  贪心: {tok.decode(gseq)[:80]!r}")
    print(f"  投机: {tok.decode(sseq)[:80]!r}")

    # ---- 吞吐（投机长步）----
    del rs; torch.cuda.empty_cache()
    rt = ModelRunnerV2(store, cfg); rt.prepare(1024)
    lg, _ = rt.forward(ids.to(dev), 0)
    pos = ids.numel()
    pd = PromptLookup(rt, n_gram=args.n_gram, draft_len=args.draft)
    pd.hist = [int(x) for x in ids.tolist()]; pd.pos = pos; pd._rebuild_tailidx()
    sync = torch.cuda.synchronize
    sync(); t0 = time.time()
    nverify = 0
    all_hist0 = len(pd.hist)
    while len(pd.hist) - all_hist0 < args.max_new:
        pos = pd._verify(pos)
        nverify += 1
    sync(); dt = time.time() - t0
    ngen = len(pd.hist) - all_hist0
    # pure greedy 同长对比（单独计时，短步）
    rg2 = ModelRunnerV2(store, cfg); rg2.prepare(1024)
    sync(); t0g = time.time()
    gseq2 = greedy_gen(rg2, ids.to(dev), ngen)
    sync(); dtg = time.time() - t0g

    print("\n" + "=" * 58)
    print(f"投机: {ngen} token / {nverify} 次前向, {dt:.2f}s → {ngen / dt:.1f} tok/s "
          f"(平均 accept {ngen / nverify:.2f}/前向)")
    print(f"贪心: {ngen} token / {ngen} 次前向, {dtg:.2f}s → {ngen / dtg:.1f} tok/s")
    print(f"加速比 ≈ {(ngen / dtg) > 0 and dtg / max(dt, 1e-9):.2f}×")
    print(f"生成(draft={args.draft}, n-gram={args.n_gram}): {tok.decode(pd.hist[all_hist0:])[:120]}")


if __name__ == "__main__":
    main()