#!/usr/bin/env python3
# align_decode_v2.py - v1 vs v2 decode 路径数值对齐
#
# 同一 prompt 贪心生成 N 步，逐 token 比较：
#   argmax 是否一致 / top10 overlap / logits cosine / KL
# v1：bf16 反量化静态权重 + bf16 cat KV + 朴素 fp4 expert kernel
# v2：静态 MXFP4 + FP8 KV + P2.2 grouped kernel（差异源全部叠加）

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "runtime"))

from weight_store import WeightStore
from model_runner import ModelRunner
from model_runner_v2 import ModelRunnerV2
from decode_bench import build_cfg

FP8 = "/home/hngs/models/Qwen3-30B-A3B-FP8"
MX = "/home/hngs/models/Qwen3-30B-A3B-MXFP4"
ST = "/home/hngs/models/Qwen3-30B-A3B-MXFP4-static"


def kl(p, q):
    return float((p * (p.clamp_min(1e-9).log() - q.clamp_min(1e-9).log())).sum())


def stats(a, b):
    p = F.softmax(a.float(), -1)
    q = F.softmax(b.float(), -1)
    cos = F.cosine_similarity(a.float().unsqueeze(0),
                              b.float().unsqueeze(0)).item()
    ta = set(a.topk(10).indices.tolist())
    tb = set(b.topk(10).indices.tolist())
    return cos, len(ta & tb), 0.5 * (kl(p, q) + kl(q, p))


def run_v1(cfg, tok, ids, steps, cap, out_path):
    store = WeightStore(FP8, MX, expert_cache_cap=cap)
    r = ModelRunner(store, cfg)
    logits, kvs, _ = r.forward(ids, "mxfp4_mma")
    nxt = logits[-1].argmax().reshape(1)
    rec = {"prefill": logits[-1].float().cpu().numpy(),
           "toks": [], "logits": []}
    for _ in range(steps):
        rec["toks"].append(int(nxt.item()))
        logits, kvs, _ = r.forward(nxt, "mxfp4_mma", past_kvs=kvs)
        rec["logits"].append(logits[-1].float().cpu().numpy())
        nxt = logits[-1].argmax().reshape(1)
    np.savez(out_path, prefill=rec["prefill"],
             toks=np.array(rec["toks"], np.int64),
             logits=np.stack(rec["logits"]))


def run_v2(cfg, tok, ids, steps, cap, max_len, out_path):
    store = WeightStore(FP8, MX, expert_cache_cap=cap, static_mxfp4_dir=ST)
    r = ModelRunnerV2(store, cfg)
    r.prepare(max_len)
    logits, _ = r.forward(ids, 0)
    pos = ids.numel()
    nxt = logits[-1].argmax().reshape(1)
    rec = {"prefill": logits[-1].float().cpu().numpy(),
           "toks": [], "logits": []}
    for _ in range(steps):
        rec["toks"].append(int(nxt.item()))
        logits, _ = r.forward(nxt, pos)
        rec["logits"].append(logits[-1].float().cpu().numpy())
        nxt = logits[-1].argmax().reshape(1)
        pos += 1
    np.savez(out_path, prefill=rec["prefill"],
             toks=np.array(rec["toks"], np.int64),
             logits=np.stack(rec["logits"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--cache-cap", type=int, default=1500)
    ap.add_argument("--max-len", type=int, default=4096)
    args = ap.parse_args()

    import tempfile
    from transformers import AutoTokenizer
    cfg = build_cfg(FP8)
    tok = AutoTokenizer.from_pretrained(FP8)
    prompt = ("In a future where artificial minds share the world with us, "
              "the most important lesson is")
    ids = tok(prompt, return_tensors="pt").input_ids[0].cuda()

    d = tempfile.mkdtemp(prefix="align_")
    p1, p2 = os.path.join(d, "v1.npz"), os.path.join(d, "v2.npz")

    print("[1/2] v1 生成中…")
    run_v1(cfg, tok, ids, args.steps, args.cache_cap, p1)
    torch.cuda.empty_cache()
    print("[2/2] v2 生成中…")
    run_v2(cfg, tok, ids, args.steps, args.cache_cap, args.max_len, p2)

    a, b = np.load(p1), np.load(p2)
    cos, ov, kld = stats(torch.from_numpy(a["prefill"]).cuda(),
                         torch.from_numpy(b["prefill"]).cuda())
    eq0 = int(a["toks"][0] == b["toks"][0])
    print(f"prefill  next_tok_eq={eq0} cos={cos:.5f} "
          f"top10={ov}/10 KL={kld:.4f}")
    agree = 0
    for i in range(args.steps):
        cos, ov, kld = stats(torch.from_numpy(a["logits"][i]).cuda(),
                             torch.from_numpy(b["logits"][i]).cuda())
        ta, tb = int(a["toks"][i]), int(b["toks"][i])
        eq = int(ta == tb)
        agree += eq
        print(f"decode{i:02d} argmax_eq={eq} cos={cos:.5f} "
              f"top10={ov}/10 KL={kld:.4f} "
              f"t1={tok.decode([ta])!r} t2={tok.decode([tb])!r}")
    print(f"\ntoken 一致率 {agree}/{args.steps}")


if __name__ == "__main__":
    main()
