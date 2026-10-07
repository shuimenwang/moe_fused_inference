#!/usr/bin/env python3
# chat_demo.py - 交互式 chat demo（Qwen3-30B-A3B MXFP4）
#
# 复用 WeightStore / ModelRunner / fp4_ext，不动核心路径。
# ChatML 模板，KV cache 跨轮增量复用（不重 prefill 历史），token 级流式打印。
#
# KV 增长预算：48 层 × 2 × 4 KV head × 128 dim × bf16 ≈ 196 KB/token；
# 8GB 留 4.8GB 余量 → ~24k token 安全，max_turns×max_tokens 短上下文无忧。

import argparse
import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "runtime"))

from weight_store import WeightStore
from model_runner import ModelRunner


def build_cfg(fp8_dir):
    import json
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


IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


def chatml(role, content):
    return f"{IM_START}{role}\n{content}{IM_END}\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp8", default="/home/hngs/models/Qwen3-30B-A3B-FP8")
    ap.add_argument("--mxfp4", default="/home/hngs/models/Qwen3-30B-A3B-MXFP4")
    ap.add_argument("--variant", default="mxfp4_mma",
                    choices=["mxfp4_mma", "mxfp4", "fp8"])
    ap.add_argument("--system", default="You are a helpful assistant.")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--max-turns", type=int, default=8)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    cfg = build_cfg(args.fp8)
    tokenizer = AutoTokenizer.from_pretrained(args.fp8)
    im_end_id = tokenizer.convert_tokens_to_ids(IM_END)

    print(f"[init] variant={args.variant}，加载静态权重...", flush=True)
    store = WeightStore(args.fp8, args.mxfp4)
    runner = ModelRunner(store, cfg)
    runner._ensure_static()
    torch.cuda.synchronize()
    print(f"[init] 就绪，显存 {torch.cuda.memory_allocated()/1024**2:.0f} MiB；"
          f"输入 /q 退出", flush=True)
    print("=" * 60, flush=True)

    kvs = None
    # system prompt 预填充（不取 logits）
    sys_ids = tokenizer(chatml("system", args.system),
                        add_special_tokens=False).input_ids
    sys_ids = torch.tensor(sys_ids, device="cuda", dtype=torch.long)
    _, kvs, _ = runner.forward(sys_ids, args.variant, past_kvs=kvs,
                               need_lm_head=False)

    for _ in range(args.max_turns):
        try:
            user = input("\n[user] > ")
        except (EOFError, KeyboardInterrupt):
            break
        if user.strip().lower() in {"", "/q", "exit", "quit"}:
            break

        # 增量 prefill：user msg + assistant header，顺带拿首 token logits
        new_text = chatml("user", user) + f"{IM_START}assistant\n"
        new_ids = tokenizer(new_text, add_special_tokens=False).input_ids
        new_ids = torch.tensor(new_ids, device="cuda", dtype=torch.long)
        logits, kvs, _ = runner.forward(new_ids, args.variant, past_kvs=kvs,
                                        need_lm_head=True)

        print("[assistant] > ", end="", flush=True)
        nxt = logits[-1].argmax().reshape(1)
        for _ in range(args.max_tokens):
            tok = int(nxt.item())
            if tok == im_end_id:
                break
            print(tokenizer.decode([tok], skip_special_tokens=True),
                  end="", flush=True)
            logits, kvs, _ = runner.forward(nxt, args.variant, past_kvs=kvs,
                                           need_lm_head=True)
            nxt = logits[-1].argmax().reshape(1)
        print(flush=True)

    print("\n[exit]", flush=True)


if __name__ == "__main__":
    main()
