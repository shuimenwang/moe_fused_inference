#!/usr/bin/env python3
# model_runner.py - Qwen3-MoE 逐层前向（正确性优先；同步加载，无 overlap）
#
# 架构（config 实测）：
#   RMSNorm → GQA(32Q/4KV, head_dim=128) + QK-RMSNorm + RoPE(theta=1e6)
#   → RMSNorm → MoE(128, top-8, softmax norm)；48 层
#
# 专家权重逐 layer 读取（算完即释放），非专家权重 GPU 常驻。
# past_kv 支持 decode：list[L] of (K, V)，新 token 追加。

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def rms_norm(x, weight, eps=1e-6):
    dtype = x.dtype
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    xf = xf * torch.rsqrt(var + eps)
    return (xf.to(dtype)) * weight


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, positions, theta):
    """q (T,H,D), k (T,Hkv,D)；half-layout rope。"""
    D = q.shape[-1]
    inv_freq = 1.0 / (theta ** (torch.arange(0, D, 2, device=q.device).float() / D))
    freqs = positions.float()[:, None] * inv_freq[None, :]       # (T, D/2) 弧度
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)[:, None]  # (T,1,D)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)[:, None]
    q = (q * cos + rotate_half(q) * sin).to(q.dtype)
    k = (k * cos + rotate_half(k) * sin).to(k.dtype)
    return q, k


class ModelRunner:
    def __init__(self, store, cfg):
        self.store = store
        self.cfg = cfg
        self.static = None       # 第一次 forward 加载
        self.fp4 = None          # 自研 MXFP4 MMA 扩展（懒加载）

    def _ensure_static(self):
        if self.static is None:
            self.static = self.store.load_static_weights()

    def attention(self, h, L, past_kv):
        c = self.cfg
        sw = self.static
        pre = f"model.layers.{L}.self_attn"

        T = h.shape[0]
        past_len = 0 if past_kv is None else past_kv[0].shape[0]

        def proj(name, heads):
            w = sw[f"{pre}.{name}_proj.weight"]
            return F.linear(h, w).view(T, heads, c["head_dim"])

        q = proj("q", c["num_attention_heads"])
        k = proj("k", c["num_key_value_heads"])
        v = proj("v", c["num_key_value_heads"])

        # QK-norm（RMSNorm per head，权重 head_dim）
        q = rms_norm(q, sw[f"{pre}.q_norm.weight"], c["rms_norm_eps"])
        k = rms_norm(k, sw[f"{pre}.k_norm.weight"], c["rms_norm_eps"])

        positions = torch.arange(past_len, past_len + T, device=h.device)
        q, k = apply_rope(q, k, positions, c["rope_theta"])

        # 追加历史 KV
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=0)
            v = torch.cat([past_kv[1], v], dim=0)
        new_kv = (k, v)

        # GQA：KV 重复到 Q heads
        rep = c["num_attention_heads"] // c["num_key_value_heads"]
        kk = k.repeat_interleave(rep, dim=1).transpose(0, 1)
        vv = v.repeat_interleave(rep, dim=1).transpose(0, 1)
        qq = q.transpose(0, 1)

        is_causal = past_len == 0
        qq = q.transpose(0, 1).unsqueeze(0)              # (1,H,T,D)
        out = F.scaled_dot_product_attention(
            qq, kk.unsqueeze(0), vv.unsqueeze(0),
            is_causal=is_causal, dropout_p=0.0,
        ).squeeze(0).transpose(0, 1).reshape(T, -1)
        return F.linear(out, sw[f"{pre}.o_proj.weight"]), new_kv

    def moe_route(self, h, L):
        gate_w = self.static[f"model.layers.{L}.mlp.gate.weight"]
        logits = F.linear(h, gate_w)
        # 官方 Qwen3 router：先对全部专家 softmax，再取 top-k 概率，最后归一化 top-k
        router_probs = F.softmax(logits.float(), dim=-1)
        topv, topi = torch.topk(router_probs, self.cfg["num_experts_per_tok"], dim=-1)
        if self.cfg["norm_topk_prob"]:
            topv = topv / topv.sum(dim=-1, keepdim=True)
        scores = topv.to(logits.dtype)
        return logits, topi, scores

    def moe_compute(self, h, topi, scores, experts, variant="fp8"):
        out = torch.zeros_like(h)
        for e in experts:
            idx, slots = torch.where(topi == e)
            if idx.numel() == 0:
                continue
            xt = h[idx]
            wg, wu, wd = experts[e]["gate"], experts[e]["up"], experts[e]["down"]
            if variant == "mxfp4_mma":
                # 真实 MXFP4 block-scale MMA：gate/up/down 都走自研 kernel（f32 输出）
                if self.fp4 is None:
                    from fp4_ext import get_ext
                    self.fp4 = get_ext()
                g = self.fp4.forward(xt, wg["wp"], wg["wsc"])
                u = self.fp4.forward(xt, wu["wp"], wu["wsc"])
                hh = F.silu(g) * u
                y = self.fp4.forward(hh, wd["wp"], wd["wsc"])
                y = y.to(out.dtype)
            else:
                hh = F.silu(F.linear(xt, wg)) * F.linear(xt, wu)
                y = F.linear(hh, wd)
            out.index_add_(0, idx, y * scores[idx, slots].unsqueeze(-1))
        return out

    def forward(self, input_ids, variant, past_kvs=None, need_lm_head=True):
        """返回 (logits or None, new_kvs, routing)。
        routing: list[L] of (topi (T,8), gate_logits (T,128))。"""
        self._ensure_static()
        c = self.cfg
        x = F.embedding(input_ids, self.static["model.embed_tokens.weight"])
        new_kvs, routing = [], []

        for L in range(c["num_hidden_layers"]):
            pre = f"model.layers.{L}"
            hp = rms_norm(x, self.static[f"{pre}.input_layernorm.weight"],
                          c["rms_norm_eps"])
            attn_out, new_kv = self.attention(
                hp, L, None if past_kvs is None else past_kvs[L])
            x = x + attn_out

            hp2 = rms_norm(x, self.static[f"{pre}.post_attention_layernorm.weight"],
                           c["rms_norm_eps"])
            gate_logits, topi, scores = self.moe_route(hp2, L)
            active = set(topi.flatten().tolist())
            experts = self.store.load_layer_experts(L, variant, active=active)
            moe_out = self.moe_compute(hp2, topi, scores, experts, variant)
            routing.append((topi, gate_logits))
            del experts
            x = x + moe_out
            new_kvs.append(new_kv)

        if need_lm_head:
            x = rms_norm(x, self.static["model.norm.weight"], c["rms_norm_eps"])
            logits = F.linear(x, self.static["lm_head.weight"]).float()
            return logits, new_kvs, routing
        return None, new_kvs, routing
