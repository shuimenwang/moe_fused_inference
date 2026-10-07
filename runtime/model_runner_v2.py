#!/usr/bin/env python3
# model_runner_v2.py - P2 高性能 decode 路径
#
# 与 v1 (model_runner.py) 的差异：
#   1. 静态权重（embed/lm_head/attn/router）MXFP4 直消费，GPU 常驻 2.94GB→0.76GB
#      （embed 行 gather 后反量化；其余走自研 mxfp4 MMA kernel）
#   2. KV cache 预分配 + FP8 K/V 存储（48KB/token，32K≈1.5GB），不再每步 torch.cat
#   3. GQA 直接走 SDPA enable_gqa（v1 repeat_interleave 复制 8×）
#   4. MoE 供给与 expert kernel 保留 v1 语义（P2.1c/P2.2 分步替换）
#
# 调用方式：runner.prepare(max_len)；forward(ids, start_pos)；调用方维护 start_pos。

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_runner import rms_norm, apply_rope
from weight_store import dequant_rows_gpu


class ModelRunnerV2:
    def __init__(self, store, cfg):
        self.store = store
        self.cfg = cfg
        self.fp4w = None     # 静态 MXFP4 spec dict
        self.bf16 = None     # 1D norm 等
        self.fp4 = None      # 自研 MXFP4 MMA 扩展（懒加载）
        self.kv = None       # (k_buf, v_buf) fp8 [L,S,Hkv,D]
        self.max_len = 0

    def prepare(self, max_len):
        fp4w, bf16 = self.store.load_static_v2()
        self.fp4w, self.bf16 = fp4w, bf16
        c = self.cfg
        L, Hkv, D = c["num_hidden_layers"], c["num_key_value_heads"], c["head_dim"]
        # FP8 K/V 预分配（e4m3，无 per-tensor scale；激活范围远小于 448）
        self.kv = (
            torch.empty(L, max_len, Hkv, D, dtype=torch.float8_e4m3fn,
                        device="cuda"),
            torch.empty(L, max_len, Hkv, D, dtype=torch.float8_e4m3fn,
                        device="cuda"),
        )
        self.max_len = max_len

    def _get_ext(self):
        if self.fp4 is None:
            from fp4_ext import get_ext
            self.fp4 = get_ext()
        return self.fp4

    def _get_dext(self):
        """P2.2 decode 专用 kernel（prequant/lin_pq/grouped MoE）。"""
        if getattr(self, "fp4d", None) is None:
            from fp4_ext import get_decode_ext
            self.fp4d = get_decode_ext()
        return self.fp4d

    def _lin(self, x, key):
        """静态 MXFP4 线性层：f32 输出转 bf16。"""
        spec = self.fp4w[key]
        return self._get_ext().forward(x, spec["wp"], spec["wsc"]).to(x.dtype)

    # ----------------------------------------------------------------
    def attention(self, h, L, start_pos, T):
        c = self.cfg
        pre = f"model.layers.{L}.self_attn"

        q = self._lin(h, f"{pre}.q_proj.weight").view(T, c["num_attention_heads"],
                                                      c["head_dim"])
        k = self._lin(h, f"{pre}.k_proj.weight").view(T, c["num_key_value_heads"],
                                                      c["head_dim"])
        v = self._lin(h, f"{pre}.v_proj.weight").view(T, c["num_key_value_heads"],
                                                      c["head_dim"])

        q = rms_norm(q, self.bf16[f"{pre}.q_norm.weight"], c["rms_norm_eps"])
        k = rms_norm(k, self.bf16[f"{pre}.k_norm.weight"], c["rms_norm_eps"])

        positions = torch.arange(start_pos, start_pos + T, device=h.device)
        q, k = apply_rope(q, k, positions, c["rope_theta"])

        # 写入预分配 FP8 KV
        self.kv[0][L, start_pos:start_pos + T] = k.to(torch.float8_e4m3fn)
        self.kv[1][L, start_pos:start_pos + T] = v.to(torch.float8_e4m3fn)
        S = start_pos + T

        # GQA：直接以 4 KV heads 调 SDPA（内部广播，无 repeat_interleave）
        qq = q.transpose(0, 1).unsqueeze(0)                          # (1,Hq,T,D)
        kk = self.kv[0][L, :S].to(torch.bfloat16).transpose(0, 1).unsqueeze(0)
        vv = self.kv[1][L, :S].to(torch.bfloat16).transpose(0, 1).unsqueeze(0)
        is_causal = start_pos == 0 and T > 1
        out = F.scaled_dot_product_attention(
            qq, kk, vv, is_causal=is_causal, dropout_p=0.0,
            enable_gqa=True,
        ).squeeze(0).transpose(0, 1).reshape(T, -1)
        return self._lin(out, f"{pre}.o_proj.weight")

    def moe_route(self, h, L):
        logits = self._lin(h, f"model.layers.{L}.mlp.gate.weight").float()
        router_probs = F.softmax(logits, dim=-1)
        topv, topi = torch.topk(router_probs, self.cfg["num_experts_per_tok"], dim=-1)
        if self.cfg["norm_topk_prob"]:
            topv = topv / topv.sum(dim=-1, keepdim=True)
        return logits, topi, topv.to(h.dtype)

    def moe_compute(self, h, topi, scores, experts):
        out = torch.zeros_like(h)
        for e in experts:
            idx, slots = torch.where(topi == e)
            if idx.numel() == 0:
                continue
            xt = h[idx]
            wg, wu, wd = experts[e]["gate"], experts[e]["up"], experts[e]["down"]
            ext = self._get_ext()
            g = ext.forward(xt, wg["wp"], wg["wsc"])
            u = ext.forward(xt, wu["wp"], wu["wsc"])
            hh = F.silu(g) * u
            y = ext.forward(hh, wd["wp"], wd["wsc"]).to(out.dtype)
            out.index_add_(0, idx, y * scores[idx, slots].unsqueeze(-1))
        return out

    # ----------------------------------------------------------------
    def _d_lin(self, dext, xq, xsc, key, M, N, K):
        """decode kernel 静态投影，f32 输出。"""
        spec = self.fp4w[key]
        return dext.lin_pq(xq, xsc, spec["wp"], spec["wsc"], M, N, K)

    def forward_decode(self, input_ids, start_pos, need_lm_head=True):
        """T==1 快路径：prequant 外提 + lin_pq 静态投影 + grouped MoE（每层 5 launch）。"""
        c = self.cfg
        dext = self._get_dext()
        H = c["hidden_size"]
        Hq, Hkv, D = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        E = c["num_experts_per_tok"]
        Mi = c["moe_intermediate_size"]      # gate/up N=768, K=H=2048
        Mk = H                               # down N=H, K=Mi=768

        emb = self.fp4w["model.embed_tokens.weight"]
        x = dequant_rows_gpu(emb["wp"], emb["wsc"], emb["K"], input_ids)  # (1,H) bf16
        routing = []

        for L in range(c["num_hidden_layers"]):
            pre = f"model.layers.{L}"
            sap = f"{pre}.self_attn"

            # -------- attention --------
            hp = rms_norm(x, self.bf16[f"{pre}.input_layernorm.weight"],
                          c["rms_norm_eps"])
            hpq, hps = dext.prequant(hp.contiguous())
            q = self._d_lin(dext, hpq, hps, f"{sap}.q_proj.weight", 1, Hq * D, H)
            k = self._d_lin(dext, hpq, hps, f"{sap}.k_proj.weight", 1, Hkv * D, H)
            v = self._d_lin(dext, hpq, hps, f"{sap}.v_proj.weight", 1, Hkv * D, H)
            q = rms_norm(q.view(Hq, D), self.bf16[f"{sap}.q_norm.weight"],
                         c["rms_norm_eps"]).bfloat16()
            k = rms_norm(k.view(Hkv, D), self.bf16[f"{sap}.k_norm.weight"],
                         c["rms_norm_eps"]).bfloat16()
            v = v.view(Hkv, D).bfloat16()

            positions = torch.tensor([start_pos], device=x.device)
            q, k = apply_rope(q.unsqueeze(0), k.unsqueeze(0), positions,
                              c["rope_theta"])
            q, k = q.squeeze(0), k.squeeze(0)
            self.kv[0][L, start_pos] = k.to(torch.float8_e4m3fn)
            self.kv[1][L, start_pos] = v.to(torch.float8_e4m3fn)
            S = start_pos + 1

            qq = q.unsqueeze(1).unsqueeze(0)                   # (1,Hq,1,D)
            kk = self.kv[0][L, :S].to(torch.bfloat16) \
                .permute(1, 0, 2).unsqueeze(0)                 # (1,Hkv,S,D)
            vv = self.kv[1][L, :S].to(torch.bfloat16) \
                .permute(1, 0, 2).unsqueeze(0)
            at = F.scaled_dot_product_attention(
                qq, kk, vv, is_causal=False, dropout_p=0.0, enable_gqa=True,
            ).reshape(1, Hq * D).bfloat16()
            aq, asc = dext.prequant(at)
            o = self._d_lin(dext, aq, asc, f"{sap}.o_proj.weight",
                            1, H, Hq * D)
            x = x + o.bfloat16()

            # -------- MoE --------
            hp2 = rms_norm(x, self.bf16[f"{pre}.post_attention_layernorm.weight"],
                           c["rms_norm_eps"])
            hp2q, hp2s = dext.prequant(hp2.contiguous())
            gate_logits = self._d_lin(dext, hp2q, hp2s,
                                      f"model.layers.{L}.mlp.gate.weight",
                                      1, c["num_experts"], H)
            router_probs = F.softmax(gate_logits, dim=-1)
            topv, topi = torch.topk(router_probs, E, dim=-1)
            if c["norm_topk_prob"]:
                topv = topv / topv.sum(dim=-1, keepdim=True)

            ids = topi.view(-1).tolist()
            active = set(ids)
            experts = self.store.load_layer_experts(L, "mxfp4_mma", active=active)
            gwp = [experts[e]["gate"]["wp"] for e in ids]
            gsc = [experts[e]["gate"]["wsc"] for e in ids]
            uwp = [experts[e]["up"]["wp"] for e in ids]
            usc = [experts[e]["up"]["wsc"] for e in ids]
            dwp = [experts[e]["down"]["wp"] for e in ids]
            dsc = [experts[e]["down"]["wsc"] for e in ids]

            gu = dext.moe_gateup(hp2q, hp2s, gwp + uwp, gsc + usc,
                                 E, 1, Mi, Mk)
            yd = dext.moe_down(gu, dwp, dsc, E, 1, H, Mi)
            moe = dext.moe_combine(yd, topv.view(1, E).float(), 1)[0]   # (H,) f32
            x = x + moe.bfloat16().unsqueeze(0)
            routing.append((topi, gate_logits))

        if need_lm_head:
            x = rms_norm(x, self.bf16["model.norm.weight"], c["rms_norm_eps"])
            xq, xsc = dext.prequant(x.contiguous())
            logits = self._d_lin(dext, xq, xsc, "lm_head.weight",
                                 1, c["vocab_size"], H)
            return logits, routing
        return None, routing

    # ----------------------------------------------------------------
    def forward_batch(self, input_ids, base_pos, need_lm_head=True, kv=None):
        """投机验证：M 个 token 批量 decode（逐行与 forward_decode 一致）。
        input_ids: (M,) cuda；写 KV[base_pos..base_pos+M)；返回 (M,V) f32 logits。
        logits[t] 预测输入完 position(base_pos+t)（= 下一 token）。"""
        c = self.cfg
        dext = self._get_dext()
        H = c["hidden_size"]
        Hq, Hkv, D = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        E = c["num_experts_per_tok"]
        Mi = c["moe_intermediate_size"]
        M = input_ids.numel()
        dev = input_ids.device
        lk = self.kv if kv is None else kv
        kvk, kvv = lk[0], lk[1]

        emb = self.fp4w["model.embed_tokens.weight"]
        x = dequant_rows_gpu(emb["wp"], emb["wsc"], emb["K"], input_ids)  # (M,H)
        dpos = input_ids.new_zeros(1, dtype=torch.int32).fill_(base_pos)

        positions = torch.arange(base_pos, base_pos + M, device=dev)
        for L in range(c["num_hidden_layers"]):
            pre = f"model.layers.{L}"
            sap = f"{pre}.self_attn"
            hp = rms_norm(x, self.bf16[f"{pre}.input_layernorm.weight"], c["rms_norm_eps"])
            hpq, hps = dext.prequant(hp.contiguous())
            q = self._lin2(dext, hpq, hps, f"{sap}.q_proj.weight", M, Hq*D, H)
            k = self._lin2(dext, hpq, hps, f"{sap}.k_proj.weight", M, Hkv*D, H)
            v = self._lin2(dext, hpq, hps, f"{sap}.v_proj.weight", M, Hkv*D, H)
            qn = rms_norm(q.view(M, Hq, D), self.bf16[f"{sap}.q_norm.weight"], c["rms_norm_eps"]).bfloat16()
            kn = rms_norm(k.view(M, Hkv, D), self.bf16[f"{sap}.k_norm.weight"], c["rms_norm_eps"]).bfloat16()
            vn = v.view(M, Hkv, D).bfloat16()
            # rope per row（位置 base_pos+m）
            qr, _ = apply_rope(qn, qn, positions, c["rope_theta"])
            _, kr = apply_rope(kn, kn, positions, c["rope_theta"])
            kr = kr.reshape(M, Hkv, D)
            qr = qr.reshape(M, Hq, D)
            kvk[L, base_pos:base_pos + M] = kr.to(torch.float8_e4m3fn)
            kvv[L, base_pos:base_pos + M] = vn.to(torch.float8_e4m3fn)
            # attention（每个 token attend 到自身位置）
            sc_b = torch.empty(M, Hq, kvk.size(1), device=dev, dtype=torch.float32)
            oa = dext.attn_decode(qr, kvk[L].view(torch.uint8), kvv[L].view(torch.uint8), dpos, sc_b, M)
            of = oa.reshape(M, Hq*D).to(torch.bfloat16).contiguous()
            po = dext.prequant(of); aq = po[0]; asc = po[1]
            o = self._lin2(dext, aq, asc, f"{sap}.o_proj.weight", M, H, Hq*D)
            x = x + o.bfloat16()

            # post-ln + gate
            h2 = rms_norm(x, self.bf16[f"{pre}.post_attention_layernorm.weight"], c["rms_norm_eps"])
            h2q, h2s = dext.prequant(h2.contiguous())
            gate = self._lin2(dext, h2q, h2s, f"{pre}.mlp.gate.weight", M, c["num_experts"], H)

            # host 路由：每行 topk，token-major 指针表
            rp = F.softmax(gate.float(), -1)
            tv, ti = torch.topk(rp, E, dim=-1)
            if c["norm_topk_prob"]:
                tv = tv / tv.sum(-1, keepdim=True)
            ids = ti.reshape(-1).tolist()
            uni = set(ids)
            ex = self.store.load_layer_experts(L, "mxfp4_mma", active=uni)
            gu_tab, gsu_tab, dw_tab, ds_tab = [], [], [], []
            for m in range(M):
                gu_tab += [ex[ids[m*E+e]]["gate"]["wp"] for e in range(E)]
                gsu_tab += [ex[ids[m*E+e]]["gate"]["wsc"] for e in range(E)]
                gu_tab += [ex[ids[m*E+e]]["up"]["wp"] for e in range(E)]
                gsu_tab += [ex[ids[m*E+e]]["up"]["wsc"] for e in range(E)]
                dw_tab += [ex[ids[m*E+e]]["down"]["wp"] for e in range(E)]
                ds_tab += [ex[ids[m*E+e]]["down"]["wsc"] for e in range(E)]
            guv = dext.moe_gateup(h2q, h2s, gu_tab, gsu_tab, E, M, Mi, H)
            ydv = dext.moe_down(guv, dw_tab, ds_tab, E, M, H, Mi)
            moe = dext.moe_combine(ydv, tv.reshape(M, E).float(), M)  # (M,H)
            x = x + moe.bfloat16()

        if need_lm_head:
            lxx = rms_norm(x, self.bf16["model.norm.weight"], c["rms_norm_eps"])
            lq, ls = dext.prequant(lxx.contiguous())
            logits = self._lin2(dext, lq, ls, "lm_head.weight", M, c["vocab_size"], H)
            return logits, []
        return None, []

    def _lin2(self, dext, xq, xsc, key, M, N, K):
        spec = self.fp4w[key]
        return dext.lin_pq(xq, xsc, spec["wp"], spec["wsc"], M, N, K)

    # ----------------------------------------------------------------
    def forward(self, input_ids, start_pos, need_lm_head=True):
        """input_ids: (T,) cuda；写入 KV [start_pos, start_pos+T)。
        T==1 走 P2.2 decode kernel；T>1（prefill）走 P2.1 朴素路径。
        返回 (logits(T,V) f32 or None, routing)。"""
        if input_ids.numel() == 1:
            return self.forward_decode(input_ids, start_pos, need_lm_head)
        return self.forward_prefill(input_ids, start_pos, need_lm_head)

    def forward_prefill(self, input_ids, start_pos, need_lm_head=True):
        """input_ids: (T,) cuda；写入 KV [start_pos, start_pos+T)。
        返回 (logits(T,V) f32 or None, routing)。"""
        c = self.cfg
        assert start_pos + input_ids.numel() <= self.max_len
        T = input_ids.numel()

        emb = self.fp4w["model.embed_tokens.weight"]
        x = dequant_rows_gpu(emb["wp"], emb["wsc"], emb["K"], input_ids)
        routing = []

        for L in range(c["num_hidden_layers"]):
            pre = f"model.layers.{L}"
            hp = rms_norm(x, self.bf16[f"{pre}.input_layernorm.weight"],
                          c["rms_norm_eps"])
            x = x + self.attention(hp, L, start_pos, T)

            hp2 = rms_norm(x, self.bf16[f"{pre}.post_attention_layernorm.weight"],
                           c["rms_norm_eps"])
            gate_logits, topi, scores = self.moe_route(hp2, L)
            active = set(topi.flatten().tolist())
            experts = self.store.load_layer_experts(L, "mxfp4_mma", active=active)
            x = x + self.moe_compute(hp2, topi, scores, experts)
            routing.append((topi, gate_logits))

        if need_lm_head:
            x = rms_norm(x, self.bf16["model.norm.weight"], c["rms_norm_eps"])
            logits = self._lin(x, "lm_head.weight").float()
            return logits, routing
        return None, routing
