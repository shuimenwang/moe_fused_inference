#!/usr/bin/env python3
# model_runner_v3.py - P3 CUDA Graph 化 decode 路径
#
# 目标：消除 P2.2 每步 ~1万 kernel 的 launch/scheduling 缝隙
# （profiler: 纯 GPU 计算仅 ~27.6ms，wall ~92ms）。
#
# 分段图（host 依赖 topk 选 expert，故每层 2 图）：
#   graphA[L] : x+attn（lin q/k/v→rope→decode-attn→o lin）+ post-ln→h2 prequant→gate lin
#   host      : 读 gate→softmax/topk→8 id；更新该层指针表 & topv
#   graphB[L] : h2q/h2s+指针表 → moe_gateup/down/combine → x+=moe
#   graphFin  : 输出层 norm + lm_head
# 动态量用"内容随 replay 更新的设备 buffer"：embedding、dpos、positions、kv 写索引、
#   每层指针表(int64)、topv。数值须与 model_runner_v2.forward_decode 一致。

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_runner import rms_norm, apply_rope


class ModelRunnerV3:
    def __init__(self, store, cfg):
        self.store = store
        self.cfg = cfg
        self.fp4w = None
        self.bf16 = None
        self.fp4d = None
        self.max_len = 0
        self._cap = False

    # ---------------------------------------------------------------
    def _dext(self):
        if self.fp4d is None:
            from fp4_ext import get_decode_ext
            self.fp4d = get_decode_ext()
        return self.fp4d

    def _dlin(self, dext, xq, xsc, key, M, N, K):
        spec = self.fp4w[key]
        return dext.lin_pq(xq, xsc, spec["wp"], spec["wsc"], M, N, K)

    def prepare(self, max_len):
        fp4w, bf16 = self.store.load_static_v2()
        self.fp4w, self.bf16 = fp4w, bf16
        c = self.cfg
        self.max_len = max_len
        H = c["hidden_size"]
        Hq, Hkv, D = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        self.H, self.Hq, self.Hkv, self.D = H, Hq, Hkv, D
        self.E = c["num_experts_per_tok"]
        self.Mi = c["moe_intermediate_size"]
        self.L = c["num_hidden_layers"]
        Kpp = (H + 31) // 32 * 32
        f8 = torch.float8_e4m3fn
        dev = torch.device("cuda")
        bf = torch.bfloat16

        self.kv_k = torch.empty(self.L, max_len, Hkv, D, dtype=f8, device=dev)
        self.kv_v = torch.empty(self.L, max_len, Hkv, D, dtype=f8, device=dev)

        z = lambda *s, **k: torch.zeros(*s, dtype=bf, device=dev, **k)
        zf = lambda *s: torch.zeros(*s, dtype=torch.float32, device=dev)
        zu = lambda *s: torch.zeros(*s, dtype=torch.uint8, device=dev)
        self.xb, self.hp = z(1, H), z(1, H)
        self.h2 = z(1, H)
        Kpo = (Hq * D + 31) // 32 * 32
        self.hpq, self.hps = zu(1, Kpp // 2), zu(1, Kpp // 32)
        self.h2q, self.h2s = zu(1, Kpp // 2), zu(1, Kpp // 32)
        self.aq_o, self.asc_o = zu(1, Kpo // 2), zu(1, Kpo // 32)
        self.lxq, self.lxs = zu(1, Kpp // 2), zu(1, Kpp // 32)
        self.q, self.qk, self.v = zf(1, Hq * D), zf(1, Hkv * D), zf(1, Hkv * D)
        self.q32, self.k32, self.v32 = z(Hq, D), z(Hkv, D), z(Hkv, D)
        self.o32 = zf(Hq, D)
        self.ao = zf(1, H)
        self.gate = zf(1, c["num_experts"])
        self.gu = zf(self.E, 2 * self.Mi)
        self.yd = zf(self.E, H)
        self.moe = zf(H)
        self.topv = zf(self.E)
        self.sc = torch.zeros(Hq, max_len, dtype=torch.float32, device=dev)
        self.llogits = zf(1, c["vocab_size"])

        self.g_wpt = torch.zeros(2 * self.E, dtype=torch.int64, device=dev)
        self.g_wst = torch.zeros(2 * self.E, dtype=torch.int64, device=dev)
        self.d_wpt = torch.zeros(self.E, dtype=torch.int64, device=dev)
        self.d_wst = torch.zeros(self.E, dtype=torch.int64, device=dev)
        self.dpos = torch.zeros(1, dtype=torch.int32, device=dev)
        self.kv_idx = torch.zeros(1, dtype=torch.int64, device=dev)
        self.positions = torch.zeros(1, dtype=torch.int64, device=dev)
        self.lx = z(1, H)

    # ---------------------------------------------------------------
    def _kv_write(self, L, dext):
        self.fp4d.kv_write(self.k32, self.v32,
                           self.kv_k[L].view(torch.uint8), self.kv_v[L].view(torch.uint8),
                           self.dpos)

    def _layer_attn(self, L, dext):
        c = self.cfg
        pre = f"model.layers.{L}"
        sap = f"{pre}.self_attn"
        hp = rms_norm(self.xb, self.bf16[f"{pre}.input_layernorm.weight"], c["rms_norm_eps"])
        self.hp.copy_(hp)
        p = dext.prequant(self.hp.contiguous()); self.hpq.copy_(p[0]); self.hps.copy_(p[1])
        self.q.copy_(self._dlin(dext, self.hpq, self.hps, f"{sap}.q_proj.weight",
                                1, self.Hq * self.D, c["hidden_size"]))
        self.qk.copy_(self._dlin(dext, self.hpq, self.hps, f"{sap}.k_proj.weight",
                                 1, self.Hkv * self.D, c["hidden_size"]))
        self.v.copy_(self._dlin(dext, self.hpq, self.hps, f"{sap}.v_proj.weight",
                                1, self.Hkv * self.D, c["hidden_size"]))
        qq = rms_norm(self.q.view(self.Hq, self.D), self.bf16[f"{sap}.q_norm.weight"],
                      c["rms_norm_eps"]).bfloat16()
        kk = rms_norm(self.qk.view(self.Hkv, self.D), self.bf16[f"{sap}.k_norm.weight"],
                      c["rms_norm_eps"]).bfloat16()
        vv = self.v.view(self.Hkv, self.D).bfloat16()
        qr, kr = apply_rope(qq.unsqueeze(0), kk.unsqueeze(0), self.positions, c["rope_theta"])
        self.q32.copy_(qr.squeeze(0))
        self.k32.copy_(kr.squeeze(0))
        self.v32.copy_(vv)
        self._kv_write(L, dext)
        self.o32.copy_(dext.attn_decode(self.q32,
                                              self.kv_k[L].view(torch.uint8),
                                              self.kv_v[L].view(torch.uint8),
                                              self.dpos, self.sc))
        # o 投影
        of = self.o32.reshape(1, self.Hq * self.D).to(torch.bfloat16).contiguous()
        po = dext.prequant(of); self.aq_o.copy_(po[0]); self.asc_o.copy_(po[1])
        self.ao.copy_(self._dlin(dext, self.aq_o, self.asc_o, f"{sap}.o_proj.weight",
                                 1, c["hidden_size"], self.Hq * self.D))
        self.xb.add_(self.ao.to(torch.bfloat16))
        # post-ln + h2 prequant + gate
        self.h2.copy_(rms_norm(self.xb, self.bf16[f"{pre}.post_attention_layernorm.weight"],
                               c["rms_norm_eps"]))
        ph = dext.prequant(self.h2.contiguous()); self.h2q.copy_(ph[0]); self.h2s.copy_(ph[1])
        self.gate.copy_(self._dlin(dext, self.h2q, self.h2s, f"{pre}.mlp.gate.weight",
                                   1, c["num_experts"], c["hidden_size"]))

    def _layer_moe(self, dext):
        self.gu.copy_(dext.moe_gateup_tab(self.h2q, self.h2s, self.g_wpt, self.g_wst,
                                          self.E, 1, self.Mi, self.H))
        self.yd.copy_(dext.moe_down_tab(self.gu, self.d_wpt, self.d_wst, self.E, self.H, self.Mi))
        self.moe.copy_(dext.moe_combine(self.yd, self.topv))
        self.xb.add_(self.moe.view(1, self.H).to(torch.bfloat16))

    def _final(self, dext):
        c = self.cfg
        self.lx.copy_(rms_norm(self.xb, self.bf16["model.norm.weight"], c["rms_norm_eps"]))
        pl = dext.prequant(self.lx.contiguous()); self.lxq.copy_(pl[0]); self.lxs.copy_(pl[1])
        self.llogits.copy_(self._dlin(dext, self.lxq, self.lxs, "lm_head.weight",
                                      1, c["vocab_size"], c["hidden_size"]))

    # ---------------------------------------------------------------
    def build(self):
        dext = self._dext()
        # 一次性在默认流上执行一遍，确保所有 kernel/分配就绪后再 capture
        pool = torch.cuda.graph_pool_handle()
        # 预热一帧避免 capture 中编译
        self.graphsA, self.graphsB = [], []
        for L in range(self.L):
            ga = torch.cuda.CUDAGraph()
            with torch.cuda.graph(ga, pool=pool):
                self._layer_attn(L, dext)
            gb = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gb, pool=pool):
                self._layer_moe(dext)
            self.graphsA.append(ga)
            self.graphsB.append(gb)
        gfin = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gfin, pool=pool):
            self._final(dext)
        self.graphFin = gfin
        self._cap = True

    # ---------------------------------------------------------------
    def forward(self, input_ids, start_pos, need_lm_head=True):
        """T==1 decode。input_ids: (1,) cuda。返回 (logits 或 None, routing)。"""
        dext = self._dext()
        if not self._cap:
            self.build()
            torch.cuda.synchronize()

        from weight_store import dequant_rows_gpu
        emb = self.fp4w["model.embed_tokens.weight"]
        row = dequant_rows_gpu(emb["wp"], emb["wsc"], emb["K"], input_ids)
        self.xb.copy_(row)

        dev = self.xb.device
        self.dpos.copy_(torch.tensor([start_pos], dtype=torch.int32, device=dev))
        self.kv_idx.copy_(torch.tensor([start_pos], dtype=torch.int64, device=dev))
        self.positions.copy_(torch.tensor([start_pos], dtype=torch.int64, device=dev))

        for L in range(self.L):
            self.graphsA[L].replay()
            torch.cuda.synchronize()   # 读 gate
            rp = F.softmax(self.gate.view(-1).float().cpu(), dim=-1)
            tv, ti = torch.topk(rp, self.E)
            tv = tv / tv.sum()
            ids = ti.tolist()
            ex = self.store.load_layer_experts(L, "mxfp4_mma", active=set(ids))
            gw = [ex[e]["gate"]["wp"] for e in ids] + [ex[e]["up"]["wp"] for e in ids]
            gs = [ex[e]["gate"]["wsc"] for e in ids] + [ex[e]["up"]["wsc"] for e in ids]
            dw = [ex[e]["down"]["wp"] for e in ids]
            ds = [ex[e]["down"]["wsc"] for e in ids]
            self.g_wpt.copy_(torch.tensor([x.data_ptr() for x in gw], dtype=torch.int64, device=dev))
            self.g_wst.copy_(torch.tensor([x.data_ptr() for x in gs], dtype=torch.int64, device=dev))
            self.d_wpt.copy_(torch.tensor([x.data_ptr() for x in dw], dtype=torch.int64, device=dev))
            self.d_wst.copy_(torch.tensor([x.data_ptr() for x in ds], dtype=torch.int64, device=dev))
            self.topv.copy_(tv.to(dev))
            self.graphsB[L].replay()

        if need_lm_head:
            self.graphFin.replay()
            return self.llogits, []
        return None, []