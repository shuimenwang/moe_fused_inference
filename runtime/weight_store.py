#!/usr/bin/env python3
# weight_store.py - 双 checkpoint 权重读取（P0 验证 / runner 共用）
#
# 提供：
#   load_static_weights()  非专家权重 bf16 → GPU 常驻（embed/lm_head/attn/gate/norm）
#   load_layer_experts(L, variant)  第L层 128 专家 ×3 权重 bf16 → GPU
#     variant='fp8'   : FP8 128×128 block dequant
#     variant='mxfp4' : MXFP4 镜像 dequant（unpack nibble × block scale）
#
# 设计：所有 safe_open handles 全程复用；逐 tensor 读取，不整模入内存。

import json
import os
import sys
from collections import OrderedDict

import numpy as np
import torch
from safetensors import safe_open

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "quant"))

from quantize_mxfp4 import MXFP4_BLOCK, dequant_fp8_block


def mxfp4_reconstruct_gpu(data_np, scode_np, n, kp, orig_k, device):
    """packed FP4 bytes + E8M0 scale codes（numpy）→ f32 cuda (n, orig_k)。
    全程 GPU：nibble 展开 / E2M1 查表 / exp2 解 scale，全部向量化。"""
    data = torch.from_numpy(np.ascontiguousarray(data_np)).to(device)
    codes = torch.empty(data.numel() * 2, dtype=torch.uint8, device=device)
    codes[0::2] = data & 0xF
    codes[1::2] = (data >> 4) & 0xF
    codes = codes[:n * kp].view(n, kp // MXFP4_BLOCK, MXFP4_BLOCK)

    sign = torch.where((codes & 0x8) != 0, -1.0, 1.0)
    table = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=device)
    vals = table[(codes & 0x7).long()]

    sc = torch.from_numpy(np.ascontiguousarray(scode_np)).to(device)
    scales = torch.exp2(sc.float() - 127.0)

    recon = (sign * vals) * scales[..., None]
    return recon.reshape(n, kp)[:, :orig_k]


def dequant_rows_gpu(wp, wsc, K, ids):
    """MXFP4 packed 权重行反量化（embedding gather 用，P2.1a）。
    wp (N,Kp/2) uint8、wsc (N,Kp/32) uint8 已在 GPU；ids (T,) long。
    返回 (T,K) bf16。"""
    data = wp[ids]                                   # (T,Kp/2)
    T, half = data.shape
    kp = half * 2
    codes = torch.empty(T * kp, dtype=torch.uint8, device=wp.device)
    flat = data.reshape(-1)
    codes[0::2] = flat & 0xF
    codes[1::2] = (flat >> 4) & 0xF
    codes = codes.view(T, kp // MXFP4_BLOCK, MXFP4_BLOCK)
    sign = torch.where((codes & 0x8) != 0, -1.0, 1.0)
    table = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                         device=wp.device)
    vals = table[(codes & 0x7).long()]
    scales = torch.exp2(wsc[ids].float() - 127.0)    # (T,Kp/32)
    recon = (sign * vals) * scales[..., None]
    return recon.reshape(T, kp)[:, :K].to(torch.bfloat16)


class WeightStore:
    def __init__(self, fp8_dir, mxfp4_dir, device="cuda",
                 expert_cache_cap=500, static_mxfp4_dir=None):
        # expert LRU cache：key=(layer, eid, variant) -> {'gate','up','down'} GPU tensors
        # 命中时跳过 numpy→torch H2D；按显存预算定 cap（单 expert ≈ 6MB，500 cap ≈ 3GB）
        self.expert_cache = OrderedDict()
        self.expert_cache_cap = expert_cache_cap
        self.cache_stats = {"hits": 0, "misses": 0}
        self.fp8_dir, self.device = fp8_dir, device

        fp8_index_path = os.path.join(fp8_dir, "model.safetensors.index.json")
        self.fp8_index = json.load(open(fp8_index_path))["weight_map"]
        self.fp8_shards = sorted(set(self.fp8_index.values()))
        self.fp8_handles = {
            s: safe_open(os.path.join(fp8_dir, s), framework="torch")
            for s in self.fp8_shards
        }

        if mxfp4_dir is None:
            # 单路径模式（只跑 FP8，如 smoke test）
            self.manifest = {"tensors": {}}
            self.mx_index = None
            self.mx_shards, self.mx_handles, self.mx_info = [], {}, {}
            return

        self.manifest = json.load(open(os.path.join(mxfp4_dir, "manifest.json")))
        mx_index_path = os.path.join(mxfp4_dir, "model.safetensors.index.json")
        if os.path.exists(mx_index_path):
            self.mx_index = json.load(open(mx_index_path))["weight_map"]
        else:
            self.mx_index = None
        self.mx_shards = sorted(
            set(self.mx_index.values()) if self.mx_index
            else [f for f in os.listdir(mxfp4_dir) if f.endswith(".safetensors")]
        )
        self.mx_handles = {
            s: safe_open(os.path.join(mxfp4_dir, s), framework="numpy")
            for s in self.mx_shards
        }
        # 源名 → 镜像 manifest 条目
        self.mx_info = self.manifest["tensors"]

        # ---- P2.1a：非专家权重 MXFP4 镜像（embed/lm_head/attn/router）----
        self.static_dir = static_mxfp4_dir
        self.st_info, self.st_handles, self.st_index = {}, {}, None
        if static_mxfp4_dir is not None:
            st_idx_path = os.path.join(static_mxfp4_dir,
                                       "model.safetensors.index.json")
            if os.path.exists(st_idx_path):
                self.st_index = json.load(open(st_idx_path))["weight_map"]
                shards = sorted(set(self.st_index.values()))
            else:
                shards = [f for f in os.listdir(static_mxfp4_dir)
                          if f.endswith(".safetensors")]
            self.st_handles = {
                s: safe_open(os.path.join(static_mxfp4_dir, s), framework="numpy")
                for s in shards
            }
            self.st_info = json.load(
                open(os.path.join(static_mxfp4_dir, "manifest.json")))["tensors"]

    def close(self):
        for h in list(self.fp8_handles.values()):
            del h
        for h in list(self.mx_handles.values()):
            del h

    # ----------------------------------------------------------------
    # FP8 checkpoint 读取
    # ----------------------------------------------------------------
    def _fp8_get(self, key):
        return self.fp8_handles[self.fp8_index[key]].get_tensor(key)

    def fp8_dequant(self, key):
        """key 对应权重 dequant → float32 GPU（压缩字节过 PCIe，dequant 在 GPU）。"""
        w = self._fp8_get(key).to(self.device)
        sk = key + "_scale_inv"
        if sk in self.fp8_index:
            return dequant_fp8_block(w, self._fp8_get(sk).to(self.device))
        return w.float()

    def fp8_dequant_cpu(self, key):
        """CPU 版本（统计/调试用）。"""
        w = self._fp8_get(key)
        sk = key + "_scale_inv"
        if sk in self.fp8_index:
            return dequant_fp8_block(w, self._fp8_get(sk))
        return w.float()

    # ----------------------------------------------------------------
    # 非专家权重（GPU 常驻）
    # ----------------------------------------------------------------
    def load_static_weights(self):
        static = {}
        for key in self.fp8_index:
            if "experts." in key or key.endswith("_scale_inv"):
                continue
            static[key] = self.fp8_dequant(key).to(torch.bfloat16)
        return static

    # ----------------------------------------------------------------
    # P2.1a：静态权重 v2（MXFP4 直消费 + 1D norm 仍 bf16）
    # ----------------------------------------------------------------
    def _st_get(self, key):
        if self.st_index is not None:
            h = self.st_handles[self.st_index[key]]
        else:
            h = next(h for h in self.st_handles.values() if key in h.keys())
        return h.get_tensor(key)

    def load_static_v2(self):
        """返回 (fp4, bf16)：
          fp4[key] = {'wp': uint8 cuda (N,Kp/2), 'wsc': uint8 cuda (N,Kp/32),
                      'K': int}
          bf16[key] = 1D norm / 其他未量化小 tensor
        """
        fp4, bf16 = {}, {}
        for key, info in self.st_info.items():
            data = np.ascontiguousarray(self._st_get(info["data_key"]))
            scode = np.ascontiguousarray(self._st_get(info["scale_key"]))
            n, kp = info["orig_shape"][0], info["padded_k"]
            fp4[key] = {
                "wp": torch.from_numpy(data.reshape(n, kp // 2)).to(self.device),
                "wsc": torch.from_numpy(scode).to(self.device),
                "K": info["orig_shape"][1],
            }
        for key in self.fp8_index:
            if "experts." in key or key.endswith("_scale_inv") or key in fp4:
                continue
            bf16[key] = self.fp8_dequant(key).to(torch.bfloat16)
        return fp4, bf16

    # ----------------------------------------------------------------
    # expert LRU cache
    # ----------------------------------------------------------------
    def _cache_get(self, layer, eid, variant):
        key = (layer, eid, variant)
        if key in self.expert_cache:
            self.expert_cache.move_to_end(key)
            self.cache_stats["hits"] += 1
            return self.expert_cache[key]
        self.cache_stats["misses"] += 1
        return None

    def _cache_put(self, layer, eid, variant, edict):
        key = (layer, eid, variant)
        self.expert_cache[key] = edict
        self.expert_cache.move_to_end(key)
        while len(self.expert_cache) > self.expert_cache_cap:
            self.expert_cache.popitem(last=False)

    def cache_hit_rate(self):
        s = self.cache_stats
        total = s["hits"] + s["misses"]
        return s["hits"] / total if total > 0 else 0.0

    # ----------------------------------------------------------------
    # 第 L 层专家
    # ----------------------------------------------------------------
    def load_layer_experts(self, layer, variant, active=None):
        """返回 dict: eid -> {'gate','up','down'} cuda tensors（或 spec dict）。
        active: 指定专家 id 集合；None 表示全部 128 个。
        LRU cache 命中时跳过 numpy→torch H2D；未命中时正常加载并存入 cache。"""
        ids = range(128) if active is None else list(active)
        experts = {}
        if variant == "fp8":
            for e in ids:
                cached = self._cache_get(layer, e, variant)
                if cached is not None:
                    experts[e] = cached
                    continue
                pre = f"model.layers.{layer}.mlp.experts.{e}"
                ed = {}
                for p in ("gate", "up", "down"):
                    key = f"{pre}.{p}_proj.weight"
                    ed[p] = self.fp8_dequant(key).to(torch.bfloat16)
                self._cache_put(layer, e, variant, ed)
                experts[e] = ed
        elif variant == "mxfp4":
            for e in ids:
                cached = self._cache_get(layer, e, variant)
                if cached is not None:
                    experts[e] = cached
                    continue
                pre = f"model.layers.{layer}.mlp.experts.{e}"
                ed = {}
                for p in ("gate", "up", "down"):
                    src_name = f"{pre}.{p}_proj.weight"
                    info = self.mx_info[src_name]
                    data = self._mx_get(info["data_key"])
                    scode = self._mx_get(info["scale_key"])
                    n, kp = info["orig_shape"][0], info["padded_k"]
                    recon = mxfp4_reconstruct_gpu(
                        data, scode, n, kp, info["orig_shape"][1], self.device)
                    ed[p] = recon.to(torch.bfloat16)
                self._cache_put(layer, e, variant, ed)
                experts[e] = ed
        elif variant == "mxfp4_mma":
            # 真实 MXFP4 MMA 路径：不反量化，直接把 packed nibble/E8M0 传 GPU，
            # 由自研 mxfp4 kernel 直接消费（返回 spec dict 而非 weight tensor）。
            for e in ids:
                cached = self._cache_get(layer, e, variant)
                if cached is not None:
                    experts[e] = cached
                    continue
                pre = f"model.layers.{layer}.mlp.experts.{e}"
                ed = {}
                for p in ("gate", "up", "down"):
                    src_name = f"{pre}.{p}_proj.weight"
                    info = self.mx_info[src_name]
                    data = np.ascontiguousarray(self._mx_get(info["data_key"]))
                    scode = np.ascontiguousarray(self._mx_get(info["scale_key"]))
                    n, kp = info["orig_shape"][0], info["padded_k"]
                    wp = torch.from_numpy(data.reshape(n, kp // 2)).to(self.device)
                    wsc = torch.from_numpy(scode).to(self.device)
                    ed[p] = {"wp": wp, "wsc": wsc,
                             "K": info["orig_shape"][1]}
                self._cache_put(layer, e, variant, ed)
                experts[e] = ed
        else:
            raise ValueError(variant)
        return experts

    def _mx_get(self, key):
        if self.mx_index:
            h = self.mx_handles[self.mx_index[key]]
        else:
            h = next(h for h in self.mx_handles.values() if key in h.keys())
        return h.get_tensor(key)
