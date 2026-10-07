#!/usr/bin/env python3
# spec_decoder.py - lossless prompt-lookup 投机解码
#
# 思路（vLLM/LMDeploy 同款 n-gram 投机，本项目自研接入 MXFP4 引擎）：
#   1. 草稿：用「最近 K 个 token」在已生成历史中找更早相同子串，取其后续 L 个 token 作候选；
#   2. 验证：把 [历史末token] + L 个草稿 一次性前向（batch T=L+1），
#      logits[t] 预测的应为草稿[t]；逐位置比较 argmax 是否等于草稿到的：
#        * 找到第一个不匹配位置 a：接受前 a 个草稿，第 a 位用模型自己的选择；
#        * 全部匹配：接受全部 L + 模型本次新预测的末尾 token。
#   3. 一次验证前向推进 ~(a+1) 个有效 token，减少 decode forward 次数 → 提速。
#
#   与纯贪心的等价性：接受标准基于「模型 argmax == 草稿 token」；
#   批量 forward 与逐步 decode 用同一权重，argmax 在绝大多数 token 一致（近边界可能差 1 token）。

import os
import sys

import torch


class PromptLookup:
    def __init__(self, runner, n_gram=7, draft_len=4, max_new=512):
        """
        runner: 提供 forward(ids_tensor, start_pos) -> (logits (T,V), routing)，
                支持 batch T>1。
        """
        self.runner = runner
        self.K = n_gram
        self.L = draft_len
        self.max_new = max_new
        self.hist = []            # token int 序列（prompt + 已生成）
        self.tailidx = {}         # tuple(last K tokens) -> 第一次出现起始位置
        self._built_tail = False

    # ---------------------------------------------------------------
    def _rebuild_tailidx(self):
        """从 hist 重建 tail->第一次出现位置。"""
        self.tailidx = {}
        K = self.K
        for i in range(len(self.hist) - K + 1):
            tail = tuple(self.hist[i:i + K])
            if tail not in self.tailidx:
                self.tailidx[tail] = i
        self._built_tail = True

    def _draft(self):
        """返回草稿候选 token 列表（可能为空 = 无 n-gram 复用）。"""
        K = self.K
        if len(self.hist) < K + 1:      # 至少要多一个可续 token
            return []
        tail = tuple(self.hist[-K:])
        j = self.tailidx.get(tail)
        if j is None:
            return []
        if j + K >= len(self.hist):      # 出现位置就是当前尾部，无更早续文
            return []
        return self.hist[j + K : j + K + self.L]

    def _push(self, toks):
        """追加 token，并增量维护 tailidx（每个新尾首次出现才记录）。"""
        K = self.K
        for tk in toks:
            self.hist.append(int(tk))
            # 以新 token 结尾的新 K-gram
            if len(self.hist) >= K:
                new_tail = tuple(self.hist[-K:])
                # 首次出现位置：该尾第一次被记录
                ntail = tuple(self.hist[-K:-1] + [int(tk)]) if False else tuple(self.hist[-K:])
                cand = len(self.hist) - K
                if ntail not in self.tailidx:
                    self.tailidx[ntail] = cand

    # ---------------------------------------------------------------
    def _greedy_one(self, pos):
        """无草稿：单步贪心，返回 (token, 新pos)。"""
        dev = self.runner.kv[0].device
        nxt = torch.tensor([self.hist[-1]], dtype=torch.long, device=dev)
        logits, _ = self.runner.forward(nxt, pos)
        tok = int(logits[-1].argmax().item())
        return tok, pos + 1

    def _draft_kv(self):
        """懒建与主 KV 同形的临时 KV（仅含已验证前缀，验证草稿不污染主 KV）。"""
        shp = self.runner.kv[0].shape
        if getattr(self, "_kv0", None) is None or self._kv0.shape != shp:
            k0 = torch.empty_like(self.runner.kv[0])
            k1 = torch.empty_like(self.runner.kv[1])
            k0.copy_(self.runner.kv[0]); k1.copy_(self.runner.kv[1])
            self._kv0, self._kv1 = k0, k1
        else:
            self._kv0.copy_(self.runner.kv[0]); self._kv1.copy_(self.runner.kv[1])
        return (self._kv0, self._kv1)

    def _verify(self, pos):
        """验证草稿：临时 KV 上一次 forward_batch，接受后提交到主 KV → lossless。"""
        dev = self.runner.kv[0].device
        D = self._draft()
        if len(D) == 0:
            # 无草稿退化单步（用主 KV）
            lg, _ = self.runner.forward_batch(
                torch.tensor([self.hist[-1]], dtype=torch.long, device=dev), pos - 1)
            tok = int(lg[0].argmax())
            self._push([tok])
            return pos + 1
        base = pos - 1
        ver = [self.hist[-1]] + D          # 长度 1+L
        vt = torch.tensor(ver, dtype=torch.long, device=dev)
        dkv = self._draft_kv()
        logits, _ = self.runner.forward_batch(vt, base, kv=dkv)   # (1+L, V)
        a = 0
        while a < len(D) and int(logits[a].argmax()) == D[a]:
            a += 1
        if a < len(D):
            append = D[:a] + [int(logits[a].argmax())]
        else:
            append = D + [int(logits[a].argmax())]
        # 提交接受前缀 [base, base+len(append)) 到主 KV
        n = len(append)
        self.runner.kv[0][..., base:base + n, :, :].copy_(dkv[0][..., base:base + n, :, :])
        self.runner.kv[1][..., base:base + n, :, :].copy_(dkv[1][..., base:base + n, :, :])
        self._push(append)
        return pos + n

    # ---------------------------------------------------------------
    def generate(self, prompt_ids):
        """prompt_ids: 1D list/int of token ids。返回生成 token 列表（不含 prompt）。"""
        self.hist = [int(x) for x in prompt_ids]
        self._build_built = True
        pos = len(self.hist)
        self._rebuild_tailidx()
        out = []
        while len(out) < self.max_new:
            pos = self._verify(pos)
        # 返回实际生成的（self.hist 已含 prompt+生成）
        return self.hist[len(prompt_ids):]

    # 便捷：暴露 hist 供基准外部读取
    def gen_hist(self, prompt_ids, max_new):
        self.hist = [int(x) for x in prompt_ids]
        pos = len(self.hist)
        self._rebuild_tailidx()
        while len(self.hist) - len(prompt_ids) < max_new:
            pos = self._verify(pos)
        return self.hist