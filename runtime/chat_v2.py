#!/usr/bin/env python3
# chat_v2.py - v2 引擎交互体验脚本
#
# 路径：静态 MXFP4 权重 + FP8 预分配 KV + 自研 sm_120a decode kernel
#       （~10.9 tok/s），可选 prompt-lookup 投机解码（重复文本最高 ~16 tok/s）。
#
# 用法：
#   交互：  python runtime/chat_v2.py
#   投机：  python runtime/chat_v2.py --spec
#   对比：  python runtime/chat_v2.py --demo
#
# 交互命令：/spec [on|off] 切换投机  /think [on|off] 切换思考  /clear 清空对话
#           /stats 查看显存       /help                /q 退出

import argparse
import os
import shutil
import sys
import time
import unicodedata

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "runtime"))

from weight_store import WeightStore
from model_runner_v2 import ModelRunnerV2
from spec_decoder import PromptLookup
from decode_bench import build_cfg, mib

FP8 = "/home/hngs/models/Qwen3-30B-A3B-FP8"
MX = "/home/hngs/models/Qwen3-30B-A3B-MXFP4"
ST = "/home/hngs/models/Qwen3-30B-A3B-MXFP4-static"

DIM = "\x1b[2m"      # 暗色
BOLD = "\x1b[1m"
CYAN = "\x1b[36m"
GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
RESET = "\x1b[0m"


# ------------------------------------------------------------------
# 终端流式输出 + 行内实时速度计（自动处理 CJK 宽度 / 自动换行）
# ------------------------------------------------------------------
class LiveStream:
    def __init__(self, muted=False):
        self.col = 0
        self.meter_w = 0
        self.tty = sys.stdout.isatty()
        self.muted = muted      # demo 基线跑法静音，只保留最终换行

    @staticmethod
    def _w(ch):
        return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1

    def _cols(self):
        return shutil.get_terminal_size((100, 30)).columns

    def write(self, text):
        """输出文本；ANSI 转义序列不计入光标列宽。"""
        if self.muted:
            return
        sys.stdout.write(text)
        i = 0
        while i < len(text):
            ch = text[i]
            if ch == "\x1b":                 # CSI 序列：跳到终止字母
                while i < len(text) and not ("@" <= text[i] <= "~"):
                    i += 1
            elif ch == "\n":
                self.col = 0
            elif ch == "\t":
                self.col = (self.col // 8 + 1) * 8
            else:
                w = self._w(ch)
                if self.col + w > self._cols():
                    self.col = 0             # 终端自动换行
                self.col += w
            i += 1
        sys.stdout.flush()

    def meter(self, text):
        """在当前行尾追加暗色速度计（不换行）；超宽则不显示。"""
        if not self.tty:
            return
        self.clear_meter()
        visible = f" {text}"
        w = sum(self._w(c) for c in visible)
        if self.col + w > self._cols():
            return
        sys.stdout.write(DIM + visible + RESET)
        sys.stdout.flush()
        self.col += w
        self.meter_w = w

    def clear_meter(self):
        if not self.tty or self.meter_w == 0:
            return
        sys.stdout.write(f"\x1b[{self.meter_w}D\x1b[0K")
        sys.stdout.flush()
        self.col -= self.meter_w
        self.meter_w = 0

    def newline(self):
        self.clear_meter()
        self.write("\n")


# ------------------------------------------------------------------
# 聊天会话
# ------------------------------------------------------------------
class ChatSession:
    def __init__(self, args):
        self.args = args
        from transformers import AutoTokenizer
        self.cfg = build_cfg(args.fp8)
        self.tok = AutoTokenizer.from_pretrained(args.fp8)
        self.im_end = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.im_start_s = self.tok.convert_ids_to_tokens(151644)
        self.im_end_s = self.tok.convert_ids_to_tokens(151645)
        self.t_think = self.tok.convert_ids_to_tokens(151667)
        self.t_ethink = self.tok.convert_ids_to_tokens(151668)

        self.store = WeightStore(
            args.fp8, args.mxfp4,
            expert_cache_cap=args.cache_cap,
            static_mxfp4_dir=args.static_mxfp4)
        self.runner = ModelRunnerV2(self.store, self.cfg)
        t0 = time.time()
        self.runner.prepare(max_len=args.max_len)
        torch.cuda.synchronize()
        self.prep_s = time.time() - t0

        self.all_ids = []          # 已进入 KV 的全部 token
        self.pos = 0
        self.spec_on = args.spec
        self.think_on = args.think
        self.last = {}

    # ---- ChatML 分块（与官方 chat_template 逐 token 对齐验证过）----
    def _enc(self, s):
        return self.tok(s, add_special_tokens=False).input_ids

    def _sys_block(self):
        return self._enc(f"{self.im_start_s}system\n{self.args.system}"
                         f"{self.im_end_s}\n")

    def _user_block(self, text):
        return self._enc(f"{self.im_start_s}user\n{text}{self.im_end_s}\n")

    def _assistant_header(self):
        if self.think_on:
            return self._enc(f"{self.im_start_s}assistant\n")
        return self._enc(f"{self.im_start_s}assistant\n{self.t_think}\n\n"
                         f"{self.t_ethink}\n\n")

    def reset(self):
        """清空对话：KV buffer 复用，位置归零重写 system。"""
        self.all_ids = []
        self.pos = 0
        ids = self._sys_block()
        self.runner.forward(torch.tensor(ids, device="cuda"), 0,
                            need_lm_head=False)
        self.pos = len(ids)
        self.all_ids = list(ids)

    def _prefill(self, ids):
        t = torch.tensor(ids, dtype=torch.long, device="cuda")
        torch.cuda.synchronize(); t0 = time.time()
        logits, _ = self.runner.forward(t, self.pos)
        torch.cuda.synchronize()
        ttft = time.time() - t0
        self.pos += len(ids)
        self.all_ids += ids
        return logits, ttft

    # ---- 贪心单步路径（forward_decode 快路径）----
    def _turn_greedy(self, logits, max_new, stream, timing):
        nxt = logits[-1].argmax().reshape(1)
        gen, dts = [], []
        printed = [""]
        HOLD = 4          # 压住最后几个 token，避免跨 token UTF-8 打印成 �

        def flush(final=False):
            if final:
                stream.clear_meter()
                safe = self.tok.decode(gen)
            elif len(gen) > HOLD:
                safe = self.tok.decode(gen[:-HOLD])
                while safe.endswith("\ufffd"):    # 边界不完整字节，继续压住
                    safe = safe[:-1]
            else:
                safe = ""
            stream.write(safe[len(printed[0]):])
            printed[0] = safe

        t_start = time.time()
        for _ in range(max_new):
            tok = int(nxt.item())
            if tok == self.im_end or tok == self.tok.eos_token_id:
                break
            gen.append(tok)
            self.all_ids.append(tok)
            flush()
            torch.cuda.synchronize(); t0 = time.time()
            logits, _ = self.runner.forward(nxt, self.pos)
            torch.cuda.synchronize()
            dts.append((time.time() - t0) * 1000)
            self.pos += 1
            nxt = logits[-1].argmax().reshape(1)
            wall = time.time() - t_start
            timing["wall"] = wall
            stream.meter(f"{len(gen)/max(wall,1e-9):.1f} tok/s "
                         f"· {len(gen)} tok · TPOT {dts[-1]:.0f}ms")
        flush(final=True)
        timing["wall"] = time.time() - t_start
        timing["dts_ms"] = dts
        return gen

    # ---- prompt-lookup 投机路径（allow_draft=False 时退化为 batch M=1 同源贪心）----
    def _turn_spec(self, max_new, stream, timing, allow_draft=True):
        pd = PromptLookup(self.runner, n_gram=self.args.n_gram,
                          draft_len=self.args.draft)
        if not allow_draft:
            pd._draft = lambda: []        # 永不命中草稿 = 逐 token batch 贪心
        pd.hist = list(self.all_ids)
        pd._rebuild_tailidx()
        pos = self.pos
        gen, n_verify = [], 0
        printed = [""]
        HOLD = 4

        def flush(final=False):
            if final:
                stream.clear_meter()
                safe = self.tok.decode(gen)
            elif len(gen) > HOLD:
                safe = self.tok.decode(gen[:-HOLD])
                while safe.endswith("\ufffd"):
                    safe = safe[:-1]
            else:
                safe = ""
            stream.write(safe[len(printed[0]):])
            printed[0] = safe

        t_start = time.time()
        while len(gen) < max_new:
            prev = len(pd.hist)
            torch.cuda.synchronize(); t0 = time.time()
            pos = pd._verify(pos)
            torch.cuda.synchronize()
            n_verify += 1
            chunk = pd.hist[prev:]
            cut = next((i for i, t in enumerate(chunk)
                        if t == self.im_end or t == self.tok.eos_token_id),
                       len(chunk))
            cut = min(cut, max_new - len(gen))
            eff = chunk[:cut]
            gen += eff
            self.all_ids += eff
            pos = prev + cut                       # 截断后的逻辑 KV 位置
            flush()
            wall = time.time() - t_start
            acc = len(gen) / max(n_verify, 1)
            timing["wall"] = wall
            stream.meter(f"{len(gen)/max(wall,1e-9):.1f} tok/s · {len(gen)} tok "
                         f"· accept {acc:.2f}/前向 · n={self.args.n_gram}")
            if cut < len(chunk):                  # EOS 或达长度上限
                break
        flush(final=True)
        timing["wall"] = time.time() - t_start
        timing["n_verify"] = n_verify
        self.pos = pos
        return gen

    # ---- 一轮完整对话 ----
    def turn(self, text, max_new, stream=None):
        stream = stream or LiveStream()
        ids = self._user_block(text) + self._assistant_header()
        logits, ttft = self._prefill(ids)

        torch.cuda.reset_peak_memory_stats()
        timing = {"ttft": ttft, "wall": 0.0}
        if self.spec_on:
            gen = self._turn_spec(max_new, stream, timing)
        else:
            gen = self._turn_greedy(logits, max_new, stream, timing)
        stream.newline()

        # 收尾 <|im_end|>\n（投机截断时覆盖多余 KV 槽位，位置一致）
        close = self._enc(f"{self.im_end_s}\n")
        self.runner.forward(torch.tensor(close, device="cuda"), self.pos,
                            need_lm_head=False)
        self.pos += len(close)
        self.all_ids += close

        n = len(gen)
        wall = max(timing["wall"], 1e-9)
        self.last = {
            "mode": "spec" if self.spec_on else "greedy",
            "n": n, "ttft": ttft, "wall": wall, "tps": n / wall,
            "think": self.think_on,
        }
        if self.spec_on:
            self.last["n_verify"] = timing.get("n_verify", 0)
            self.last["accept"] = n / max(timing.get("n_verify", 1), 1)
            self.last["ver_ms"] = timing.get("ver_ms", [])
        else:
            self.last["dts_ms"] = timing.get("dts_ms", [])
        return self.last

    # ---- 统计面板 ----
    def panel(self):
        L = self.last
        total = torch.cuda.get_device_properties(0).total_memory
        peak = torch.cuda.max_memory_allocated()
        kv_mib = (self.cfg["num_hidden_layers"] * 2
                  * self.cfg["num_key_value_heads"] * self.cfg["head_dim"]
                  * self.pos) / 1024**2
        hit = self.store.cache_hit_rate()
        mode = f"投机 n-gram={self.args.n_gram}/draft={self.args.draft}" \
            if L["mode"] == "spec" else "贪心 decode kernel"
        lines = [
            f"{DIM}── 本轮统计 ───────────────────────────{RESET}",
            f"模式 {CYAN}{mode}{RESET}    "
            f"TTFT {YELLOW}{L['ttft']*1000:.0f}ms{RESET}",
            f"生成 {BOLD}{L['n']}{RESET} tok / {L['wall']:.1f}s → "
            f"{GREEN}{BOLD}{L['tps']:.1f} tok/s{RESET}",
        ]
        if L["mode"] == "greedy" and L.get("dts_ms"):
            d = np.array(L["dts_ms"])
            lines.append(f"TPOT mean {d.mean():.0f}ms · p50 "
                         f"{np.percentile(d,50):.0f}ms · p90 "
                         f"{np.percentile(d,90):.0f}ms")
        else:
            lines.append(f"验证 {L['n_verify']} 次前向 → 接受 {L['n']} tok "
                         f"（{L['accept']:.2f} tok/前向）")
        lines.append(
            f"上下文 {self.pos}/{self.args.max_len} tok（KV {kv_mib:.0f} MiB）"
            f" · 显存 {mib(peak):.0f}/{mib(total):.0f} MiB "
            f"（{peak/total*100:.0f}%）· 专家缓存命中 {hit*100:.0f}%")
        print("\n".join(lines))


# ------------------------------------------------------------------
# demo 预设：创意（投机弱） vs 代码 / 表格（重复多，投机强）
# ------------------------------------------------------------------
DEMO_PROMPTS = [
    ("创意续写", "请续写下面的故事，约 150 字：\n"
                 "夜幕降临时，灯塔守护者发现海面上出现了一排不属于任何船只的灯。"),
    ("代码生成", "用 Python 分别实现冒泡排序和选择排序，要求：完整类型注解、"
                 "中文 docstring、每个函数末尾给 2 个断言示例。"),
    ("结构化表格", "用 Markdown 表格列出 8 种常见深度学习优化器，"
                   "列为：名称 | 提出年份 | 核心思想 | 主要优点。"),
]


def _demo_gen(sess, prompt, mode, max_new, muted):
    """单路生成。mode: decode=生产贪心 kernel / batch=投机同源基线 / spec=投机。"""
    sess.reset()
    ids = sess._user_block(prompt) + sess._assistant_header()
    logits, ttft = sess._prefill(ids)
    stream = LiveStream(muted=muted)
    timing = {"ttft": ttft, "wall": 0.0}
    t0 = time.time()
    if mode == "decode":
        gen = sess._turn_greedy(logits, max_new, stream, timing)
    elif mode == "batch":
        gen = sess._turn_spec(max_new, stream, timing, allow_draft=False)
    else:
        gen = sess._turn_spec(max_new, stream, timing, allow_draft=True)
    stream.newline()
    wall = time.time() - t0
    return gen, {"tps": len(gen) / max(wall, 1e-9),
                 "accept": len(gen) / max(timing.get("n_verify", 1), 1)}


def run_demo(sess, max_new):
    results = []
    for i, (name, prompt) in enumerate(DEMO_PROMPTS):
        print(f"\n{BOLD}■ 场景 {i+1}/3：{name}{RESET}")
        print(f"{DIM}Q: {prompt[:60]}...{RESET}\n", flush=True)
        # decode 贪心（生产默认，静音）→ batch 同源贪心（静音）→ 投机（流式展示）
        _, p1 = _demo_gen(sess, prompt, "decode", max_new, muted=True)
        b, p2 = _demo_gen(sess, prompt, "batch", max_new, muted=True)
        print(f"{DIM}── 投机路径输出 ──{RESET}")
        s, p3 = _demo_gen(sess, prompt, "spec", max_new, muted=False)
        ncmp = min(len(s), len(b))
        agree = sum(a == x for a, x in zip(s, b)) / max(ncmp, 1)
        first_div = next((i for i, (a, x) in enumerate(zip(s, b))
                          if a != x), ncmp)
        spd = p3["tps"] / max(p2["tps"], 1e-9)
        results.append((name, p1, p2, p3, agree, first_div, spd))
        print(f"{DIM}decode 贪心 {p1['tps']:.1f} tok/s │ batch 同源贪心 "
              f"{p2['tps']:.1f} tok/s │ 投机 {p3['tps']:.1f} tok/s "
              f"(accept {p3['accept']:.2f}) │ 对同源加速 {spd:.2f}× │ "
              f"首分歧@{first_div} · 位置一致率 {agree*100:.0f}%{RESET}")

    print(f"\n{BOLD}── 汇总（tok/s，{max_new} tok/场景）"
          f"──────────────────────{RESET}")
    print(f"{BOLD}  {'场景':<8} {'decode':>6} {'batch':>6} {'投机':>6} {'加速':>6} "
          f"{'accept':>6} {'首分歧':>6}{RESET}")
    for name, p1, p2, p3, agree, first_div, spd in results:
        print(f"  {BOLD}{name:<8}{RESET} {p1['tps']:>6.1f} {p2['tps']:>6.1f} "
              f"{p3['tps']:>6.1f} {spd:>5.2f}x {p3['accept']:>6.2f} "
              f"{first_div:>6}")
    print(f"{DIM}口径：decode=生产默认（自研 decode kernel + SDPA 注意力）；"
          f"batch=投机同源基线（同一 batch attn kernel，M=1 逐 token）。{RESET}")
    print(f"{DIM}「首分歧」= 投机与同源贪心第一个不同 token 的位置（贪心序列一次早翻转即"
          f"全文分叉，故只看分叉点）；接受的草稿 token 按构造等于模型当步 argmax。{RESET}")
    print(f"{DIM}注：两条序列是同一 batch kernel 的两次独立运行，GPU 归约顺序非严格确定，"
          f"近并列 logits 首 token 即可能翻转，属已知数值现象，非投机逻辑错误。{RESET}")
    print(f"{DIM}投机收益取决于文本重复度：代码/表格/文档续写高，创意写作低；"
          f"无命中时验证有额外开销（<1x 即此因）。{RESET}")


# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Qwen3-30B-A3B MXFP4 v2 交互体验")
    ap.add_argument("--fp8", default=FP8)
    ap.add_argument("--mxfp4", default=MX)
    ap.add_argument("--static-mxfp4", default=ST)
    ap.add_argument("--system", default="You are a helpful assistant. 用简体中文回答。")
    ap.add_argument("--max-tokens", type=int, default=256, help="每轮最大生成 token")
    ap.add_argument("--max-len", type=int, default=8192, help="KV 上下文槽位")
    ap.add_argument("--cache-cap", type=int, default=1400, help="GPU 专家缓存上限")
    ap.add_argument("--n-gram", type=int, default=5)
    ap.add_argument("--draft", type=int, default=4)
    ap.add_argument("--spec", action="store_true", help="默认开启投机解码")
    ap.add_argument("--think", action="store_true", help="开启思考模式（默认 no-think）")
    ap.add_argument("--demo", action="store_true", help="非交互：三场景贪心/投机对比")
    ap.add_argument("--demo-new", type=int, default=64, help="demo 每场景 token 数")
    args = ap.parse_args()

    gpu = torch.cuda.get_device_name(0)
    print(f"{BOLD}[init]{RESET} {gpu} · 加载静态 MXFP4 权重 + 编译 decode kernel ...",
          flush=True)
    sess = ChatSession(args)
    total = torch.cuda.get_device_properties(0).total_memory
    print(f"{BOLD}[init]{RESET} 就绪 {sess.prep_s:.1f}s · "
          f"常驻 {mib(torch.cuda.memory_allocated()):.0f} MiB · "
          f"KV 预算 {args.max_len} tok（{48*2*4*128*args.max_len/1024**2:.0f} MiB FP8）"
          f" · 设备 {mib(total):.0f} MiB", flush=True)
    sess.reset()
    print(f"{DIM}默认：{'投机' if sess.spec_on else '贪心'}解码 / "
          f"{'思考' if sess.think_on else 'no-think'}模式 · "
          f"/help 查看命令{RESET}\n")

    if args.demo:
        run_demo(sess, args.demo_new)
        return

    stream = LiveStream()
    while True:
        try:
            user = input(f"{GREEN}{BOLD}[user]>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            continue
        if user in ("/q", "/exit", "/quit"):
            break
        if user == "/help":
            print(f"{DIM}/spec [on|off] 投机解码  /think [on|off] 思考模式\n"
                  f"/clear 清空对话  /stats 显存状态  /q 退出{RESET}")
            continue
        if user == "/clear":
            sess.reset()
            print(f"{DIM}[已清空对话，上下文归零]{RESET}")
            continue
        if user == "/stats":
            free, tot = torch.cuda.mem_get_info()
            print(f"{DIM}显存占用 {(tot-free)/1024**3:.1f}/{tot/1024**3:.1f} GiB · "
                  f"上下文 {sess.pos}/{args.max_len} tok · "
                  f"专家缓存 {len(sess.store.expert_cache)} · "
                  f"命中率 {sess.store.cache_hit_rate()*100:.0f}%{RESET}")
            continue
        if user.startswith("/spec"):
            p = user.split()
            sess.spec_on = (len(p) < 2 or p[1] == "on")
            print(f"{DIM}[投机解码：{'ON' if sess.spec_on else 'OFF'}]{RESET}")
            continue
        if user.startswith("/think"):
            p = user.split()
            sess.think_on = (len(p) < 2 or p[1] == "on")
            print(f"{DIM}[思考模式：{'ON' if sess.think_on else 'OFF（no-think）'}]{RESET}")
            continue

        if sess.pos + args.max_tokens + 64 > args.max_len:
            print(f"{YELLOW}[上下文将超 {args.max_len}，请 /clear 后重试]{RESET}")
            continue
        stream.write(f"{CYAN}{BOLD}[assistant]>{RESET} ")
        try:
            sess.turn(user, args.max_tokens, stream=stream)
        except torch.cuda.OutOfMemoryError:
            print(f"\n{YELLOW}[OOM] 请 /clear 或减小 --max-tokens 后重试{RESET}")
            raise
        sess.panel()
        print()

    print(f"\n{DIM}[exit] 再见{RESET}")


if __name__ == "__main__":
    main()
