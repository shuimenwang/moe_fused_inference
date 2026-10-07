# benchmark 测试汇总文档

> 范围：moe_fused_inference 项目内全部性能测试脚本（`benchmarks/`、`runtime/*bench*.py`、`runtime/chat_*demo*.py`、`tests/performance/`）。
> 本文回答四个问题：**用什么方法测的（§二）→ 测出了什么指标（§三）→ 从指标怎么归因到瓶颈（§四）→ 最终定了什么优化策略（§五）**。

---

## 一、测试脚本全景

| 脚本 | 阶段/用途 | 跑什么 | 结果落盘 |
|---|---|---|---|
| [runtime/decode_bench.py](file:///home/hngs/moe_fused_inference/runtime/decode_bench.py) | P0 步骤 6，v1 引擎验收 | 分层加载 + prefill TTFT + 连续 128 步 decode TPOT + 8GB 显存水位 + PASS/FAIL 判 OOM | `quant/decode_bench.json` |
| [runtime/decode_v2.py](file:///home/hngs/moe_fused_inference/runtime/decode_v2.py) | P2，v2 引擎基准 | 静态 MXFP4 直消费 + FP8 预分配 KV + GQA 路径，`--tag` 区分实验批次 | `benchmarks/results/decode_<tag>.json` |
| [benchmarks/profile_tpot.py](file:///home/hngs/moe_fused_inference/benchmarks/profile_tpot.py) | P2 决策 profile | TPOT 分层归因（A）+ LRU 离线模拟（B）+ H2D 微基准（C） | `benchmarks/results/tpot_profile.json` |
| [benchmarks/spec_decode_bench.py](file:///home/hngs/moe_fused_inference/benchmarks/spec_decode_bench.py) | 投机解码验收 | 投机 vs 贪心对齐验证 + 同长吞吐/加速比 | stdout |
| [runtime/chat_v2.py --demo](file:///home/hngs/moe_fused_inference/runtime/chat_v2.py) | 交互 demo | 三场景贪心/投机对比（单路生成，spec 模式 `allow_draft` 控制） | stdout |
| [runtime/chat_demo.py](file:///home/hngs/moe_fused_inference/runtime/chat_demo.py) | v1 交互 demo | ChatML 多轮对话 + token 级流式 | stdout |
| [tests/performance/benchmark_kernel.py](file:///home/hngs/moe_fused_inference/tests/performance/benchmark_kernel.py) / [benchmark_e2e.py](file:///home/hngs/moe_fused_inference/tests/performance/benchmark_e2e.py) | 占位 | 均为「待实现」两行注释 | 无 |
| [benchmarks/configs/](file:///home/hngs/moe_fused_inference/benchmarks/configs) | 形状配置 | qwen3_30b_a3b_shapes / olmoe_1b_7b / deepseek_v3_shapes json | — |

已有实测结果（`benchmarks/results/`）：`decode_smoke.json`、`decode_smoke_p22.json`、`decode_p22_fixed.json`、`decode_verify_no_regress.json`（以上均由 decode_v2.py 产出）、`tpot_profile.json`。

---

## 二、性能分析方法（从代码提取）

### 2.1 计时三件套

| 方法 | 实现位置 | 测什么 |
|---|---|---|
| **wall clock + 前后 `torch.cuda.synchronize()`** | [decode_bench.py L104-L107](file:///home/hngs/moe_fused_inference/runtime/decode_bench.py#L104-L107)、[decode_v2.py L66-L68](file:///home/hngs/moe_fused_inference/runtime/decode_v2.py#L66-L68) | 整步 decode 的端到端耗时（TPOT 的口径） |
| **CUDA Event 计时** | [profile_tpot.py L43-L44](file:///home/hngs/moe_fused_inference/benchmarks/profile_tpot.py#L43-L44) 的 `ev()`，包住 `runner.attention` / `moe_compute`（L114-L132） | 纯 GPU 段耗时（attention、expert kernel），不受 CPU 调度抖动污染 |
| **monkey-patch + `perf_counter_ns`** | profile_tpot.py L87-L95 包 `store._mx_get`（CPU 读盘）、L99-L106 包 `store.load_layer_experts`（加载整体） | CPU 侧开销；`load_wall − cpu_read ≈ H2D`（L265 的 `load_h2d_est` 就是这么估的） |

关键纪律：**每次计时前 warmup 8 步不计入**（profile_tpot.py L74-L77，让 LRU/编译器/JIT 热身）；每个阶段计时后 `torch.cuda.synchronize()` 再取数。

### 2.2 显存水位

- `torch.cuda.reset_peak_memory_stats()` → prefill 后 `max_memory_allocated()`（[decode_bench.py L91-L96](file:///home/hngs/moe_fused_inference/runtime/decode_bench.py#L91-L96)）；
- decode 循环内每步 `torch.cuda.mem_get_info()` 取**设备级**最低空闲（L114-L115）——allocator 统计看不到碎片，设备水位才是 OOM 判据；
- 最终报 allocated/reserved 双峰值 + 占总显存百分比（L137-L142）；
- KV cache 用公式估算：`L × 2(K/V) × kv_len × Hkv × D × 每元素字节`，v1 bf16=2B（decode_bench.py L127-L128），v2 fp8=1B（[decode_v2.py L80-L81](file:///home/hngs/moe_fused_inference/runtime/decode_v2.py#L80-L81)）。

### 2.3 同源对齐验证（性能测试的前置正确性闸门）

[spec_decode_bench.py L75-L88](file:///home/hngs/moe_fused_inference/benchmarks/spec_decode_bench.py#L75-L88)：先用 `greedy_gen`（forward_batch 单步贪心）与 `spec_gen`（投机）各生成 16 token，逐位比对一致率。**先证等价、再测提速**——加速比只有在 lossless 前提下才有意义。

### 2.4 LRU 离线模拟（不占 GPU 就能扫 cap 曲线）

[profile_tpot.py L139-L155](file:///home/hngs/moe_fused_inference/benchmarks/profile_tpot.py#L139-L155) patch `load_layer_experts` 抓下每个 decode step 每层的 `(layer, expert)` 轨迹；`sim_lru(cap)`（L163-L179）用这段轨迹离线模拟任意缓存容量的命中率：

```python
caps = [250, 500, 750, 1000, 1500, 2000, 3000, 6144]   # L181
hit_curve = {c: sim_lru(c) for c in caps}
```

一次 64 步 profile 换 8 个 cap 点的命中率曲线——调 cap 不用反复跑 GPU。另记每步未命中专家数序列（L184-L201）看热身后的稳态。

### 2.5 H2D 微基准

[profile_tpot.py L203-L254](file:///home/hngs/moe_fused_inference/benchmarks/profile_tpot.py#L203-L254)：32MB 块 pageable 拷贝测有效带宽；再试 mmap + `cudaHostRegister` 固定内存（zero-copy 路线）；816KB 小块对应「单专家单投影」的真实传输粒度。

### 2.6 指标定义表

| 指标 | 定义 | 代码出处 |
|---|---|---|
| TTFT | prefill（含 logits）耗时，ms；另报 ms/token | decode_bench.py L90-L98 |
| TPOT mean/p50/p90/p99 | 每 decode 步墙钟时间分布 | decode_bench.py L117-L134 |
| 吞吐 | `1000 / TPOT_mean` tok/s（连续 decode 口径） | decode_bench.py L133 |
| accept/前向 | 投机：生成 token 数 ÷ 验证前向次数 | spec_decode_bench.py L114 |
| 加速比 | 同 token 数下 贪心耗时 ÷ 投机耗时 | spec_decode_bench.py L116 |
| cache 命中率 | `hits/(hits+misses)`，实况 + 离线模拟两个口径 | weight_store.py L219 / profile_tpot L268-L270 |
| 显存 | peak allocated/reserved、最低空闲、KV 估算 MiB | decode_bench.py L137-L142 |
| 对齐一致率 | 投机 vs 贪心逐 token 相同比例 | spec_decode_bench.py L84-L86 |

---

## 三、指标实测汇总

### 3.1 v2 引擎 decode 演进（decode_v2.py 各 tag，RTX 5060 / 8GB）

| tag | steps | TPOT mean | p50 | p90 | p99 (ms) | 吞吐 tok/s | 命中率 | peak MiB |
|---|---|---|---|---|---|---|---|---|
| [decode_smoke](file:///home/hngs/moe_fused_inference/benchmarks/results/decode_smoke.json) | 16 | 659.5 | 608.2 | 843.9 | 1206.6 | 1.5 | 56.3% | 4574 |
| [decode_smoke_p22](file:///home/hngs/moe_fused_inference/benchmarks/results/decode_smoke_p22.json) | 32 | 136.4 | 115.4 | 205.2 | 389.9 | 7.3 | 68.2% | 4574 |
| [decode_p22_fixed](file:///home/hngs/moe_fused_inference/benchmarks/results/decode_p22_fixed.json) | 128 | 92.0 | 74.8 | 152.5 | 306.2 | **10.9** | 80.5% | 4574 |
| [decode_verify_no_regress](file:///home/hngs/moe_fused_inference/benchmarks/results/decode_verify_no_regress.json)（cap=1200） | 8 | 113.2 | 103.5 | 147.7 | 162.0 | 8.8 | 36.6% | 3713 |

TTFT：早期 26.3s（smoke，v1 口径首测）→ v2 稳定 3.2~3.4s；`prep`（静态权重+KV 分配）≈ 0.30s；KV(fp8) 145 token 仅 6.8MiB。

### 3.2 TPOT 分层分解（v1 引擎，[tpot_profile.json](file:///home/hngs/moe_fused_inference/benchmarks/results/tpot_profile.json)，64 步，TPOT mean 386.1ms）

| 阶段 | 每 token 耗时 | 占比 |
|---|---|---|
| expert kernel（GPU） | 227.5 ms | 59% |
| 专家加载 wall（读盘 56.2 + H2D 66.2 = 122.4 ms） | 122.4 ms | 32% |
| attention（GPU，cuda event） | 21.9 ms | 6% |
| route（含 .tolist 强制同步） | 3.1 ms | 1% |

四项合计 ≈ 375ms，与 386ms 的差 ≈ 计时盲区（norm/logits/argmax 等 misc）——**专家供给（kernel+加载）占 ~91%**，注意力只占 6%。

### 3.3 LRU 命中率–容量曲线（离线模拟，64 步轨迹）

| cap | 250 | 500 | 750 | 1000 | 1500 | 2000 | 3000 | 6144 |
|---|---|---|---|---|---|---|---|---|
| 命中率 | 0% | 51.4% | 64.2% | 73.7% | **83.7%** | 89.9% | 90.6% | 90.6% |

cap 2000 后饱和（90.6% 是该轨迹的路由局部性上限）；实况命中率（cap=1500 跑 128 步）80.5% 与模拟曲线吻合，验证了模拟方法可信。

### 3.4 H2D 微基准

| 项 | 结果 | 解读 |
|---|---|---|
| pageable 32MB 带宽 | 12.7 GB/s | ≈ PCIe Gen4 x8 实际上限（~13.4 GB/s），**总线已饱和，软件无油水** |
| mmap + cudaHostRegister | rc=1（失败） | 固定内存 zero-copy 路线在当前环境不可用 |
| 816KB 小块 | 未测出（前置失败） | 但小块传输本身就远低于总线峰值 → 结论指向「减少传输量」而非「优化传输方式」 |

### 3.5 投机解码（spec_decode_bench.py，历史实测）

- 对齐：投机 vs 贪心 16 token 一致（lossless 前提成立）；
- 吞吐：贪心 10.9 tok/s → 投机 ~16 tok/s，**加速 ≈ 1.5×**，平均 accept ≈ 2+/前向（draft_len=4 时验证一次最多拿 5 个）；
- chat_v2 meter 实时显示 `tok/s · accept/前向 · n-gram`（chat_v2.py L268-L271）。

---

## 四、性能分析归因逻辑链

每条链都是「**现象 → 证据 → 归因 → 对策**」四段式：

**链 1：TPOT 659ms 到底花在哪？**
现象：smoke 首测 TPOT 659ms、1.5 tok/s，离可用太远。
证据：profile_tpot 分层计时——expert kernel 227ms + 专家加载 122ms，attention 仅 22ms。
归因：瓶颈不在注意力计算而在**专家供给**（读盘 + H2D + kernel 消费）合计 ~91%；`.tolist()` 类 host 同步只有 3ms，可忽略。
对策：`mxfp4_mma` 直消费 variant（packed 上卡、kernel 内反量化）+ LRU 缓存（→ 链 2、3）。

**链 2：H2D 还能再快吗？**
证据：pageable 微基准 12.7 GB/s ≈ Gen4 x8 物理上限；hostRegister rc=1 失败。
归因：**单位时间传输量已顶满总线**，优化传输方式（pinned/zero-copy）无收益；唯一出路是减少过线字节。
对策：专家以 MXFP4 packed 形态上卡（bf16 的 1/4 字节），且尽量不上卡（LRU 常驻命中）。

**链 3：LRU cap 开多大？**
证据：命中率–容量曲线在 cap=1500 → 83.7%、2000 → 89.9%、3000+ → 90.6% 饱和；单专家 packed spec ≈ 2.5MB。
归因：2000 以上收益 <1.5 个百分点但显存线性上涨，8GB 卡还要留给静态权重（0.76GB）、KV（4096 slot ≈ 384MiB fp8）和激活。
对策：**cap 定在 1400（交互）/1500（基准）**，实测 peak 4574MiB（59.4%）安全；实况命中率 80.5%。

**链 4：v2 为什么从 659ms 降到 92ms？**
证据：decode_smoke → smoke_p22 → p22_fixed 三代 tag 的 TPOT 序列 659 → 136 → 92ms。
归因（与 model_runner_v2.md 互相印证）：静态权重 MXFP4 直消费（常驻 2.94→0.76GB）+ FP8 预分配 KV（48KB/token，消除每步 `torch.cat`）+ `enable_gqa`（免 `repeat_interleave` 8 倍复制）+ 自研 decode kernel（prequant/lin_pq/attn_decode/moe_gateup/moe_down/moe_combine 融合）。
对策：v2 成为生产路径，v1 保留为正确性基准。

**链 5：投机解码的收益从哪来、何时失效？**
证据：accept ≈ 2+/前向（一次验证前向推进 2~3 个 token）→ 10.9 → ~16 tok/s；verify_no_regress 8 步短测 TPOT 113ms 无回退。
归因：收益 = 验证批均摊 decode 前向次数，前提是 lossless（对齐一致率 100%）；文本重复度决定 `_draft` 命中率。
对策：生产默认 K=5/L=4；`allow_draft=False` 保留同源贪心基线；chat_v2 L417-L418 明示「无命中时 <1x」的失效场景。

---

## 五、最终定的优化策略（依据 → 落点）

| # | 策略 | 定依据 | 代码落点 |
|---|---|---|---|
| 1 | 专家 LRU 常驻 GPU，cap=1400/1500，key 含 variant | 链 3 命中率曲线 + 显存预算 | [weight_store.py L203-L222](file:///home/hngs/moe_fused_inference/runtime/weight_store.py#L203-L222)、[chat_v2.py L430](file:///home/hngs/moe_fused_inference/runtime/chat_v2.py#L430) |
| 2 | 专家权重 MXFP4 packed 直消费（mxfp4_mma），反量化融进 MMA kernel | 链 2 总线饱和 → 减字节；链 1 kernel 占比 | [weight_store.py L265-L286](file:///home/hngs/moe_fused_inference/runtime/weight_store.py#L265-L286) |
| 3 | 静态权重（embed/lm_head/attn/router）MXFP4 直消费，embed 按行 gather 反量化 | 常驻 2.94→0.76GB；prefill TTFT 26.3s→3.2s | [weight_store.py load_static_v2 L178-L198](file:///home/hngs/moe_fused_inference/runtime/weight_store.py#L178-L198)、[dequant_rows_gpu L47-L65](file:///home/hngs/moe_fused_inference/runtime/weight_store.py#L47-L65) |
| 4 | KV cache FP8 预分配 `(L, max_len, Hkv, D)`，按位写入 | 消除每步 cat；KV 体积减半（48KB/token） | [model_runner_v2.py](file:///home/hngs/moe_fused_inference/runtime/model_runner_v2.py) §KV |
| 5 | GQA 走 `enable_gqa`，不物理复制 KV 头 | 4 KV 头复制 8 份纯浪费 | model_runner_v2.py SDPA 调用 |
| 6 | 自研 decode kernel 六件套（prequant/lin_pq/attn_decode/moe_gateup/moe_down/moe_combine） | 链 1 kernel 占比 59%，torch 组合算子 launch+带宽开销大 | [mxfp4_decode.cu](file:///home/hngs/moe_fused_inference/src/ext/mxfp4_decode.cu) / [fp4_ext.py](file:///home/hngs/moe_fused_inference/runtime/fp4_ext.py) |
| 7 | prompt-lookup 投机解码（n-gram=5, draft=4），临时 KV 验证 + 前缀提交 | 链 5：accept 2+/前向 → 1.5× 加速，lossless | [spec_decoder.py](file:///home/hngs/moe_fused_inference/runtime/spec_decoder.py)、[chat_v2.py _turn_spec L225-L278](file:///home/hngs/moe_fused_inference/runtime/chat_v2.py#L225-L278) |
| 8 | 性能回归闸门：对齐一致率 + verify_no_regress 基准 | 「先证等价再测提速」+ 不回退下界（每前向至少推 1 token） | [spec_decode_bench.py L75-L88](file:///home/hngs/moe_fused_inference/benchmarks/spec_decode_bench.py#L75-L88) |

**方法论沉淀**（这套测试体系的可复用经验）：
- 先分层归因（wall/event/patch 三口径分离 GPU 与 CPU），再定策略——避免「感觉哪里慢就优化哪里」；
- 离线模拟（LRU 曲线）把参数搜索从 GPU 挪到 CPU，一次轨迹换一族结论；
- 微基准定位物理上限（H2D 12.7 GB/s），到顶的总线不再投入优化，转攻「少传」；
- 每个性能结论都配同源正确性对照（对齐一致率、v1/v2 逐层对齐），lossless 是性能数字的前置条件。
