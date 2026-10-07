# MoE Fused Inference — 消费级 Blackwell 上的 30B MoE 推理引擎

> 在 **单张 RTX 5060（8GB，sm_120a）** 上，通过 **MXFP4 量化 + 专家 offload + 自研 fused CUDA kernel + 投机解码**，部署并优化 **Qwen3-30B-A3B（30B 参数 MoE）** 的端到端推理。
>
> 贪心 **10.9 tok/s** · 投机解码 **~16 tok/s** · 峰值显存 **4.6GB / 8GB（59%）**

---

## 项目定位

MoE 大模型（Qwen3-30B-A3B、DeepSeek-V3 等）的参数量动辄 30B-70B，即使量化到 FP4 也远超消费级显卡显存。我想解决的问题是，在显存远小于模型体积的硬件条件下，通过分层显存管理、专家按需加载、低精度量化与算子融合，让大模型在消费级 GPU 上达到可用的推理速度。

这不是一个通用推理框架，而是一次针对**「单卡 8GB + Blackwell MXFP4 + PCIe Gen4 x8」**这一特定硬件角落的极限部署实践。我把从量化、kernel、运行时到投机解码的整条链路都自己写了一遍，在 vLLM/SGLang 尚未充分覆盖的消费级 Blackwell + MXFP4 场景给出了可复现的实测数字，也顺带整理了一套从阶段拆解、曲线定参到负结果证伪的性能归因过程。

---

## 实测性能

硬件：RTX 5060 8GB（sm_120a，PCIe Gen4 x8，H2D 12.7 GB/s）
模型：Qwen3-30B-A3B（48 层，128 专家 top-8，H=2048，Mi=768）
量化：MXFP4（E2M1 + E8M0，block=32）权重，FP8 KV cache

### 端到端 decode（128 步，cap=1500）

| 指标 | 数值 |
|---|---|
| TPOT mean | 92.0 ms |
| TPOT p50 | 74.8 ms |
| TPOT p90 | 152.5 ms |
| TPOT p99 | 306.2 ms |
| 吞吐（贪心） | 10.87 tok/s |
| 吞吐（投机解码） | ~16 tok/s |
| 专家 LRU 命中率 | 80.5% |
| 峰值显存 | 4574 MiB（59.4%） |
| KV（145 token） | 6.8 MiB |

> 数据来源：[`benchmarks/results/decode_p22_fixed.json`](benchmarks/results/decode_p22_fixed.json)

### 量化精度

| 路径 | PPL | 相对 FP8 |
|---|---|---|
| FP8 官方（E4M3） | 17.87 | — |
| MXFP4 反量化后 BF16 GEMM | 18.29 | +2.35% |
| MXFP4 真实 MMA + 激活量化 | 18.16 | +5.84%（其中激活量化 +3.75%） |

算子级 cosine ≈ 0.992。详见 [`docs/reports/P0_quant_report.md`](docs/reports/P0_quant_report.md)。

---

## 性能归因链条

本项目的优化决策全部基于可追溯的测量数据，而非直觉。完整链路见 [`docs/performance_analysis.md`](docs/performance_analysis.md)，核心逻辑如下：

### 阶段一：v1 基线归因（TPOT 386ms）

用 CUDA event + monkey-patch 对 64 步做阶段拆解：

| 阶段 | 耗时/token | 占比 |
|---|---|---|
| Expert kernel GPU | 227.5 ms | 59% |
| 专家加载（读盘 56 + H2D 66） | 122.4 ms | 32% |
| Attention GPU | 21.9 ms | 6% |
| Router | 3.1 ms | 1% |

**归因**：瓶颈在专家计算（227ms，逐专家 Python 循环 + 非融合 kernel）和专家搬运（122ms，PCIe Gen4 x8 已打满）。

### 阶段二：融合 kernel + LRU 提容量（TPOT → 92ms）

- **融合 grouped kernel**（`moe_gateup/moe_down/moe_combine`）：把 E=8 个专家的 GEMM 打包进一个 kernel，消除逐专家 launch 开销；
- **packed 直消费**：专家权重以 MXFP4 packed 形态上 GPU（2.5MB/专家），不反量化，使 LRU cap 从 500 提到 1500；
- **LRU 曲线定参**：录 64 步真实路由轨迹，离线模拟 8 个容量点，取拐点 cap=1500（命中率 84%，再大收益 <1.5pt）。

### 阶段三：调度瓶颈（92ms 里纯计算仅 27.6ms）

时间线 profiler 显示：纯 GPU kernel self-time 仅 27.6ms/step，墙钟 92ms，差值 65ms 是 **每步 ~10,858 个 kernel 的 launch/同步缝隙**。

**尝试**：CUDA Graph 固化（v3）。
**结果**：TPOT 92 → **117ms（更差）**，logits cos 降至 0.986。
**根因**：MoE 每层有 host 依赖（gate → topk 选专家），48 层 × 1 次 `synchronize` = 48 次强制同步/步，抵消了 graph 收益；且全专家 fused 需 15GB > 8GB。
**决策**：放弃 Graph，转向投机解码。详见 [`docs/reports/P3_graph_decision.md`](docs/reports/P3_graph_decision.md)。

### 阶段四：投机解码（→ ~16 tok/s）

lossless prompt-lookup：用最近 K 个 token 在历史中找重复模式，预测 L 个草稿，一次批量前向验证。接受失败的 token 与贪心逐 token 完全一致（零质量风险）。

剩余瓶颈：p99=306ms 来自专家缓存未命中的搬运长尾（PCIe 物理约束）。

---

## 架构

```
┌──────────────────────────────────────────────────────┐
│ 应用层    chat_v2.py（交互） · benchmarks/（测试）    │
├──────────────────────────────────────────────────────┤
│ 运行时    runtime/                                    │
│   model_runner_v2.py  48 层 attention + MoE 主循环     │
│   weight_store.py     分层加载 + 专家 LRU offload      │
│   spec_decoder.py     prompt-lookup 投机解码          │
│   fp4_ext.py          CUDA 扩展 JIT 编译 + 加载        │
├──────────────────────────────────────────────────────┤
│ 算子层    src/ext/（CUDA/C++，由 fp4_ext.py 编译）     │
│   mxfp4_linear.cu    prefill GEMM（mma m16n8k64）    │
│   mxfp4_decode.cu    decode 6 个 fused kernel         │
│   *_binding.cpp      torch 绑定（bf16↔half、指针表）   │
├──────────────────────────────────────────────────────┤
│ 量化层    quant/                                      │
│   quantize_mxfp4.py   BF16/FP8 → MXFP4 离线量化       │
│   fp4_format.py       E2M1/E8M0/nibble 原语            │
└──────────────────────────────────────────────────────┘
```

详细架构说明见 [`docs/architecture.md`](docs/architecture.md)，各模块代码级文档：

| 模块 | 文档 |
|---|---|
| 推理主循环 | [`docs/model_runner_v2.md`](docs/model_runner_v2.md) |
| 权重加载与 LRU | [`docs/weight_store.md`](docs/weight_store.md) |
| CUDA 扩展编译 | [`docs/fp4_ext.md`](docs/fp4_ext.md) |
| 投机解码 | [`docs/spec_decoder.md`](docs/spec_decoder.md) |
| 性能基准 | [`docs/benchmarks.md`](docs/benchmarks.md) |

---

## 快速开始

### 环境

- NVIDIA Blackwell GPU（sm_120a，已在 RTX 5060 验证）
- CUDA 12.x + nvcc
- PyTorch 2.x
- safetensors、numpy
- conda 环境的 g++12（用于扩展编译，见 [`docs/fp4_ext.md`](docs/fp4_ext.md)）

### 1. 量化模型权重

```bash
# 下载官方 FP8 checkpoint（Qwen3-30B-A3B-FP8）
python quant/download_checkpoint.py

# 离线量化为 MXFP4
python quant/quantize_mxfp4.py --input <fp8_dir> --output <mxfp4_dir>

# 自检（可选）
python quant/quantize_mxfp4.py --selftest
```

### 2. 运行交互对话

```bash
export TORCH_CUDA_ARCH_LIST="12.0a+PTX"
python runtime/chat_v2.py --fp8-dir <fp8_dir> --mxfp4-dir <mxfp4_dir> --max-len 4096
```

首次运行会 JIT 编译 CUDA 扩展（约几十秒），之后缓存复用。

### 3. 跑性能基准

```bash
# TPOT + cache 命中率
python benchmarks/decode_v2.py --output benchmarks/results/my_run.json

# TPOT 阶段拆解 + LRU 容量曲线
python benchmarks/profile_tpot.py

# 投机解码
python benchmarks/spec_decode_bench.py
```

---

## 目录结构

```
moe_fused_inference/
├── runtime/            # Python 推理运行时
│   ├── model_runner_v2.py   # 主推理引擎（active）
│   ├── model_runner.py      # v1 基线 + 共享工具
│   ├── model_runner_v3.py   # CUDA Graph 实验（已证伪，留档）
│   ├── weight_store.py      # 权重加载 + 专家 LRU
│   ├── spec_decoder.py      # 投机解码
│   ├── fp4_ext.py           # CUDA 扩展编译/加载
│   ├── chat_v2.py           # 交互入口
│   └── decode_v2.py         # decode 基准
├── src/
│   ├── ext/            # 活跃 kernel + binding（编译为 .so）
│   │   ├── mxfp4_linear.cu / mxfp4_decode.cu
│   │   ├── mxfp4_binding.cpp / mxfp4_decode_binding.cpp
│   │   └── mxfp4_common.cuh
│   ├── kernels/        # 独立 kernel 实验（k1/k2/k3，未接入主路径）
│   └── runtime/        # 独立 CUDA runtime 实验
├── quant/              # MXFP4 量化工具链
├── tests/              # 正确性 + 性能测试
├── benchmarks/         # 基准脚本 + 实测数据
└── docs/               # 架构、模块、性能文档
```

---

## 正确性验证

三层验证，逐层收敛：

1. **算子级**：自研 block-scale MMA vs FP32 reference，cosine ≈ 0.992；
2. **系统级**：投机 vs 贪心逐 token 对齐测试（[`tests/correctness/align_decode_v2.py`](tests/correctness/align_decode_v2.py)）；
3. **端到端**：PPL、logits KL、路由分布漂移监控。

---

## 路线图

### V0 ✅（已完成）

- MXFP4 量化工具链 + 精度验证
- 双源架构 CUDA 扩展（mma.sync m16n8k64 fused kernel）
- 分层权重加载 + 专家 LRU offload
- FP8 预分配 KV cache + GQA 逻辑广播
- prompt-lookup 投机解码

### V1（进行中/下一阶段）

- **专家预取**：利用路由时间局部性，异步 stream 预取下一层/下一批专家，掩盖 PCIe 长尾（p99）；
- **批量并发搬运**：将当前逐专家串行 `.to(device)` 改为单流并发拷贝列表，消除串行叠加；
- **NCU 微观分析**：对 `moe_down/lin_pq/attn_decode` 做 Nsight Compute 采样，确认 roofline 位置（当前 27.6ms 纯计算的内部瓶颈未量化）；
- **投机解码 batch 对齐**：将投机验证严格定义为 batch-greedy 的 lossless 形式，在重复文本任务上量化 accept 率。

### V2（长期）

- **完整 fused MoE kernel**（gate-topk + 多专家 GEMM + combine 全 GPU 内自洽）：消除 host topk 同步，为 CUDA Graph 铺路；需 >8GB 显存或更激进量化；
- **更大上下文**（32K+）的 KV 管理与 TPOT 衰减曲线；
- **多模型支持**：DeepSeek-V3、Mixtral 等 MoE 架构适配；
- **vLLM/SGLang 插件化**：将 MXFP4 量化 + kernel 抽成独立模块，可被主流推理框架调用。

---

## 硬件约束与诚实边界

- **PCIe Gen4 x8（12.7 GB/s）是硬约束**：未命中的专家搬运无法被完全隐藏，是 p99 长尾的物理来源；
- **8GB 显存无法常驻全部 6144 个专家**（约 15GB），分层 offload + LRU 是必然选择，完整 fused MoE 本机不可行；
- **投机解码收益依赖文本重复度**：代码/文档类文本 accept 率高，创意文本接近无收益；
- 当前 27.6ms 纯 GPU 时间的内部瓶颈**未经 NCU 实测**，2.4% 带宽利用率为基于 TPOT 的反推推断。

这些约束是架构选择的依据，也是后续优化的方向。

---

## 文档索引

- [`docs/benchmarks.md`](docs/benchmarks.md) — 性能基准与分析方法
- [`docs/performance_analysis.md`](docs/performance_analysis.md) — 优化决策链条
- [`docs/reports/P0_quant_report.md`](docs/reports/P0_quant_report.md) — 量化精度报告
- [`docs/reports/P3_graph_decision.md`](docs/reports/P3_graph_decision.md) — CUDA Graph 负结果复盘
