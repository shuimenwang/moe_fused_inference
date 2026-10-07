# P0 量化精度验证报告（MXFP4 vs FP8，Qwen3-30B-A3B）

- 生成时间：2026-09-21 18:07:55
- 源：官方 `Qwen3-30B-A3B-FP8`（E4M3，128×128 block scale）
- 自量化：MXFP4（E2M1，1×32 block，E8M0 scale，FP32 计算）
- 硬件：RTX 5060 8GB sm_120a，实测显存带宽 388.4 GB/s，PCIe H2D 13.4 GB/s（Gen4 x8）

## 0. 验证范围与方法论说明（步骤5/6 补）

本报告 §1–§7 为 P0 初始**离线**验证，其计算路径为：

- **MXFP4**：镜像 unpack nibble × E8M0 scale **反量化为 BF16，再走标准 BF16 GEMM**；激活保持 BF16，**不经过 FP4 tensor core**。
- **FP8 基线**：同样 dequant 后走 BF16 GEMM（非真实 FP8 MMA）。

因此 §2–§4 的 PPL / KL / 路由只衡量「**权重量化误差**」，**不含**「真实 MXFP4 硬件计算路径」与「**激活量化**」两项。该缺口由步骤5（真实 block-scale MMA，见 §8）与步骤6（8GB decode，见 §9）补齐：

- 算子级：自研 m16n8k64 block-scale MMA（sm_120a）对 f32 reference cosine ≈ **0.992**；
- 端到端真实路径 PPL **18.16**（vs FP8 **+5.84%**），其中权重 +2.0%、**激活量化 +3.75%**；
- 8GB 连续 128 步 decode，峰值显存 **3107 MiB（40.4%）**，不 OOM。

## 1. 权重级误差（抽样 1536 / 18432 专家 tensor，seed=0 分层随机）

| 投影 | cosine mean | cosine p5 | cosine min | rel-L1 mean | max-abs mean |
|---|---|---|---|---|---|
| weight | 0.9930 | 0.9928 | 0.9908 | 0.1183 | 0.0373 |

最差 10 个 tensor（按 cosine）：

- `model.layers.35.mlp.experts.25.gate_proj.weight` cosine=0.9908 rel_l1=0.1360
- `model.layers.38.mlp.experts.32.gate_proj.weight` cosine=0.9919 rel_l1=0.1268
- `model.layers.39.mlp.experts.119.gate_proj.weight` cosine=0.9920 rel_l1=0.1281
- `model.layers.7.mlp.experts.113.gate_proj.weight` cosine=0.9920 rel_l1=0.1270
- `model.layers.27.mlp.experts.0.gate_proj.weight` cosine=0.9920 rel_l1=0.1263
- `model.layers.19.mlp.experts.113.gate_proj.weight` cosine=0.9920 rel_l1=0.1265
- `model.layers.28.mlp.experts.0.gate_proj.weight` cosine=0.9922 rel_l1=0.1240
- `model.layers.41.mlp.experts.54.gate_proj.weight` cosine=0.9922 rel_l1=0.1256
- `model.layers.41.mlp.experts.122.up_proj.weight` cosine=0.9922 rel_l1=0.1256
- `model.layers.33.mlp.experts.100.gate_proj.weight` cosine=0.9923 rel_l1=0.1232

## 2. PPL（WikiText-2-raw，seq=512，8 段）

| 格式 | PPL |
|---|---|
| FP8（官方） | 17.873 |
| MXFP4（自量化） | 18.293 |

**PPL 相对劣化：+2.35%**

## 3. Logits 分布（KL(p_fp8 \|\| p_mxfp4)）

- 逐 token KL 均值：**0.041311**，段最大 0.056009
- top-1 token 一致率：**91.75%**

## 4. 路由漂移（MoE 头号指标）

- top-8 专家集合匹配率（交集大小/8）：**95.45%**
- router softmax KL 全局均值：**0.003024**
- router KL 最差 5 层：L30=0.00549, L32=0.00530, L31=0.00494, L33=0.00478, L36=0.00478

> 注：router 权重本身未量化；router KL 完全来自上游逐层累积误差。

## 5. Greedy 生成 sanity（固定 seed）

- Prompt: `The meaning of life is`
- FP8 生成：` a question that has been pondered by philosophers, scientists, and thinkers for centuries. It is a question that is both profound and elusive, and there is no single answer that satisfies everyone. However, there are several perspectives that can help us understand the meaning of life from different angles.
One perspective is the philosophical one.`
- MXFP4 生成：` a question that has been asked for centuries. It is a question that has no definitive answer, but it is one that continues to be explored by philosophers, scientists, and individuals alike. The meaning of life can be interpreted in many ways, depending on one's beliefs, values, and experiences. Some people believe that the`
- token 匹配：7.8%

> 说明：greedy 生成对首 token 的微小扰动极敏感（蝴蝶效应），token 级匹配率低不代表语义失真。从上方文本可见，FP8 与 MXFP4 生成语义高度一致（均围绕"生命的意义被哲学家/科学家思考数百年"展开），仅措辞/句式分叉——这是 FP4 量化的正常表现，PPL 与 KL 已证明数值保真。

## 6. 结论（判定基准：以 FP8 自身误差线为锚）

- PPL 劣化 +2.35%（阈值 <10%）
- 路由匹配 95.5%（阈值 >90%）
- top-1 一致 91.7%（阈值 >90%）

**总体判定：PASS — MXFP4 可进入步骤5/6**

## 7. 输入源消融（FP8 源 vs BF16 源）

同一套 MXFP4 量化器，分别从官方 FP8 checkpoint 和 BF16 原版量化，对比重建权重与 FP8 真值的 cosine：

| 输入源 | cosine mean | cosine p5 | cosine min | rel-L1 mean |
|---|---|---|---|---|
| FP8 → MXFP4 | 0.9930 | 0.9928 | 0.9908 | 0.1183 |
| BF16 → MXFP4 | 0.9927 | 0.9925 | 0.9905 | 0.1201 |

**差值仅 -0.00033（0.03%）**。结论：官方 FP8 的 128×128 block 量化精度极高，FP8 权重已充分逼近 BF16 真值；从 FP8 量化与从 BF16 量化产出的 MXFP4 镜像无实质差异。**主路径选择 FP8→MXFP4 合理**（省去下载 60GB BF16 的成本）。

## 8. 步骤5：真实 MXFP4 block-scale MMA 验证

### 8.1 `torch.mx` 不可用 → 自研

torch 2.11.0+cu130 仅有 dtype `torch.float4_e2m1fn_x2` / `torch.float8_e8m0fnu`，**无 GEMM 算子**，`import torch.mx` 报 ModuleNotFoundError → 走自研 CUDA 路径。

### 8.2 算子与布局

- PTX：`mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0`，**FP32 累加**；A/B fragment 与 scale selector 严格按 PTX ISA 8.7 权威公式。
- 权重直接消费 packed E2M1 nibble（LSB-first）+ E8M0 block scale；**激活在 kernel 内逐 1×32 block 在线量化**。
- 正确性优先的简单静态 tiling（每 warp 一个 16×8 tile），不做 persistent / SMEM 流水线 / ldmatrix（留 P2）。
- 代码：[src/ext/mxfp4_linear.cu](src/ext/mxfp4_linear.cu)（nvcc 设备）+ [src/ext/mxfp4_binding.cpp](src/ext/mxfp4_binding.cpp)（宿主绑定）。

### 8.3 算子级数值（`tests/correctness/test_mxfp4_mma.py`）

单 tile / 多 tile / 非整除 K（pad）/ 真实专家形（gate/up 768×2048、down 2048×768）：

- MMA vs「量化权重理想执行 x·Wrecᵀ」cosine ≈ **0.992** → **fragment 布局与 scale 字节行映射正确**（若错位 cosine 会≈0/负）；
- vs 量化前真值 cosine ≈ 0.985–0.99，误差来自权重+激活的 FP4 量化本身。

### 8.4 端到端三路（`tests/correctness/test_e2e_mma.py`，WikiText-2，seq=512，4 段）

| 路径 | 说明 | PPL |
|---|---|---|
| fp8 | 官方权重 dequant + BF16 GEMM | 17.156 |
| mxfp4 | 权重反量化 BF16 + BF16 GEMM（激活不量化） | 17.501 |
| mxfp4_mma | 真实 block-scale MXFP4 MMA（激活 1×32 在线量化） | **18.157** |

- PPL：权重 FP4 贡献 **+2.0%**，**激活 FP4 量化贡献 +3.75%**；真实 MMA vs FP8 总劣化 **+5.84%**。
- logits KL：fp8‖mxfp4=0.0376，**mxfp4‖mma=0.0690**（激活量化），fp8‖mma=0.0984。
- 路由 top-8 匹配：fp8~mxfp4=95.4%，**fp8~mma=93.2%**，mxfp4~mma=94.3%。

**判定：按 P0 既定准则（PPL 劣化 <10%、路由匹配 >90%）达标。** 诚实标注：真实硬件路径的主要代价是**激活量化（+3.75%）**而非权重量化——这是初始离线路径未覆盖、步骤5实证补出的量。

## 9. 步骤6：8GB 分层 loader + decode

**分层策略**：非专家 FP8 权重（embed/lm_head/attn/gate/norm，435 tensor）GPU 常驻；128 专家 MXFP4 按 top-8 active 集合逐层从镜像读取、算完即释放（不常驻）。

| 指标 | 数值 |
|---|---|
| 非专家权重常驻 / 加载耗时 | 2940 MiB / 8.27 s |
| TTFT（17 token，冷启动 on-demand） | 9108 ms（≈536 ms/token） |
| TPOT mean / p50 / p90 | 454.9 / 445.5 / 486.0 ms（≈2.2 tok/s） |
| allocated 峰值（占比） | **3107 / 7700 MiB（40.4%）** |
| reserved 峰值 / decode 最低空闲 | 3208 MiB / 4246 MiB |
| KV cache（长度 145） | 14 MiB |

- 连续 **128 步 decode 不 OOM**；生成英文流利、语法标点正确、无乱码（真实 MMA 生成 sanity）。
- **瓶颈归因**：TPOT 455ms 由「**同步 on-demand 权重重读 + H2D，无 overlap、无缓存**」主导——显存仅用 40%、非算力墙。这正是 P2 的优化靶点：专家权重缓存 / 预取 / 计算-搬运 overlap / 更优 tiling。
