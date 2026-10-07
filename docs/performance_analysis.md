# 优化执行链条复盘：Qwen3-30B-A3B MXFP4 在 RTX 5060 8GB 上

> 目的：把分阶段报告（[P0_quant_report](file:///home/hngs/moe_fused_inference/docs/reports/P0_quant_report.md)、[P0_dev_lessons](file:///home/hngs/moe_fused_inference/docs/reports/P0_dev_lessons.md)、[P2_acceptance](file:///home/hngs/moe_fused_inference/docs/reports/P2_acceptance.md)、[P3_graph_decision](file:///home/hngs/moe_fused_inference/docs/reports/P3_graph_decision.md)）串成**一条完整的执行链条**：每一步做了什么、碰到什么问题、基于什么证据做了什么决策。
> 硬件：RTX 5060 8GB（Blackwell，probe 实证 sm_120a），DRAM 实测 388.4 GB/s，PCIe H2D 实测 13.4 GB/s（Gen4 x8）。
> 一句话结果：30B MoE 以 0.76GB 静态权重 + FP8 KV 在 8GB 上连续 decode，吞吐从 **2.2 → 2.8 → 10.9 → 16.4 tok/s**，32K 上下文可行；20 tok/s 未达成，瓶颈最终被定位为「调度缝隙 + 显存容量」硬边界而非计算。

---

## 〇、一页纸总览

| 阶段 | 关键动作 | 碰到的核心问题 | 决策 | TPOT / 吞吐 |
|---|---|---|---|---|
| **基线探测** | probe kernel 枚举硬件能力 | 5060 是 120 / 120a / 120f？文档查不到变体 | 运行时 probe，只信本机能跑的指令 | — |
| **格式选型** | MXFP4 vs NVFP4、混合精度分配 | 瓶颈在 PCIe 带宽不在算力 | MXFP4 主 + NVFP4 fallback；专家 FP4 / 常驻层 FP8 / KV FP8 | — |
| **P0 能跑** | 自量化 → 自研 MMA → 8GB 分层 loader | torch.mx 无 GEMM；fragment 推反；GCC11 工具链 | 自研 CUDA；PTX 权威公式+数值单测；.cu/.cpp 解耦 | 455ms / **2.2** |
| **P2 v1** | expert LRU cache + 全指标测量 | expert 体积算错 OOM；命中率远低预估 | cap=500 + 碎片配置；如实记录 43% 命中 | 355ms / **2.8** |
| **P2 v2** | 静态量化 + kernel 族重写 + FP8 KV | 每步上万 kernel，launch 缝隙主导 | grouped MoE 3 次 launch/层；直消费 packed nibble | 92ms / **10.9** |
| **P3 Graph** | 逐层分段 CUDA Graph | host topk 强制 48 次同步；数值漂移 | **放弃 Graph**（结构性不可行，非 bug） | 117ms / 8.6（倒退） |
| **P3 投机** | lossless prompt-lookup | 创意文本 n-gram 命中低 | 接受文本相关的收益上限 | 61ms / **16.4（1.51×）** |

贯穿始终的证据优先级：**本机硬证据（probe/数值单测/计数器） > 官方权威文档 > 第三方博客 > 自己的推导**。

---

## 一、起点：硬件实测与格式选型（一切决策的地基）

### 1.1 硬件变体只能运行时探测

- **问题**：RTX 5060 的 compute capability 只读到 `12.0`，而 `nvcc --list-gpu-arch` 不列变体（a/f）；plain sm_120 编不了 block-scale FP4，早期按 sm_89 编译又报 unsupported toolchain。120a 与 120f 的 FP4 能力不同，无法从任何静态信息区分。
- **决策**：写 [`quant/probe_mxfp4.cu`](file:///home/hngs/moe_fused_inference/quant/probe_mxfp4.cu) 在真机上逐条枚举指令，以「能编译能跑」为唯一判据。
- **实证结论**：5060 = **sm_120a**；三条指令全 PASS——FP8 `m16n8k32.e4m3`、MXFP4 `m16n8k64.kind::mxf4...scale_vec::2X...ue8m0`、NVFP4 `...mxf4nvf4...4X...ue4m3`。
- **配套决策**：[CMakeLists.txt](file:///home/hngs/moe_fused_inference/CMakeLists.txt) 在 `project()` **之前**置 `CMAKE_CUDA_ARCHITECTURES OFF`（CMake 3.20 不认 sm_120 会强注 sm_75），再显式 `-gencode=arch=compute_120a,code=sm_120a`。

### 1.2 带宽实测纠正了预算前提

- 原假设 PCIe 是 Gen5；实测 H2D 仅 **13.4 GB/s（Gen4 x8）**，DRAM 388.4 GB/s。
- 影响：这是一条「带宽紧、显存小」的机器，后续所有选型围绕**减少搬运**而非压榨算力展开。

### 1.3 格式与精度分配（见 [project_design §2](file:///home/hngs/moe_fused_inference/docs/project_design.md)）

- **MXFP4 为主**：E8M0 block=32，元数据 0.25 bit/elem（比 NVFP4 少一半），PCIe 瓶颈下等效多搬有效 payload；OCP 开放标准；probe 已实证；业界实现少（差异化）。NVFP4 留作精度 fallback。
- **混合精度**：专家权重 MXFP4（~15GB 大头，CPU pinned）/ Attention+router+norm FP8（~1.2GB GPU 常驻，对量化敏感）/ KV cache FP8（48KB/token，32K≈1.5GB）/ 累加 FP32，epilogue 再量化。
- **模型选型**：Qwen3-30B-A3B（128 专家选 8，激活仅 3.3B；有官方 FP8 checkpoint；工业主流）。

---

## 二、P0「能跑」：从量化到 8GB 跑通

执行顺序：**量化器 → 框架能力探测 → 自研 MMA 算子 → 离线精度 → 真实硬件路径 → 分层 loader**。正确性金字塔自下而上：算子级 → 端到端 → 系统级，下层不变绿不往上堆。

### 2.1 量化器：三个容易写错的格式细节

代码：[quant/fp4_format.py](file:///home/hngs/moe_fused_inference/quant/fp4_format.py)、[quant/quantize_mxfp4.py](file:///home/hngs/moe_fused_inference/quant/quantize_mxfp4.py)。

1. **E2M1 用幅值表而非指数公式**：8 个对称幅值 `{0,.5,1,1.5,2,3,4,6}`，最近邻查表 + ties-up 取大；nibble 的 bit3 是符号。不要硬套「2 指数 1 尾数」去逐个算。
2. **E8M0 scale 用 FP32 计算**：scale 仅 2 的幂（`2^(code-127)`），但求 scale 的过程必须 FP32，否则误差沿层复合。
3. **LSB-first nibble 打包**：偶数下标在低 4 位，与 PTX 寄存器布局一致（延续 FP8 项目 `__byte_perm` selector `0x7362` 的纪律）。

**输入源决策（消融实验）**：从官方 FP8 checkpoint 量化 vs 从 BF16 原版量化，重建权重 cosine 仅差 0.03%——官方 128×128 block FP8 已足够逼近 BF16。故主路径用 FP8→MXFP4，省掉 60GB BF16 下载。

### 2.2 框架能力探测：先探测再自研

- 打印 `dir(torch)` 并尝试 `import torch.mx`：torch 2.11.0+cu130 只有 `float4_e2m1fn_x2`/`float8_e8m0fnu` dtype，**无 GEMM 算子**，`torch.mx` ModuleNotFoundError → 决策自研 CUDA。
- PTX 官网 WebFetch 超时 → `curl` 把 PTX ISA 拉到本地用正则解析，权威文档离线可引用。
- 第三方博客写 `m16n8k32 / kind::mxf8f6f4 / 1X`，与本机 probe 通过的 `m16n8k64 / kind::mxf4 / 2X` 冲突 → **以本机 probe 为准**。

### 2.3 自研 MMA：fragment 布局不靠脑补

代码：[src/ext/mxfp4_linear.cu](file:///home/hngs/moe_fused_inference/src/ext/mxfp4_linear.cu)、[src/ext/mxfp4_common.cuh](file:///home/hngs/moe_fused_inference/src/ext/mxfp4_common.cuh)。

- **问题 1：A fragment 的 reg1/reg2 行映射推反**。最初把 reg1(i 8–15)→g，PTX ISA 8.7 权威公式是 reg1→g+8；此前 FP8 项目已踩过 a[1]/a[2] 翻转。**决策**：一律先套 ISA 规范公式，再用数值单测实证，不靠推导。
- **问题 2：scale 字节→scale 行的映射文档留白**。把「字节↔行」做成**待实证假设**，保留权威 selector（byte-id/thread-id=0），用单 tile 数值单测裁决——cosine≈0.992 且不随 shape 变，确认映射正确（若错位 cosine 会≈0/负）。
- **问题 3：三目与位运算混用的优先级歧义** → 行判定显式加括号，可读性优先。
- 指令定型：`mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0`，FP32 累加；P0 用正确性优先的简单静态 tiling，不做流水线（留 P2）。

### 2.4 工具链：本阶段最曲折的一段

| 现象 | 根因 | 解决/决策 |
|---|---|---|
| `__half`/`__half2float` undefined | 只 include 了 cuda_runtime.h | 补 `cuda_fp16.h`（类型未定义先想漏头文件） |
| torch 头 `need 'typename' before decltype` | **GCC11 解析缺陷**（GCC12 修复） | 用「同一份 torch 头：g++12 直编过、nvcc 中转失败」的最小复现隔离；新建独立 conda gcc12 环境 |
| 架构污染 | nvcc 编设备、g++ 编 torch 绑定混在一起 | **`.cu` 不含任何 torch 头；torch/pybind 绑定拆到 `.cpp` 由 g++12 直编**（[mxfp4_binding.cpp](file:///home/hngs/moe_fused_inference/src/ext/mxfp4_binding.cpp)）——既绕 bug 也让架构更干净 |
| ninja 装了仍报错 | cpp_extension 找 PATH 上的二进制 | 把 env 的 bin 加 PATH |
| 两个 `-ccbin` 冲突 | 设了 CC 又手写 flag | 不设 CC，用 CXX + 单个显式 -ccbin |

### 2.5 精度验证：判定标准必须先于测量

离线路径（packed nibble 反量化 BF16 再 GEMM，**只衡量权重量化误差**）：PPL +2.35%、top-8 路由匹配 95.5%、top-1 一致 91.7% → PASS。

**一次重要的自我纠错**：曾自创一条子标准「要求激活量化 KL < 权重量化 KL」，这是无物理依据的假设。真实 MMA 路径实测相反——**权重 +2.0%、激活量化 +3.75%**（总 PPL 18.16，vs FP8 +5.84%）。处理：回到动手前定的准则（PPL 劣化 <10%、路由匹配 >90%）判定达标，并显式写明「原假设错了」。教训：区分「假设」与「准则」，既不能事后松阈值（p-hacking），也不能用过严假设误杀正确实现。

同时把**验证范围写在报告最前面**：离线反量化 BF16 ≠ 真实 FP4 Tensor Core 计算，二者分步骤报告，不把「算子正确」和「系统正确」混为一谈。

### 2.6 分层 loader：8GB 跑通，并定位下一个靶点

- 非专家 FP8 权重 2940MiB GPU 常驻；6144 个专家 MXFP4 放 CPU pinned，按每层 top-8 同步 on-demand 搬运、算完即释。
- 结果：连续 128 步 decode 不 OOM，峰值仅 **3107 MiB（40.4%）**；但 TPOT 455ms ≈ 2.2 tok/s。
- **物理量归因**：显存才用 40%（不是显存墙），DRAM 388 vs PCIe 13.4 GB/s 且每步重读百 MB~GB 级专家 → 455ms 是**同步 on-demand 搬运/同步瓶颈，不是算力**。这决定了 P2 的优化靶点是缓存/预取/overlap，而不是先抠 kernel。
- 附带教训：测试脚本漏写层索引、标量/列表命名混淆，浪费两轮完整 GPU run → 端到端 run 很贵，先用 1 段 10 步做廉价 smoke。

---

## 三、P2 v1：expert cache + 完整测量（先量准再优化）

- 加 LRU expert cache（cap 预算 ~3GB）。
- **问题：expert 体积算错导致首次 OOM**。预估单专家 2.35MB，实测 6MB（3 投影 × 2048² packed nibble + scale），cap=1500 需 9GB 必然 OOM → 修正 cap=500≈3GB，配 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 治碎片。
- **命中率 43.2%，远低于预估的 70–90%**：128 选 8 的稀疏路由相邻 token 漂移比想象大；如实记录，不套用「MoE 局部性一定强」的脑补。
- 收益：TPOT 433→355ms（-18%），2.3→2.8 tok/s；p99/mean=1.29（cache 引入轻微长尾）。
- **关键对照实验（结构性结论）**：MXFP4 直消费 355ms vs FP8 反量化+cublas 776ms，**快 2.2×**。根因是 FP8 路径每步要做 48×8×3=1152 次反量化，直消费 packed nibble 省掉整张中间张量——证明 FP4 的价值不只是省显存，更省反量化开销。

---

## 四、P2 v2：静态化 + kernel 族重写（计算层压到位）

代码：[src/ext/mxfp4_decode.cu](file:///home/hngs/moe_fused_inference/src/ext/mxfp4_decode.cu)、[runtime/model_runner_v2.py](file:///home/hngs/moe_fused_inference/runtime/model_runner_v2.py)。

- **静态权重 MXFP4 直消费**：embed/lm_head/attn/gate 2.94GB → **0.76GB** GPU 常驻（embed 行 gather 后反量化，其余走自研 MMA）。
- **kernel 族**：`prequant`（激活逐 1×32 block 在线量化）/ `lin_pq`（cp.async 权重流）/ 分组 `gateup` / fused `silu+down` / `combine`；**grouped MoE 每层只发 3 次 kernel**，支持 M 个 token 批量（为投机验证铺路），与朴素 kernel 数值 cos>0.999。
- **KV/attention**：FP8(e4m3) KV 预分配（48KB/token），免每步 `torch.cat`；GQA 直接走 SDPA `enable_gqa`，免 8× `repeat_interleave`。
- 结果：TPOT **92ms / 10.9 tok/s**，算子核对 cos>0.997，PPL fp8=17.16 / mxfp4=17.50。
- **profiler 暴露新瓶颈**：纯 GPU kernel self-time 仅 **27.6ms/步**（理论已够 ~36 tok/s），墙钟却 92ms，差值 ~65ms 是 launch/scheduling/同步/H2D 缝隙；每步发射约 **10,858 个 kernel**。→ 优化对象从「计算」明确转向「调度」，催生 P3。

---

## 五、P3：CUDA Graph 失败 → 转向投机解码

### 5.1 Graph 方案与实测（一次有价值的失败）

- 为消 launch 缝隙，新增 graph 友好组件（均单测 cos>0.99）：`attn_decode_kernel`（M=1、读 device 指针、就地 e4m3 解码替代 torch SDPA）、`kvwrite_kernel`（fp8 不支持 index_copy）、gateup/down 指针设备表；runner 做**逐层分段图** `graphA[L]`(attn) → host topk → `graphB[L]`(MoE)。
- 实测**反而更差**：92ms → 117ms（8.6 tok/s），且 logits cos≈0.986、生成出现重复循环。
- **根因是结构性的，不是实现 bug**：该 MoE 每层都有 host 依赖（gate logits → host topk 选 8 专家 → 填指针表 → MoE），每个决策点必须 `torch.cuda.synchronize()`，48 层 = **每步至少 48 次强制同步**，彻底抵消 graph 消除 launch 的收益。数值漂移来自自定义 e4m3 编码与 torch `.to(fp8)` 的 RNE 舍入/下溢边界不一致，经残差长链累积。
- **决策：放弃整步 Graph**。结论沉淀为原则——CUDA Graph 只适合「结构固定、无 host 中转」的负载；逐层动态路由是它的天敌。

### 5.2 fused MoE 的硬边界评估

工业级做法（vLLM/TRT-LLM）是把 gate-topk + 多专家 GEMM + combine 全部 GPU 内自洽，从而无 host 同步、graph 顺畅。但全专家常驻需 48×128×2.45MB ≈ **15GB > 8GB**。→ 在 8GB 上「完整 fused + 无 host + graph 顺畅」三者不可兼得，故当下不采用，保留给更大显存/更狠量化。

### 5.3 转向 lossless prompt-lookup 投机解码

- 用 prompt n-gram 生成草稿、多 token 并行验证；被拒 token 与贪心逐 token **完全一致（零质量风险）**。
- 实测 61ms / **16.4 tok/s（1.51×）**；但加速比强依赖文本重复度——创意文本 accept≈1.1（几乎无收益），收益主要在代码/文档/自提示等重复文本。20 tok/s 目标仍未达成。

---

## 六、最终状态与诚实边界

- **达成**：30B MoE 在 8GB 上稳定 decode（峰值显存 40% 量级），10.9 tok/s、投机 16.4 tok/s；32K 上下文（FP8 KV ≈1.5GB）；生成连贯，PPL 17.50，算子 cos 0.99+。
- **未达成且写明原因**：20 tok/s。计算层已压到 27.6ms/步（理论 ~36 tok/s），真实墙是每步上万 kernel 的调度缝隙 + 8GB 无法常驻全专家的容量硬边界。
- **后续按 ROI 排序的可行路径**：① batch/single 位级对齐，把投机定义为 batch-greedy 的 lossless 并在重复任务上实测；② 供给侧重构（后台 stream 预取 expert，overlap ~6ms H2D）+ 合并小 launch；③ 更大显存常驻全专家后，再上完整 fused MoE + CUDA Graph（本机不可行）。

---

## 七、问题 → 决策 速查表

| # | 问题 | 决策 | 依据类型 |
|---|---|---|---|
| 1 | 120/120a/120f 无法静态区分 | runtime probe，只信能跑的指令 | 本机硬证据 |
| 2 | sm_89/sm_120 编译报错 | project() 前关 ARCHITECTURES + 显式 gencode 120a | 编译实证 |
| 3 | PCIe 代际假设错（Gen5→Gen4 x8） | 全部围绕「减少搬运」重构选型 | 带宽实测 |
| 4 | 选 MXFP4 还是 NVFP4 | MXFP4 主（元数据少、省带宽）+ NVFP4 fallback | 瓶颈推导 |
| 5 | torch.mx 无 GEMM | 自研 block-scale MMA | 能力探测 |
| 6 | 博客指令与本机冲突 | 以 probe 通过的 m16n8k64/mxf4/2X 为准 | 本机 > 第三方 |
| 7 | fragment reg 行映射推反 | 套 ISA 8.7 公式 + 单测实证 | 权威文档+实证 |
| 8 | scale 字节映射文档留白 | 待实证假设，cos≈0.992 裁决 | 数值单测 |
| 9 | GCC11 编不过 torch 头 | gcc12 环境 + .cu/.cpp 解耦 | 最小复现隔离 |
| 10 | 自创「激活KL<权重KL」门 | 废弃，回到事先定的 PPL/路由准则 | 方法论自律 |
| 11 | expert 体积算错 OOM | cap=500 + expandable_segments | 预算重算 |
| 12 | 命中率 43% 低于预估 | 如实记录，路由漂移是模型特性 | 实证 > 脑补 |
| 13 | 455ms 慢因何 | 显存 40%+带宽对比 → 搬运瓶颈非算力 | 物理量归因 |
| 14 | 92ms 中计算仅 27.6ms | 转向调度优化（Graph/投机） | profiler 拆解 |
| 15 | Graph 117ms 反而更差 | 放弃（逐层 host topk=48 次同步，结构性） | 根因定位 |
| 16 | fused MoE 诱人 | 不采用（全专家 15GB>8GB 硬边界） | 显存预算 |
| 17 | 投机创意文本无收益 | 接受文本相关上限，定位重复任务 | 实测接受率 |

---

## 八、可迁移的方法论

1. **证据优先级**：本机 probe/计数器 > 官方文档 > 第三方 > 自己脑补；代码最终只认这台机器上的证据。
2. **正确性金字塔**：算子级（cos 0.99+）→ 端到端（PPL/KL/路由）→ 系统级（不 OOM/水位/TPOT），下层不绿不往上堆。
3. **用物理量归因，不靠感觉**：40% 显存 ⇒ 非显存墙；388 vs 13.4 GB/s ⇒ 搬运瓶颈；kernel 27.6ms vs 墙钟 92ms ⇒ 调度瓶颈。
4. **判定标准先于测量**：区分假设与准则；不事后松阈值，也不让无据假设误杀实现；预期与实测的偏差本身是结论。
5. **验证范围前置声明**：反量化 BF16 GEMM ≠ 真实 FP4 MMA，分步骤诚实标注。
6. **无基线不立指标**：同程序、同 shape、同环境对比；先用廉价 smoke 暴露笔误，再放大规模。
7. **优化依赖顺序**：搬运被缓存/overlap 隐藏后，算子优化才显价值——这决定了 P0→P1→P2 的包顺序，也解释了为何在带宽紧的机器上先做系统、后抠 kernel。
8. **区分「实现 bug」与「结构约束」**：CUDA Graph 的倒退不是代码写错，而是逐层 host 路由的结构使然；识别这一点，才能果断止损而不是反复调参。
