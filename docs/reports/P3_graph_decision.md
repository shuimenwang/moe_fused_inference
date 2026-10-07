# P2→P3：CUDA Graph 化 decode 的取舍复盘

> 日期：2026-XX（v3 实测）
> 结论：**在 8GB 显存 + 逐层 host 路由依赖下，CUDA Graph 整步化不可行**；改用 lossless prompt-lookup 投机解码保 20 tok/s。

## 一、目标与核心问题

- 目标：decode 步冲 20+ tok/s（< 50ms/step）。
- profiler 实测（cap=2000）：纯 GPU kernel 计算仅 `~27.6ms/step`（理论已够 ~36 tok/s），
  但墙钟 `~92ms/step`，差值 ~65ms 是 **launch/scheduling/同步/H2D 缝隙**。
  每步发射 **~10,858 个 kernel**。

## 二、CUDA Graph 方案与实现（P2.2 kernel → P3）

思路：把每步上万次 kernel 发射固化为图，消 launch 缝隙。
- 新增 graph 友好的组件并全部单测通过（数值 cos>0.99）：
  - `attn_decode_kernel`（M=1、读 device `dpos`、就地 e4m3 解码，替代 torch SDPA）
  - `kvwrite_kernel`（读 device `dpos` 写 fp8 KV；`index_copy_` 不支持 fp8）
  - `moe_gateup_tab / moe_down_tab`（指针设备表，随 replay 更新内容）
- `model_runner_v3`：预分配持久 buffer + **逐层分段图**：
  - `graphA[L]`（attn）：layernorm→prequant→q/k/v→rope→decode-attn→o→gate
  - host：读 gate→softmax/topk→8 expert→更新指针表
  - `graphB[L]`（MoE）：gateup/down/combine（读指针表）

## 三、实测结果（v3 vs v2，同一 prefill + 单步 decode）

| 指标 | v2（现状） | v3 Graph |
|---|---|---|
| TPOT mean | 92ms | **117ms（更差）** |
| 吞吐 | 10.9 tok/s | **8.6 tok/s** |
| logits vs v2 | 良好 | **cos≈0.986**（累积漂移，生成出现重复循环） |

## 四、根因（结构性，非实现 bug）

本 MoE 模型**每层都有 host 依赖**：
```
gate logits → host topk 选 8 专家 → 填指针表 → 该层 MoE
```
每个"host 决策点"必须 `torch.cuda.synchronize()` 读回 gate。
- 48 层 × 至少 1 次 sync/步 = **48 次/步强制同步**，彻底抵消了 graph 消除 launch 的收益。
- 数值漂移来自（a）自定义 e4m3 编码与 torch `.to(fp8)` 的 RNE 舍入/下溢边界不一致；
  （b）多步小误差经残差累积分叉。

**关键教训：CUDA graph 的价值在"结构固定、无 host 中转"的工作；对逐层动态路由（host topk）的 MoE decode，graph 分段反而引入同步，得不偿失。**

## 五、fused MoE 单算子评估（工业级形态）

- 若把 gate-topk + 多专家 GEMM + combine 全部 GPU 内自洽 → 无 host 同步 → graph 顺。
  这是 vLLM/TensorRT-LLM 的做法，方向正确。
- **硬约束：全模型专家权重常驻 = 48 × 128 × 2.45MB ≈ 15GB，超 RTX 5060 的 7.5GB。**
- 结论：8GB 上"完整 fused + 无 host + graph 顺畅"三者不可兼得；逐层流式权重又回到 host 选路。
- 故**本项目当下不采用完整 fused**，保留为更大显存/更狠量化的长期路径。

## 六、当前采用：lossless prompt-lookup 投机解码

- 利用 prompt n-gram 生成草稿，并行验证多 token；接受 run 则一次推进多步。
- 接受失败的 token 与贪心**逐 token 完全一致**（零质量风险）。
- 预期接受率 2–3×，把 v2（10.9 tok/s）推到 22–32 tok/s。
- 技术要点：lossless prompt-lookup 投机解码，与 MXFP4 引擎的批量验证调度。

## 七、可迁移原则

1. **先定位瓶颈是"计算"还是"调度"**：profiler 区分 kernel self-time（27.6ms）
   与 launch/同步墙缝（65ms），再造方案——否则优化错对象。
2. **host 异步（topk/路由）是 graph 的天敌**：有逐层 host 决策的结构，graph 收益被同步抹平。
3. **显存是方案的硬边界**：工业 fused MoE 依赖大卡全驻留；8GB 必须做权重流式或投机覆盖。
4. **数值对齐要早定**：自定义量化编码与 torch 转换舍入边界不一致会在长链累积漂移。