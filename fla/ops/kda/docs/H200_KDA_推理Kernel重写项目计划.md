# H200 KDA 推理 Kernel 重写个人计划

版本：2026-07-16

这份文档不是给外部 review 的方案说明，而是我自己推进 H200 KDA 推理 kernel 重写时用的工作台。目标是随时回答三件事：

- 现在最该做什么。
- 做到什么程度可以进入下一步。
- 哪些方向看起来诱人，但当前不该碰。

核心判断先写死：**不复刻 FlashKDA 的执行调度，只复用数学分解、数值处理和 correctness oracle。** 这次重写的重点不是论证有没有价值，而是把设计路径落到代码和数据上：V-major WGMMA、V split 扩并行度、TMA/WGMMA/CUDA Core 流水，以及 K1/K2 中间 layout 如何让这些优势进入端到端。

## 0. 一句话目标

在 H200 / SM90a 上做一个 KDA 推理后端：

- K1：把每个 Chunk64 内部所有不依赖历史 state 的计算压成 K2 能消费的 intermediate。
- K2：沿 Chunk64 做 recurrent scan，用 V-major WGMMA 和 V split 解决低 $B \times H_{local}$ 下并行度不足的问题。
- 先 standalone kernel，后 nano-vLLM 单卡；LoRA 和 TP 放到主路径跑通之后。

优先级按 kernel 主路径推进：先 K2 standalone，再 K1+K2 端到端，最后接 nano-vLLM。框架接入排在后面，是为了避免 scheduler/cache/LoRA 把 kernel 问题混在一起。

## 1. 当前固定决策

这些决策先不要反复摇摆。benchmark 的作用是定阈值、定 layout、定调度版本，而不是反复重开方向选择。

| 项目 | 当前决定 | 备注 |
| --- | --- | --- |
| 硬件 | H200 / SM90a | 不背 SM80 兼容包袱。 |
| 语言 | CuTe DSL | 保留 TMA、mbarrier、WGMMA、layout 显式控制，同时少一点 C++ 模板和 Python binding 成本。 |
| 参考 | FLA / FlashKDA | 只当数学、数值和 correctness oracle；不照搬 Chunk16 调度。 |
| Chunk | Chunk64 | 让主计算自然落到 64 相关的 WGMMA 形状。 |
| Subchunk | 16 | 处理 diagonal/off-diagonal block、strict lower、gate 和局部三角依赖。 |
| 主 shape | $D_k=128, D_v=128$ | 第一版只把这条路打穿。 |
| Kernel 划分 | K1 + K2 | 第一版允许中间量落 HBM；后面再压缩 layout 和字节数。 |
| K2 版本 | V128 baseline + V64 split | V64 是低并行度主攻方向，V128 是高并行度和重复读取的对照。 |
| 框架节奏 | 单卡 prefill/decode 后再谈 TP | 不让 scheduler、LoRA、TP 过早污染 kernel 判断。 |

## 2. 不做清单

这些不是永久不做，是在主路径成立前不做。

- 不做 context/sequence parallel。跨 rank recurrent state scan 是另一个算法问题。
- 不把 LoRA 融进 KDA kernel。LoRA 留在线性层，kernel 只吃投影后的 tensor。
- 不追求任意 $D_k$、$D_v$、dtype、模型变体。先锁 H200 + BF16/FP16 输入 + FP32 累加 + $128/128$。
- 不为了“全 WGMMA”把所有东西都塞进 Tensor Core。三角求解、gate、scale、epilogue 该用 CUDA Core 就用 CUDA Core。
- 不先写框架胶水。K2 standalone 指标、接口和 state 语义稳定后，再进入 nano-vLLM。
- 不把文档写成论文。这个文件只服务执行、判断和复盘。

## 3. 总体路线

我要落地的是这条链：

```text
Chunk64 数学正确
  -> CuTe DSL 能稳定发出目标 TMA/WGMMA
  -> K1 intermediate 语义正确
  -> K2 V-major WGMMA baseline 正确且可 profile
  -> V64 split 在低 B*H_local 形成稳定优势
  -> producer/consumer 流水降低 steady-state chunk 时间
  -> K1+K2 端到端没有被中间 HBM 吃掉
  -> 单卡框架接入
```

每一步都留下可复现数据。不要靠口头判断往后滚，也不要让后面的框架工作掩盖前面的 kernel 问题。

## 4. 数学和数据流备忘

### 4.1 两 Kernel 的职责

K1 只做 chunk-local 工作。它看得到当前 Chunk64 内的 $Q,K,V,gate,beta$，但不依赖历史 state，所以所有 chunk 可以并行。

K2 做跨 chunk recurrent scan。它维护 state，按 chunk 顺序推进：

$$
S_{c+1} = F(S_c, I_c)
$$

其中 $I_c$ 是 K1 产生的 intermediate。K2 的 chunk 间存在硬依赖，但每个 chunk 内的搬运、WGMMA、elementwise 和 store 可以流水。

```text
Q / K / V / gate / beta
  |
  v
K1: Chunk64 local transform
  - QK^T: 64 x 128 by 128 x 64 -> 64 x 64
  - gate / mask / scale / stability
  - 4 diagonal 16x16 blocks
  - 6 off-diagonal 16x16 blocks
  - write K2 intermediate
  |
  v
K2: recurrent scan over Chunk64
  - load state
  - compute output from state and Q
  - compute state update from local terms and K
  - store output and final state
```

### 4.2 K1 最小目标

K1 第一版不要一上来就追流水。先拿到可以逐块对 reference 的 intermediate。

K1 内部阶段：

| 阶段 | 目标 | 第一版要求 |
| --- | --- | --- |
| 搬运 | Q64x128、K64x128、gate/beta，必要时 V64x128 | TMA 到 WGMMA operand 友好的 SMEM layout。 |
| 大矩阵 | $QK^T$ 得到 64x64 | WGMMA m64n64k16，K 方向 8 次累加。 |
| 局部处理 | gate、strict lower mask、scale、数值稳定 | 正确优先，可以先 CUDA Core 串一点。 |
| Subchunk 求解 | 4 个 diagonal + 6 个 off-diagonal 16x16 block | 先逐 block dump 对 reference。 |
| 输出 | K2 intermediate | shape、dtype、stride、layout 先写死并测试。 |

K1 的危险点：看上去 QK 很快，但后面的三角处理、elementwise 和 intermediate store 可能主导时间。必须分段 profile，不要只看总 latency。

### 4.3 K2 核心映射

K2 的关键不是“多开几个 block”，而是把 V 维映射到 WGMMA 的 M 维。以 V64 slice 为例：

| 计算 | 形状 | WGMMA |
| --- | --- | --- |
| output | $S^T[64 \times 128] \cdot Q^T[128 \times 64] \to O^T[64 \times 64]$ | m64n64k16 |
| state update | $U^T[64 \times 64] \cdot K[64 \times 128] \to \Delta S^T[64 \times 128]$ | m64n128k16 |
| 窄 token 阶段 | $S^T[64 \times 128] \cdot Q^T[128 \times 16] \to O^T[64 \times 16]$ | m64n16k16 |

两个 K2 版本：

| 版本 | 用法 | 预期 |
| --- | --- | --- |
| K2-V128 | 一个 CTA 处理完整 $D_v=128$ | $B \times H_{local}$ 足够大时避免重复读取。 |
| K2-V64 | V 切成两个 CTA | $B \times H_{local}$ 小时扩 grid，并让 V 对齐 WGMMA M=64。 |

dispatch 不能靠感觉。至少用 $B$、$H_{local}$、$T$ 和实际 profile 指标定阈值。

## 5. SM90 调度草图

目标不是“代码里出现了 TMA/mbarrier/WGMMA”，而是真的形成 memory pipeline 和 compute pipeline：

```text
Producer warp : TMA(c+1) ---- TMA(c+2) ---- TMA(c+3)
Consumer WG0  : O0(c) dS0(c) O0(c+1) dS0(c+1)
CUDA side     : epiO  prepU  epiO    prepU
Consumer WG1  : O1(c) dS1(c) O1(c+1) dS1(c+1)   # 后做
Dependency    : dS(c) commit 后才能用 S(c+1)
```

第一版顺序：

1. 单 CTA / V128 / 简单调度，把数值和 state 传递跑通。
2. V64 双 CTA，验证扩 grid 是否真有用。
3. 再做 producer/consumer 流水。
4. 最后才评估 V128 单 CTA 双 WG 是否值得。

双 WG 不要过早做。它可能省 Q/K/TMA 读取，但会把 register、SMEM、barrier、WG 协调复杂度一起拉高。先用 V64 双 CTA 拿到清楚的性能边界。

## 6. Layout 规则

这个项目最容易死在 layout 来回转置上。第一版要少而硬：

- GMEM logical tensor -> TMA tile -> SMEM physical layout -> WGMMA partition，这条路径每个 tensor 都写清楚。
- K1 和 K2 不各自发明一套 swizzle。先维护两类确认过的 WGMMA operand layout。
- K2 的 state/intermediate store layout 优先服务 V-major WGMMA。
- 通用但运行时要 transpose 的 layout 只能当 fallback。
- 每次改 layout，同时检查三个边界：TMA 到 SMEM、SMEM 到 WGMMA、accumulator 到 store。

每个 layout 变更都要留下一个小测试或 dump。否则后面数值错了很难定位。

## 7. 阶段计划

下面时间是全力开发估计。实际如果按业余时间做，可以等比例拉长，但顺序不变。

### P0. 冻结数学与接口，2-3 天

要做：

- 写清 Chunk64/Subchunk16 的数学表达。
- 定义 K1 intermediate：字段、shape、dtype、stride、layout。
- 建立 reference harness：单 chunk、两 chunk、随机 initial state、非 64 倍数长度。
- 明确 varlen 和 chunk boundary 的处理规则。

退出条件：

- 小 shape reference 全过。
- K1 intermediate 的语义能独立解释，不依赖 kernel 实现。
- 每个 intermediate 字段都知道由谁写、由谁读、是否参与 recurrent state。

先不做：

- 不写完整 CuTe kernel。
- 不做流水。
- 不做框架 API。

### P1. CuTe DSL SM90 骨架，3-5 天

要做：

- 固定 CUDA、CUTLASS、CuTe DSL 版本。
- 迁移或写一个最小 TMA + WGMMA m64n64k16 样板。
- 建 compile、launch、profile 脚本。
- 用 Nsight Compute 保存第一份基线。

退出条件：

- H200 上能稳定编译运行。
- 能确认生成目标 WGMMA/TMA 指令。
- 能拿到 register、SMEM、occupancy、WGMMA issue、TMA bytes。

先不做：

- 不把 KDA 逻辑塞进去。
- 不追求漂亮抽象。helper 够用即可。

### P2. K1 correctness，5-8 天

要做：

- 实现 QK^T 64x64 WGMMA。
- 加 gate/mask/scale/stability。
- 加 4 diagonal + 6 off-diagonal Subchunk16 求解。
- dump raw QK、处理后矩阵、每个 block、最终 intermediate。

退出条件：

- K1 每层中间量都能和 reference 对上。
- 非 64 倍数长度边界正确。
- 第一版 K1 latency 能拆成搬运、QK、局部处理、store 几部分。

先不做：

- 不把 K1 流水作为 correctness 阶段目标。
- 不压 intermediate 字节数，除非已经明显爆炸。

### P3. K2-V128 baseline，4-6 天

要做：

- 单 CTA 处理 $D_v=128$。
- 用最简单调度实现 recurrent scan。
- 验证 output 和 final state。
- 建立 K2 profile baseline。

退出条件：

- 单 chunk、多 chunk、随机 state 都正确。
- final state 随 chunk 增长没有异常漂移。
- 长序列上相对非 WGMMA / MMA baseline 的 latency、资源和 stall 差异都有记录。

调参顺序：

- 如果 V-major WGMMA baseline 没达到预期，先查 operand layout、WGMMA 粒度、state layout 和寄存器压力。
- K2 指标稳定前，不把问题带进框架层。

### P4. K2-V64 split + dispatch，4-6 天

要做：

- 沿 V 切成两个 CTA。
- 测 $B \times H_{local}$ 小、T 长的场景。
- 比较 V64 重复读取 Q/K/metadata 的代价。
- 做一个简单 dispatch 阈值。

退出条件：

- 低 $B \times H_{local}$ 区间 V64 稳定优于 V128。
- 高并行度区间通过 dispatch 不回归。
- 记录清楚 V64 优势来源：occupancy、eligible warp、WGMMA active、还是 steady-state chunk time。

调参顺序：

- 如果只在单个 shape 表现突出，先保留为特化策略，不急着进默认路径。
- 如果 V64 被重复读取吃掉，优先看 L2 locality 和 intermediate layout，不要马上做双 WG。

### P5. K2 producer/consumer 流水，5-8 天

要做：

- 加 TMA stage。
- 管 WGMMA group。
- 把 CUDA Core epilogue、prepU、gate/scale 尽量塞到 WGMMA 间隙。
- 控制 accumulator lifetime，避免 register 爆。

退出条件：

- steady-state chunk latency 相比串行 K2 明显下降。
- register/SMEM 没把 active CTA/SM 降到抵消流水带来的 latency 下降。
- fill/drain 和 steady-state 分开统计。

调参顺序：

- 如果流水让 occupancy 崩了，先降 stage、缩短 accumulator lifetime、拆 CTA。
- 如果仍不行，接受简单调度，不为了形式复杂化。

### P6. K1 流水与两 Kernel 联调，5-8 天

要做：

- 给 K1 的 Q/K/V/intermediate 搬运建 producer/consumer pipeline。
- 调 K1 intermediate store layout。
- 跑 K1+K2 端到端 benchmark。
- 统计 intermediate HBM bytes。

退出条件：

- K1+K2 在目标矩阵领先 FLA/FlashKDA 或至少领先明确 baseline。
- K2 的加速没有被 K1 store 和 K1->K2 HBM 中间量抵消。
- varlen 和 chunked prefill 的 standalone 路径正确。

调参顺序：

- 如果端到端优势不明显，先减少 intermediate 或改 store layout。
- K1/K2 路径稳定后，再进入框架扩展。

### P7. nano-vLLM 单卡接入，7-12 天

要做：

- 设计 KDAStateCache。
- 实现 prefill/decode API。
- 接 slot_ids、cu_seqlens、chunk_offsets。
- 跑 continuous batching 和 chunked prefill。

目标 API：

```python
kda_prefill(
    q, k, v, gate, beta,
    initial_state,
    cu_seqlens,
    chunk_offsets,
) -> output, final_state

kda_decode(
    q, k, v, gate, beta,
    state_cache,
    slot_ids,
) -> output
```

cache 形状先按这个理解：

```text
state_cache[
    layer,
    slot,
    local_head,
    Dk,
    local_Dv
]
```

退出条件：

- 单卡真实模型请求能跑 prefill 和 decode。
- chunked prefill 能把前一段 final_state 正确接到下一段 initial_state。
- state slot 重排和连续请求不会串 state。

### P8. LoRA 与 TP，7-14 天

要做：

- LoRA 仍放在线性层，KDA kernel 不知道 adapter。
- TP 优先按 head 切；每 rank 保存本地 state shard。
- 验证单卡与 TP 数值一致。

先不做：

- 不在 KDA kernel 内做通信。
- 不做 sequence parallel。
- 不为了 mixed-adapter batching 改 kernel ABI。

退出条件：

- LoRA 和 TP 不改变 KDA kernel API。
- 多卡结果与单卡 reference 对齐。

## 8. 第一周任务清单

第一周只追这些，不扩散。

- [ ] 写完 Chunk64/Subchunk16 公式和 K1 intermediate 定义。
- [ ] 建 reference harness：单 chunk、两 chunk、随机 state、非 64 倍数长度。
- [ ] 跑通一个 CuTe DSL TMA + m64n64k16 WGMMA 小核。
- [ ] 实现 K1 raw QK^T dump。
- [ ] 保存第一份 NCU baseline：WGMMA issue、TMA bytes、register、SMEM、occupancy、stall。
- [ ] 写下 P0/P1 遇到的 CuTe DSL 限制和 workaround。

每天结束前只更新三件事：

- 今天确认了什么。
- 今天卡在哪里。
- 明天第一件事做什么。

## 9. 正确性测试

不要只比最终 output。recurrent state 会把早期错误传播和放大，必须逐层对。

| 层级 | 比较对象 | 目的 |
| --- | --- | --- |
| K1-A | raw QK^T 64x64 | 隔离 TMA/SMEM/WGMMA layout。 |
| K1-B | gate/mask/scale 后矩阵 | 隔离 elementwise 和数值稳定。 |
| K1-C | 4 diagonal + 6 off-diagonal block | 验证 Chunk64 层次化求解。 |
| K1-D | 最终 intermediate | 锁定 K1/K2 接口语义和物理 layout。 |
| K2-A | 单 chunk output/state | 验证 V-major WGMMA 和 state update。 |
| K2-B | 多 chunk final state | 验证 recurrent 顺序、barrier、stage phase。 |
| E2E | prefill output、decode output、final state | 验证 scheduler、cache、varlen、chunk boundary。 |

必测边界：

- sequence length：1、15、16、17、63、64、65、127、128、非对齐长序列。
- batch/varlen：空尾块、不同 sequence 长度、slot 重排、同一 state slot 连续 chunked prefill。
- 数值：极端 gate/log-gamma、接近上溢/下溢、BF16/FP16 输入、FP32 state reference。
- 同步：stage 数 2/3/4、单 chunk pipeline fill/drain、多 chunk parity wraparound。
- layout：V64/V128、不同 contiguous stride、K1 intermediate direct-consume 和 transpose fallback。

容差不要拍脑袋。第一版同时记录：

- max_abs
- max_rel
- mse
- final state drift per chunk

如果短序列对、长序列漂，优先查 gate 缩放、log-gamma、subchunk 边界和 chunk boundary，不要先放宽 tolerance。

## 10. 性能测试

### 10.1 Benchmark 矩阵

| 维度 | 取值 | 目的 |
| --- | --- | --- |
| B | 1, 2, 4, 8, 16 | 覆盖低并行度到足够 block。 |
| H_local | 8, 16, 32, 64 | 模拟单卡和 TP 后每卡 head 数。 |
| T | 64, 256, 1024, 4096, 8192 | 分离固定开销和稳态流水。 |
| K2 variant | MMA baseline, WGMMA-V128, WGMMA-V64 | 拆出 WGMMA、V split、流水各自贡献。 |
| 输入 | fixed, varlen, chunked prefill | 避免只优化整齐矩阵。 |

### 10.2 必看指标

| 层级 | 指标 | 用途 |
| --- | --- | --- |
| Kernel | K1/K2 latency、per-chunk steady-state cycle | 分离 fill/drain 和长期吞吐。 |
| Tensor Core | WGMMA active/issue、eligible warp、dependency stall | 确认不是“用了 WGMMA 但喂不饱”。 |
| Memory | HBM/L2 bytes、TMA throughput、L2 hit、重复 Q/K bytes | 判断 V split 和 intermediate HBM 成本。 |
| 资源 | register/thread、SMEM/CTA、active CTA/SM | 识别双 WG 或多 stage 是否把 occupancy 打崩。 |
| 端到端 | prefill latency、tokens/s、TTFT、chunked prefill interference | 确认 kernel 优势能进框架。 |

### 10.3 决策门槛

| 检查点 | 稳定条件 | 调整动作 |
| --- | --- | --- |
| P3: WGMMA-V128 | 长序列 K2 相对 baseline 有稳定优势。 | 查 V-major layout、WGMMA 粒度、state layout；先把指标稳定，再排框架接入。 |
| P4: V64 split | 小 $B \times H_{local}$ 明显优于 V128，高并行区 dispatch 不回归。 | 只在单 shape 突出时先作为特化策略。 |
| P5: 流水 | steady-state chunk time 下降，资源没爆。 | 降 stage、拆 CTA、缩短 lifetime，必要时放弃双 WG。 |
| P6: 端到端 | K1+K2 目标矩阵领先，HBM 中间量可接受。 | 先压 intermediate，再定框架接入节奏。 |

## 11. 框架接入笔记

KDA state 是固定大小 recurrent state，不要硬塞进 PagedAttention 的 token KV block。框架里应该是每个 layer cache 自己选择 paged KV 或 recurrent state。

单卡：

- head 是全局 H。
- state_cache 保存完整 state。
- prefill 返回 final_state。
- decode 原地更新 slot 对应 state。

TP：

- 每个 rank 使用 $H_{local}$。
- 每 rank 保存本地 state shard。
- collective 放在 projection 周围。
- KDA 主体保持本地计算。

LoRA：

- adapter 影响 Q/K/V/gate/out projection。
- KDA kernel 只看投影后的 tensor。
- mixed-adapter batching 不进入 kernel specialization。

## 12. 风险清单

| 风险 | 表现 | 先怎么处理 |
| --- | --- | --- |
| K1 不是 Tensor Core bound | QK 很快，三角处理和 elementwise 主导。 | 分段 profile；只对大 GEMM 强上 WGMMA。 |
| K2 双 WG 资源爆 | register/SMEM 高，active CTA/SM 下降。 | 先 V64 双 CTA；双 WG 放后面。 |
| V split 重复读取太贵 | Q/K/metadata bytes 翻倍，L2 miss 后优势变小。 | dispatch；查 L2 locality；高并行度走 V128。 |
| K1->K2 intermediate 太大 | K2 单独很好，端到端优势被 HBM 中间量吃掉。 | 最小化 intermediate；改 store layout；最后才考虑 fusion。 |
| CuTe DSL corner case | 窄 WGMMA、pipeline API、代码生成出问题。 | 固定版本；保留 C++ CuTe 小核对照。 |
| 长序列 state 漂 | 短序列正确，chunk 多了误差指数增长。 | 逐 chunk 比 state；查 log-gamma、subchunk 边界和累加 dtype。 |
| 过早框架化 | 时间耗在 scheduler/cache/LoRA，kernel 指标还没定。 | P3/P4/P6 指标稳定后再进 P7。 |

## 13. 交付物

这些交付物是给我自己判断进度的，不是外部汇报件。

| 交付物 | 内容 | 完成定义 |
| --- | --- | --- |
| D1 数学与接口 | Chunk64/Subchunk16 公式、K1 intermediate、K2 recurrence、shape/layout | reference 可独立实现并通过小 shape。 |
| D2 CuTe DSL 骨架 | SM90 layout、TMA、pipeline、WGMMA helper、launch/profile harness | 固定版本可复现编译运行和 profile。 |
| D3 K1 kernel | correctness 版 + 后续流水版 | 中间量测试全过，分段性能清楚。 |
| D4 K2 kernel | V128、V64、dispatch、pipeline | 有稳定胜区和明确 dispatch 边界。 |
| D5 单卡 backend | prefill/decode、varlen、chunked prefill、state cache | 真实模型请求能跑。 |
| D6 性能报告 | latency、吞吐、资源、shape 热图 | 能回答哪里快、为什么快、哪里不该启用。 |
| D7 扩展项 | LoRA、TP | 不改 KDA kernel API，数值与单卡一致。 |

## 14. 每次优化前后固定流程

1. 先比中间数值，不先看最终 output。
2. 确认生成的 WGMMA/TMA 指令和目标 shape。
3. 看 register、SMEM、active CTA/SM。
4. 分离 pipeline fill/drain 和 steady-state chunk time。
5. 比较 HBM/L2 bytes，尤其是 V split 重复读取和 intermediate store。
6. 最后才看端到端 tokens/s、TTFT、scheduler 影响。

## 15. 当前下一步

下一步从 P0 开始，不写 kernel：

1. 在单独 reference 文件里固定 Chunk64/Subchunk16 数学。
2. 定义 K1 intermediate 的最小字段集合。
3. 写小 shape reference test，覆盖 1、16、17、63、64、65、128。
4. 再开始 CuTe DSL TMA + WGMMA 小核。

只要 P0 没完成，任何关于流水、双 WG、nano-vLLM 接入的工作都算跑偏。
