# KDA 算子阅读指南

本文档用于帮助理解 `fla.ops.kda` 中 Kimi Delta Attention (KDA) 算子的实现。建议先从 PyTorch 参考实现和 Python 调用链入手，再进入 Triton kernel。

## 核心概念

KDA 可以看作 Delta Rule 的一个扩展。它和 Gated DeltaNet 的主要区别是 gate 的粒度：

- Gated DeltaNet 的 gate 通常是每个 head 一个标量。
- KDA 的 gate 是每个 value head、每个 key 维度一个向量，形状为 `[B, T, HV, K]`。

参考递推在 [`naive_recurrent_kda`](naive.py) 中：

```text
S <- exp(g) * S + beta * k * (v - k^T S)
o <- q^T S
```

其中：

- `q`, `k`: `[B, T, H, K]`
- `v`: `[B, T, HV, V]`
- `g`: `[B, T, HV, K]`，log-space decay
- `beta`: `[B, T, HV]`
- `S`: `[B, HV, K, V]`

当 `HV > H` 时，KDA 使用 GVA (Grouped Value Attention)，要求 `HV % H == 0`。实现中会把 `q/k` 从 qk head 维度扩展到 value head 维度。

## 推荐阅读顺序

### 1. 先读参考实现

入口文件：

- [`naive.py`](naive.py)

重点函数：

- `naive_recurrent_kda`
- `naive_chunk_kda`

`naive_recurrent_kda` 是最直接的 token-by-token 递推，适合理解数学含义。`naive_chunk_kda` 则展示了 chunk 化之后如何把递推拆成块内矩阵计算和跨块 state 更新。

### 2. 再读模型层调用

入口文件：

- [`../../layers/kda.py`](../../layers/kda.py)

重点位置：

- `KimiDeltaAttention.forward`
- `chunk_kda(...)`
- `fused_recurrent_kda(...)`

真实模型路径里通常会启用：

```text
use_qk_l2norm_in_kernel=True
use_gate_in_kernel=True
use_beta_sigmoid_in_kernel=True
state_v_first=True
```

因此传入 `chunk_kda` 的 `g` 和 `beta` 在常见路径中还是 raw input，gate activation、q/k L2 norm 和 beta sigmoid 会在算子入口附近融合处理。

如果 `use_short_conv=True`，`q/k/v` 在进入 KDA 前还会经过 [`../../modules/conv/short_conv.py`](../../modules/conv/short_conv.py) 中的 `ShortConvolution`。

### 3. 读 Python 算子入口

入口文件：

- [`chunk.py`](chunk.py)

重点函数和类：

- `chunk_kda`
- `ChunkKDAFunction.forward`
- `ChunkKDAFunction.backward`

`chunk_kda` 负责参数校验、默认值处理和 autograd 封装。这里也是理解各种开关的最好位置：

- `use_qk_l2norm_in_kernel`
- `use_gate_in_kernel`
- `use_beta_sigmoid_in_kernel`
- `allow_neg_eigval`
- `safe_gate`
- `lower_bound`
- `state_v_first`
- `cu_seqlens`
- `cp_context`

### 4. 单独读 gate

入口文件：

- [`gate.py`](gate.py)

重点函数：

- `naive_kda_gate`
- `naive_kda_lowerbound_gate`
- `fused_kda_gate`
- `kda_gate_chunk_cumsum`

默认 gate 形式是：

```text
g = -exp(A_log) * softplus(raw_g + dt_bias)
```

当 `safe_gate=True` 时，gate 形式会变为：

```text
g = lower_bound * sigmoid(exp(A_log) * (raw_g + dt_bias))
```

这样可以把 log-space gate 限制在 `[lower_bound, 0)`，并允许部分 kernel 使用更高吞吐的路径。

### 5. 读 chunk 前向拆解

入口文件：

- [`chunk_fwd.py`](chunk_fwd.py)

核心流程：

```text
chunk_kda_fwd
  -> gate activation / chunk cumsum
  -> chunk_kda_fwd_intra
  -> chunk_gated_delta_rule_fwd_h
  -> chunk_gla_fwd_o_gk
```

含义如下：

- gate activation / chunk cumsum：把 log-space gate 转成 chunk 内累计形式。
- `chunk_kda_fwd_intra`：计算块内的 `Aqk`、`Akk`、`w`、`u` 等中间量。
- `chunk_gated_delta_rule_fwd_h`：复用 common delta-rule state kernel，做跨 chunk recurrent state 更新。
- `chunk_gla_fwd_o_gk`：结合 state 和块内 attention 项得到输出。

### 6. 最后读 Triton kernel

入口文件：

- [`chunk_intra.py`](chunk_intra.py)
- [`chunk_intra_token_parallel.py`](chunk_intra_token_parallel.py)
- [`chunk_bwd.py`](chunk_bwd.py)

建议先从 wrapper 读起：

- `chunk_kda_fwd_intra`
- `chunk_kda_bwd_intra`
- `chunk_kda_bwd`

然后再进入具体 Triton kernel：

- `chunk_kda_fwd_kernel_inter_solve_fused`
- `chunk_kda_fwd_kernel_intra_sub_chunk`
- `chunk_kda_bwd_kernel_intra`
- `chunk_kda_bwd_kernel_wy_dqkg_fused`

读 kernel 时可以先跟踪张量含义，不急着逐行看指针偏移：

- `Aqk`: 块内 query-key 相关项，用于输出。
- `Akk`: 块内 key-key 相关项，用于 triangular solve / WY 表示。
- `w`, `u`: WY/chunk 形式中的中间表示。
- `kg`, `qg`: 融合 gate 后的 key/query 表示。
- `h`: chunk 级 recurrent state。
- `v_new`: 结合当前 state 修正后的 value 表示。

## 前向调用链

常见训练路径：

```text
KimiDeltaAttention.forward
  -> chunk_kda
    -> ChunkKDAFunction.forward
      -> l2norm_fwd(q/k)                      # optional
      -> fused_beta_sigmoid(beta)             # optional
      -> chunk_kda_fwd
        -> kda_gate_chunk_cumsum              # if use_gate_in_kernel=True
        -> chunk_kda_fwd_intra
        -> chunk_gated_delta_rule_fwd_h
        -> chunk_gla_fwd_o_gk
```

短序列推理路径可能切到：

```text
KimiDeltaAttention.forward
  -> fused_recurrent_kda
```

对应文件是 [`fused_recurrent.py`](fused_recurrent.py)，它更适合 decode / short sequence inference。

## 反向调用链

反向从 `ChunkKDAFunction.backward` 开始：

```text
ChunkKDAFunction.backward
  -> chunk_kda_bwd
    -> recompute_w_u_fwd                      # 如果没有保存中间量
    -> chunk_kda_bwd_dAv
    -> chunk_gated_delta_rule_bwd_dhu
    -> chunk_kda_bwd_wy_dqkg_fused
    -> chunk_kda_bwd_intra
    -> kda_gate_bwd                           # if use_gate_in_kernel=True
    -> fused_beta_sigmoid_bwd                 # if use_beta_sigmoid_in_kernel=True
    -> l2norm_bwd(q/k)                        # if use_qk_l2norm_in_kernel=True
```

阅读反向时，建议先确认前向保存了哪些 tensor，再看 `disable_recompute` 如何影响内存和计算量。

## 测试入口

入口文件：

- [`../../../tests/ops/test_kda.py`](../../../tests/ops/test_kda.py)

建议重点读：

- `test_naive_chunk`
- `test_fused_recurrent`
- `test_chunk`
- `test_chunk_varlen`
- `test_gate`
- `test_chunk_return_intermediate_states`

这些测试分别覆盖参考实现、chunk 实现、varlen、gate fusion、state layout 和推理中间 state。

## 一句话地图

如果只想快速建立全局结构，可以记住这条线：

```text
naive.py 解释数学
layers/kda.py 展示模型如何调用
chunk.py 是 Python/autograd 入口
gate.py 处理 KDA 特有 gate
chunk_fwd.py 拆前向阶段
chunk_intra.py 和 chunk_bwd.py 才是真正的 Triton 核心
```
