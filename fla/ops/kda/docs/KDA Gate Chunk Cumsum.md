# KDA Gate Chunk Cumsum

这份文档专门解释 `kda_gate_chunk_cumsum_vector_kernel`。它处理的是 `use_gate_in_kernel=True` 时的 gate 路径：输入 `g` 还不是 log-space decay，而是模型层产生的 raw gate logits。

## 1. Gate 从哪里来

在 `KimiDeltaAttention` 里，`f_proj` 生成 KDA gate 的 raw logits：

$$
g^{logit}\in\mathbb{R}^{B\times T\times HV\times K}
$$

其中 $HV$ 是 value head 数，$K$ 是 key head dim。KDA 的 gate 是 per value-head、per key-channel 的向量 gate，所以 `f_proj` 的输出维度是：

$$
gate\_dim=HV\cdot K
$$

reshape 后得到：

$$
g^{logit}_{b,t,h,c},\qquad h\in[0,HV),\quad c\in[0,K)
$$

这和 GDN 的常见 scalar gate 不同。GDN 通常每个 head 一个 gate；KDA 每个 value head 的每个 key channel 都有自己的 gate，所以同一个 head 内可以同时存在长记忆 channel 和短记忆 channel。

## 2. Raw Logits 到 Log-Space Decay

`A_log` 和 `dt_bias` 把 raw logits 变成真正参与 recurrent decay 的 log gate：

$$
A\_log\in\mathbb{R}^{HV}
$$

$$
dt\_bias\in\mathbb{R}^{HV\cdot K}
$$

计算时 `dt_bias` 按 head 和 channel 视作：

$$
dt\_bias\in\mathbb{R}^{HV\times K}
$$

如果没有传 `dt_bias`，可以把它当成 0。因此每个元素先得到：

$$
x_{b,t,h,c}=g^{logit}_{b,t,h,c}+dt\_bias_{h,c}
$$

普通 gate 路径使用：

$$
g^{raw}_{b,t,h,c}=-\exp(A\_log_h)\operatorname{softplus}(x_{b,t,h,c})
$$

`safe_gate=True` 且传入 `lower_bound` 时使用：

$$
g^{raw}_{b,t,h,c}=lower\_bound\cdot\sigma(\exp(A\_log_h)x_{b,t,h,c})
$$

这两个式子的输出都是非正数，仍然是 log-space decay。普通路径中 $\exp(A\_log_h)$ 控制该 value head 的 decay 速度，`softplus` 保证 decay 幅度为正，再加负号得到 log decay。`dt_bias` 是每个 head-channel 的偏置，用来给不同 channel 不同的初始时间尺度。

`safe_gate` 路径把输出限制在：

$$
[lower\_bound,0)
$$

所以每步实际 decay $\exp(g^{raw})$ 被限制在：

$$
[\exp(lower\_bound),1)
$$

## 3. Kernel 融合了什么

`kda_gate_chunk_cumsum_vector_kernel` 融合了三件事：

```text
raw gate logits -> log-space decay -> chunk-local cumsum -> log2 scale
```

对应 Python 语义可以写成：

```python
# g_logit: [B, T, HV, K]
# A_log:   [HV]
# dt_bias: [HV * K] or None

x = g_logit
if dt_bias is not None:
    x = x + dt_bias.view(HV, K)
if lower_bound is None:
    g_raw = -A_log.exp().view(HV, 1) * softplus(x)
else:
    g_raw = lower_bound * sigmoid(A_log.exp().view(HV, 1) * x)

g_cumsum = chunk_local_cumsum(g_raw, chunk_size=BT)
g_out = g_cumsum * RCP_LN2
```

最后乘 `RCP_LN2` 是因为后续 Triton kernel 里大量使用 `exp2`。也就是说数学里的 $\exp(g_r-g_i)$ 在实现里通常变成：

$$
2^{g^{out}_r-g^{out}_i}
$$

其中：

$$
g^{out}=g^{cumsum}\log_2(e)
$$

所以数学含义没有变化。

## 4. Shape 变化

kernel 的 wrapper `kda_gate_chunk_cumsum` 接收的 shape 是：

```text
g / s:      [B, T, H, S]
o:          [B, T, H, S]
A_log:      [H]
dt_bias:    [H * S] 或 None
```

在 KDA 调用里：

```text
H = HV
S = K
BT = chunk_size
```

因此输入输出实际就是：

```text
g_logit:    [B, T, HV, K]
g_out:      [B, T, HV, K]
A_log:      [HV]
dt_bias:    [HV * K]
```

shape 不变，语义发生变化：

```text
进入 kernel 前:
  g 是 raw gate logits，还不能直接作为 decay 使用。

离开 kernel 后:
  g 是每个 chunk 内累计后的 log2-space decay，默认 dtype 是 float32，
  后续 chunk_intra/state/output kernel 会直接消费它。
```

## 5. Triton Program 划分

kernel 的 grid 是：

```text
(ceil(S / BS), NT, B * H)
```

含义分别是：

```text
第 0 维: key/channel 维度分块，每个 program 处理 BS 个 channel。
第 1 维: chunk 维度，每个 program 处理一个长度 BT 的 token 块。
第 2 维: batch 和 head 合并后的维度。
```

每个 Triton program 处理一个 `[BT, BS]` tile：

$$
s_{tile}\in\mathbb{R}^{BT\times BS}
$$

并在 token 维做 chunk-local prefix sum：

$$
o_{r,c}=\sum_{j=1}^{r}g^{raw}_{j,c}
$$

这里的 $r$ 只在当前 chunk 内计数。换到全局序列时，每个 chunk 都重新从 0 开始累计；跨 chunk 的 recurrent state decay 由后续 `chunk_gated_delta_rule_fwd_h` 使用 chunk 末端累计值处理。

## 6. Varlen 路径

varlen 路径下输入会被 flatten 成 batch size 1：

```text
g:             [1, total_tokens, HV, K]
cu_seqlens:    [N + 1]
chunk_indices: [NT, 2]
```

`chunk_indices` 的每一行表示：

```text
[sequence_id, chunk_id_inside_sequence]
```

kernel 先用它找到当前 chunk 所属序列的 `bos/eos`，再在该序列内部做 chunk-local cumsum。这样 prefix sum 不会跨不同样本串起来。
