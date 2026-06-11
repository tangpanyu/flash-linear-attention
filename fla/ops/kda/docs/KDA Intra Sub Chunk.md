# KDA Intra Sub Chunk

这份文档解释 `chunk_kda_fwd_kernel_intra_sub_chunk`。它是 `chunk_kda_fwd_intra` 在 `safe_gate=True` 时使用的第一个子 kernel，用来计算一个 chunk 内 diagonal sub-chunk 的 `Aqk` 和 `Akkd`。

## 1. 调用位置

`chunk_kda_fwd_intra` 的前向路径可以简化成：

```text
safe_gate=True:
  chunk_kda_fwd_kernel_intra_sub_chunk
  -> chunk_kda_fwd_kernel_inter_solve_fused
  -> recompute_w_u_fwd_kda_kernel

safe_gate=False:
  chunk_kda_fwd_kernel_intra_token_parallel
  -> chunk_kda_fwd_kernel_inter_solve_fused
  -> recompute_w_u_fwd_kda_kernel
```

所以 `chunk_kda_fwd_kernel_intra_sub_chunk` 只替换了默认路径里的 token-parallel intra kernel。它负责每个 16-token sub-chunk 内部的计算；sub-chunk 之间的 off-diagonal 项和整个 chunk 的 merge/solve 仍然交给 `chunk_kda_fwd_kernel_inter_solve_fused`。

## 2. 输入输出 Shape

Python wrapper 中相关 shape 是：

```text
q:     [B, T, H, K]
k:     [B, T, H, K]
g:     [B, T, HV, K]
beta:  [B, T, HV]
Aqk:   [B, T, HV, BT]
Akkd:  [B, T, HV, BC]
```

其中：

```text
BT = chunk_size，当前只支持 32 或 64
BC = 16
NC = ceil(BT / BC)
BK = next_power_of_2(K)
```

`g` 已经不是 raw gate logits，而是 chunk-local cumsum 后、乘过 `RCP_LN2` 的累计 log2-space decay。因此后续使用：

$$
2^{g_i-g_j}
$$

等价于数学推导里的：

$$
\exp(g^{math}_i-g^{math}_j)
$$

## 3. Grid 和 Program 粒度

kernel grid 是：

```text
(NT, NC, B * HV)
```

三个 program id 分别表示：

```text
i_t:  第几个 chunk
i_i:  chunk 内第几个 16-token sub-chunk
i_bh: batch 和 value head 合并后的索引
```

每个 Triton program 处理一个 value head 的一个 `[BC, BK]` token-channel tile，并输出一个 `[BC, BC]` diagonal block。

在 GVA 下，`q/k` 的 head 数是 `H`，`v/g/beta` 的 head 数是 `HV`。kernel 用：

```text
i_h = i_hv // (HV // H)
```

把 value head 映射回对应的 qk head。

## 4. 它计算的两个矩阵

令当前 sub-chunk 的 token 下标是 $r,i$，都在同一个 16-token block 内。这个 kernel 计算：

$$
Aqk_{r,i}
=
\operatorname{scale}\cdot q_r^\top\left(2^{g_r-g_i}\odot k_i\right),
\qquad r\ge i
$$

以及 raw key-key lower-triangular block：

$$
Akk^{raw}_{r,i}
=
\beta_r\cdot k_r^\top\left(2^{g_r-g_i}\odot k_i\right),
\qquad r>i
$$

注意 `Aqk` 保留 diagonal，因为当前 token 可以读到当前写入；`Akk` 是 strict lower triangular，因为 triangular solve 里 diagonal 由单位矩阵提供。

## 5. 为什么减去 `b_gn`

代码没有直接算 `2 ** (g_r - g_i)`。它先取当前 sub-chunk 中间附近的一行 gate：

```text
g_ref = g[min(BC // 2, valid_len - 1)]
```

然后拆成：

$$
2^{g_r-g_i}
=
2^{g_r-g_{ref}}\cdot 2^{g_{ref}-g_i}
$$

代码里的对应关系是：

```text
b_gm = b_g - b_gn
b_gq = exp2(b_gm)
b_gk = exp2(-b_gm)
```

随后：

```text
b_Aqk = dot(q * b_gq, transpose(k * b_gk))
b_Akk = dot(k * b_gq, transpose(k * b_gk)) * beta[:, None]
```

这样做的目的主要是数值稳定。`safe_gate=True` 时每步 gate 被限制在 `[lower_bound, 0)`，在 16-token sub-chunk 内用中间位置做 reference，可以让指数差值更靠近 0，降低 `exp2` 溢出或下溢风险。

## 6. Mask 规则

计算完 `[BC, BC]` dense block 后，kernel 用两个 mask 截断：

```text
Aqk: r >= i
Akk: r > i
```

也就是：

$$
Aqk=\operatorname{Tril}(Aqk^{dense})
$$

$$
Akk^{raw}=\operatorname{StrictTril}(Akk^{dense})
$$

然后写入：

```text
Aqk[:, i_i * BC : (i_i + 1) * BC]
Akkd[:, 0 : BC]
```

这里传进 kernel 的参数名叫 `Akk`，但在 wrapper 里实际传入的是 `Akkd`，也就是 diagonal 16x16 block 的 fp32 临时 buffer。

## 7. Diagonal Block 内 Solve

写入 raw `Akk` 后，kernel 立刻在同一个 `[BC, BC]` block 上做 forward substitution：

```text
b_Ai = -b_Akk
...
b_Ai += I
```

数学上它在计算：

$$
Akkd
=
\left(I+\operatorname{StrictTril}(Akk^{raw})\right)^{-1}
$$

这一步只解当前 16-token diagonal block 内部的 unit lower-triangular 系统。跨 sub-chunk 的 block merge 仍然由下一个 kernel `chunk_kda_fwd_kernel_inter_solve_fused` 完成。

因为 `safe_gate=True` 路径已经在这里把 diagonal block 解好了，后面的 `chunk_kda_fwd_kernel_inter_solve_fused` 会通过 `USE_SAFE_GATE=True` 跳过 diagonal block 内部的重复 forward substitution，直接把 `Akkd` 当成已经 solved 的 diagonal inverse 使用。

## 8. Varlen 路径

varlen 时输入被 flatten 成 batch size 1，`chunk_indices` 给出每个 chunk 属于哪个样本：

```text
q/k/g/beta:   [1, total_tokens, ...]
cu_seqlens:   [N + 1]
chunk_indices: [NT, 2]
```

kernel 先从 `chunk_indices` 读出：

```text
[sequence_id, chunk_id_inside_sequence]
```

再用 `cu_seqlens` 得到当前序列的 `bos/eos`。所有 offset 都在该序列内部计算，因此 sub-chunk 不会跨样本读取。
