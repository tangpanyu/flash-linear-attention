# KDA Kernel 简化版

这份笔记按 `naive_chunk_kda` 的矩阵计算顺序写，目标是帮助写 kernel，而不是从 recurrent 公式慢慢推导。

核心流程是：

```text
reshape/GVA -> g.cumsum -> 构造 key-key 矩阵 B -> triangular solve 得到 A
-> w/u -> chunk 内 Aqk -> v_delta -> output -> state update
```

本文省略 batch、value-head、chunk id，只看一个 chunk。令 chunk size 为 $C$，key dim 为 $d_k$，value dim 为 $d_v$：

$$
q,k,g,w,k_{\text{right}}\in\mathbb{R}^{C\times d_k}
$$

$$
v,u,v_{\Delta},o\in\mathbb{R}^{C\times d_v}
$$

$$
S_0,S_C\in\mathbb{R}^{d_k\times d_v}
$$

$$
\beta\in\mathbb{R}^{C},\qquad
B,A,Aqk\in\mathbb{R}^{C\times C}
$$

其中 `naive_chunk_kda` 里的 `A` 会被复用：一开始概念上是 key-key interaction，最后变成 triangular solve 后的矩阵。为了避免混淆，本文把 raw key-key interaction 记成 $B$，把 solve 后的最终矩阵记成 $A$。

## 1. 入口 Shape

原始输入：

$$
q,k\in\mathbb{R}^{B\times T\times H\times K}
$$

$$
v,o\in\mathbb{R}^{B\times T\times HV\times V}
$$

$$
g\in\mathbb{R}^{B\times T\times HV\times K},\qquad
\beta\in\mathbb{R}^{B\times T\times HV}
$$

`naive_chunk_kda` 先按 chunk reshape：

```python
q, k = rearrange(x, 'b (n c) h ... -> b h n c ...', c=BT)
v, g, beta = rearrange(x, 'b (n c) h ... -> b h n c ...', c=BT)
```

再把 `q/k` 从 qk head 扩展到 value head：

```python
G = HV // H
q = q.repeat_interleave(G, dim=1) * scale
k = k.repeat_interleave(G, dim=1)
```

进入每个 chunk 后：

$$
q,k,g\in\mathbb{R}^{C\times d_k}
$$

$$
v\in\mathbb{R}^{C\times d_v},\qquad
\beta\in\mathbb{R}^{C}
$$

这里的 $q$ 已经乘过 attention scale，也已经按 GVA 扩展到 value-head 维。

## 2. Chunk 内累计 Gate

`naive_chunk_kda` 的第一步是：

```python
g = g.cumsum(-2)
```

原始 per-token log gate 记为 $g^{raw}$，累计后：

$$
g_r=\sum_{t=0}^{r}g^{raw}_t
$$

因此：

$$
\exp(g_r)=\prod_{t=0}^{r}\exp(g^{raw}_t)
$$

同一个 chunk 内，从 token $i$ 的写入传播到 token $r$ 的 channel-wise 相对 decay 是：

$$
\rho_{r,i}=\exp(g_r-g_i)\in\mathbb{R}^{d_k}
$$

kernel 中常用 `exp2`，所以实际 `g` 往往已经乘过 $\log_2(e)$。这时：

$$
\exp(g^{math}_r-g^{math}_i)
=
2^{g^{kernel}_r-g^{kernel}_i}
$$

数学写法不变，只是底层指数函数换成 `exp2`。

## 3. 构造 Key-Key 矩阵 B

`naive_chunk_kda` 里构造 `A` 的第一段是：

```python
A = torch.zeros(*g.shape[:-1], BT)
for i in range(BT):
    k_i = k[..., i, :]
    g_i = g[..., i:i+1, :]
    A[..., i] = torch.einsum('... c d, ... d -> ... c', k * (g - g_i).exp(), k_i)
```

按矩阵记号，这个 raw key-key interaction 是：

$$
B_{r,i}
=
k_r^\top(\exp(g_r-g_i)\odot k_i)
$$

等价矩阵写法是：

$$
B=(\exp(g)\odot k)(\exp(-g)\odot k)^\top
$$

注意这不是 GDN 里的 $(kk^\top)\odot\Gamma$。KDA 的 $\exp(g_r-g_i)$ 是 $d_k$ 维向量，必须进入 dot product 内部：

$$
B_{r,i}
=
\sum_{c=1}^{d_k}k_{r,c}\exp(g_{r,c}-g_{i,c})k_{i,c}
$$

这是写 KDA kernel 时最重要的区别。

## 4. 从 B 到 A：Triangular Solve

`naive_chunk_kda` 接着做：

```python
A = A * beta[..., None]
A = -A.masked_fill(mask, 0)  # mask 是 triu(diagonal=0)
for i in range(1, BT):
    A[..., i, :i] = A[..., i, :i].clone() + (A[..., i, :, None].clone() * A[..., :, :i].clone()).sum(-2)
A = (A + torch.eye(BT)) * beta[..., None, :]
```

第一行把 $\beta_r$ 乘到 row 上：

$$
C_{r,i}=\beta_r B_{r,i}
$$

然后保留 strict lower triangular 并取负：

$$
L=-\operatorname{StrictTril}(C)
$$

for loop 做的是 unit lower-triangular inverse 的 forward substitution：

$$
M=(I+\operatorname{StrictTril}(C))^{-1}
$$

因为 $L=-\operatorname{StrictTril}(C)$，所以代码里的 strict-lower 部分相当于在累积 $M-I$。

最后一行把 $\beta_i$ 乘到 column 上：

$$
A=M\operatorname{Diag}(\beta)
$$

也就是：

$$
\boxed{
A=
\left(I+\operatorname{StrictTril}(\operatorname{Diag}(\beta)B)\right)^{-1}
\operatorname{Diag}(\beta)
}
$$

kernel 里的 `Akk` 最终存的就是这个 $A$。`Akkd` 是 diagonal 16x16 block 的临时 buffer，后续 `inter_solve_fused` 会用它拼出完整 `Akk`。

## 5. 计算 w 和 u

`naive_chunk_kda` 里：

```python
w = A @ (g.exp() * k)
u = A @ v
```

矩阵形式：

$$
w=A(\exp(g)\odot k)
$$

$$
u=Av
$$

shape 是：

$$
w\in\mathbb{R}^{C\times d_k},\qquad
u\in\mathbb{R}^{C\times d_v}
$$

含义：

```text
u: 当前 chunk 的 value 经过 triangular solve 后的写入项。
w: 当前 chunk 对 chunk 初始 state S0 的读取/擦除系数。
```

后面真正用于输出和 state update 的 pseudo-value 是：

$$
v_{\Delta}=u-wS_0
$$

代码对应：

```python
v_i = u_i - w_i @ S
```

## 6. 构造 Chunk 内 Aqk

在每个 chunk 的输出循环中，`naive_chunk_kda` 构造 `Aqk`：

```python
Aqk = torch.zeros(B, HV, BT, BT)
for j in range(BT):
    k_j = k[:, :, i, j]
    g_j = g[:, :, i, j:j+1, :]
    Aqk[..., j] = torch.einsum('... c d, ... d -> ... c', q_i * (g_i - g_j).exp(), k_j)
Aqk = Aqk.masked_fill(mask, 0)  # mask 是 triu(diagonal=1)
```

逐元素：

$$
Aqk_{r,i}
=
q_r^\top(\exp(g_r-g_i)\odot k_i),
\qquad r\ge i
$$

矩阵写法：

$$
Aqk=\operatorname{Tril}\left((\exp(g)\odot q)(\exp(-g)\odot k)^\top\right)
$$

这里 `Tril` 保留 diagonal。因为输出 token $r$ 可以读到当前 token $r$ 的写入。

## 7. Output 和 State Update

对于 chunk $n$，设进入 chunk 前的 state 是 $S_0$。代码：

```python
v_i = u_i - w_i @ S
o[:, :, i] = (q_i * g_i.exp()) @ S + Aqk @ v_i
S = S * rearrange(g_i[:, :, -1].exp(), 'b h k -> b h k 1')
S += rearrange((g_i[:, :, -1:] - g_i).exp() * k_i, 'b h c k -> b h k c') @ v_i
```

矩阵形式：

$$
v_{\Delta}=u-wS_0
$$

$$
o=(\exp(g)\odot q)S_0+Aqk\,v_{\Delta}
$$

令 chunk 最后一个 token 的累计 gate 为 $g_C$，定义：

$$
k_{\text{right},i}=\exp(g_C-g_i)\odot k_i
$$

则 state update 是：

$$
S_C=\operatorname{Diag}(\exp(g_C))S_0+k_{\text{right}}^\top v_{\Delta}
$$

这三行就是写 kernel 时最常用的 chunkwise 主公式：

```text
v_delta = u - w @ S0
o       = q_abs @ S0 + Aqk @ v_delta
S_next  = g_last_exp[:, None] * S0 + k_right.T @ v_delta
```

## 8. 和 Kernel 中间量的对应

`chunk_kda_fwd_intra` 主要负责得到：

```text
Aqk: [B, T, HV, BT]
Akk: [B, T, HV, BT]
w:   [B, T, HV, K]
u:   [B, T, HV, V]
qg:  [B, T, HV, K] 或 None
kg:  [B, T, HV, K]
```

对应本文变量：

```text
Aqk -> Aqk
Akk -> A
w   -> A @ (exp(g) * k)
u   -> A @ v
qg  -> exp(g) * q，用于 disable_recompute=True 时保存
kg  -> exp(g_last - g) * k，用于后续 state update
```

前向 kernel 拆分：

```text
1. intra diagonal:
   safe_gate=False 用 chunk_kda_fwd_kernel_intra_token_parallel
   safe_gate=True  用 chunk_kda_fwd_kernel_intra_sub_chunk

2. inter-sub-chunk + solve:
   chunk_kda_fwd_kernel_inter_solve_fused

3. w/u/qg/kg:
   recompute_w_u_fwd_kda_kernel
```

然后 `chunk_kda_fwd` 会继续调用：

```text
chunk_gated_delta_rule_fwd_h  # 跨 chunk recurrent state
chunk_gla_fwd_o_gk            # 输出 o
```

## 9. Kernel Checklist

按矩阵计算写 kernel 时，可以按这个顺序检查：

```text
1. q/k 是否已经按 GVA 从 H 扩展到 HV，q 是否已经乘 scale。
2. g 是否已经是 chunk-local cumsum；Triton 路径里通常已经是 log2-space。
3. B[r,i] = dot(k[r] * exp(g[r]-g[i]), k[i])。
4. 只保留 B 的 strict lower 部分参与 Akk solve。
5. A = inverse_lower(I + strict_lower(diag(beta) @ B)) @ diag(beta)。
6. w = A @ (exp(g) * k)。
7. u = A @ v。
8. Aqk[r,i] = dot(q[r] * exp(g[r]-g[i]), k[i])，保留 lower 含 diagonal。
9. v_delta = u - w @ S。
10. o = (exp(g) * q) @ S + Aqk @ v_delta。
11. k_right = exp(g_last - g) * k。
12. S = exp(g_last)[:, None] * S + k_right.T @ v_delta。
```

实现注意：

```text
exp(g_r-g_i) 是 channel-wise，不能像 GDN 那样提出 dot product。
不要 materialize exp(g) / exp(-g) 的完整矩阵，kernel 中通常边 load 边乘。
Akk 的 diagonal 来自单位矩阵，不来自 raw B 的 diagonal。
Aqk 保留 diagonal，Akk/B solve 使用 strict lower。
Akkd 只是 diagonal sub-chunk 的临时 fp32 buffer，最终消费的是完整 Akk。
```

## 10. 和 Recurrent 语义的最短连接

如果只想确认 chunk 公式和 recurrent reference 是同一件事，可以记住单步更新：

$$
S_t=\operatorname{Diag}(\exp(g^{raw}_t))S_{t-1}
+\beta_t k_t\left(v_t-S_{t-1}^{\top}\left(\exp(g^{raw}_t)\odot k_t\right)\right)^\top
$$

`naive_chunk_kda` 做的事情就是把一个 chunk 内所有 token 的这种依赖整理成 lower-triangular matrix solve。矩阵 $A$ 吸收了 chunk 内 token 之间的 delta-rule 擦写依赖，`Aqk` 负责 chunk 内新写入对输出的贡献，`w/u` 负责把 chunk 初始 state $S_0$ 的影响扣掉。
