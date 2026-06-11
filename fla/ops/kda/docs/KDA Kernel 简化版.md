# KDA Kernel 简化版

这份笔记按 `Gated DeltaNet Kernel 简化版.md` 的顺序写，方便逐节对照实现：

```text
单步语义 -> chunk 内 decay -> chunkwise 主公式 -> A/w/u/v_i 的区别 -> C=2,d=3 数值例子 -> kernel checklist
```

先记住一句话：

```text
KDA 用固定大小的 state S 代替完整 KV cache。
它和 GDN 的主线一样：decay 旧 state，再用 delta rule 擦写。
区别是：GDN 的 decay 是 scalar，KDA 的 decay 是 key/channel-wise vector。
```

本文省略 batch/head/chunk 维度，只看一个 chunk。变量名尽量和 `naive.py` 保持一致。令

$$
q,k,g,w,k_{\text{right}}\in\mathbb{R}^{C\times d_k}
$$

$$
v,u,v_{\text{delta}},o\in\mathbb{R}^{C\times d_v}
$$

$$
S_0,S_C\in\mathbb{R}^{d_k\times d_v}
$$

$$
\beta\in\mathbb{R}^{C},\qquad
A,Aqk\in\mathbb{R}^{C\times C}
$$

其中：

```text
g:             log-space decay。chunk 代码里会被原地概念上改成 cumsum 后的累计 log decay。
A:             代码里的块内三角解矩阵；论文里常记作 M。
w:             代码里的 w = A @ (g.exp() * k)。
u:             代码里的 u = A @ v。
v_delta:       本文给有效写入取的名字；代码里在 chunk 循环中写成 v_i = u_i - w_i @ S。
Aqk:           代码里的 chunk 内 query-key 读取矩阵；论文里常记作 E。
k_right:       代码里不单独命名，对应 (g_last - g).exp() * k。
```

入口张量的完整形状是：

$$
q,k\in\mathbb{R}^{B\times T\times H\times K},\quad
v,o\in\mathbb{R}^{B\times T\times HV\times V},\quad
g\in\mathbb{R}^{B\times T\times HV\times K},\quad
S\in\mathbb{R}^{B\times HV\times K\times V}
$$

进入 `naive_chunk_kda` 后会先 rearrange 并把 `q/k` 扩展到 value-head 维：

$$
q,k,g\in\mathbb{R}^{B\times HV\times NT\times BT\times K},\quad
v,o\in\mathbb{R}^{B\times HV\times NT\times BT\times V}
$$

KDA kernel 的入口是：

```text
输入: q, k, v, g, beta, initial_state
输出: o, final_state
```

decode 时一次只处理一个 token，直接用 recurrent update。prefill/training 时一次处理一段 prompt，所以要按 chunk 并行。

论文 Appendix C 的伪代码里还会做：

$$
q\leftarrow q\cdot d_k^{-1/2}
$$

本文后面的公式默认 $q$ 已经完成这个 scale；写 reference 时要和这个约定保持一致。

state 方向固定为：

$$
S_t\in\mathbb{R}^{d_k\times d_v},\qquad
o_t=S_t^\top q_t\in\mathbb{R}^{d_v}
$$

## 1. 单步语义

对一个 token，

$$
q_t,k_t,g_t\in\mathbb{R}^{d_k},\quad
v_t,v_{\text{delta},t},o_t\in\mathbb{R}^{d_v},\quad
\beta_t\in\mathbb{R}
$$

KDA 的单步公式是论文 Eq. 1。为了和代码一致，本文用 $\exp(g_t)$ 表示 per-channel decay：

$$
S_t
=
\left(I-\beta_t k_tk_t^\top\right)\operatorname{Diag}(\exp(g_t))S_{t-1}
+
\beta_tk_tv_t^\top
$$

和 GDN 一样，把它拆成“先 decay，再 delta 擦写”更好懂：

$$
\bar S_t=\operatorname{Diag}(\exp(g_t))S_{t-1}
$$

这个 $\bar S_t$ 只是把原公式里的 decayed old state 单独命名。把它代回原式：

$$
S_t
=
\left(I-\beta_t k_tk_t^\top\right)\bar S_t
+
\beta_tk_tv_t^\top
$$

展开括号：

$$
S_t
=
\bar S_t
-
\beta_t k_tk_t^\top\bar S_t
+
\beta_tk_tv_t^\top
$$

其中：

$$
k_t^\top\bar S_t
=
(\bar S_t^\top k_t)^\top
$$

所以定义：

$$
\hat v_t=\bar S_t^\top k_t
$$

它表示“decayed old state 在当前 key $k_t$ 上已经读出来的 old value”。于是擦除项可以写成：

$$
\beta_t k_tk_t^\top\bar S_t
=
\beta_t k_t\hat v_t^\top
$$

再和写入项合并：

$$
S_t
=
\bar S_t
+
\beta_tk_t(v_t-\hat v_t)^\top
$$

因此定义 correction，也就是 chunk 代码里最终的 `v_i` 语义：

$$
v_{\text{delta},t}=\beta_t(v_t-\hat v_t)
$$

就得到：

$$
S_t=\bar S_t+k_tv_{\text{delta},t}^\top
$$

$$
o_t=S_t^\top q_t
$$

最小 decode reference：

```python
# q_i, k_i, g_i: [dk]
# v_i:           [dv]
# b_i:           scalar
# S:             [dk, dv]

S = S * g_i[:, None].exp()
v_delta = b_i * (v_i - S.T @ k_i)
S = S + k_i[:, None] @ v_delta[None, :]
o_i = S.T @ q_i
```

这里最重要的区别是：

$$
\text{GDN:}\quad \bar S_t=\exp(g_t)S_{t-1}
$$

$$
\text{KDA:}\quad \bar S_t=\operatorname{Diag}(\exp(g_t))S_{t-1}
$$

GDN 是整张 state 共享一个衰减；KDA 是 state 的每个 key/channel 行有自己的衰减。

## 2. Chunk 内 decay

GDN 里 $g_t$ 是 scalar，所以累计 decay 也是 scalar。KDA 里 $g_t$ 是 $d_k$ 维向量，所以累计 decay 也是 $d_k$ 维向量。

论文 Appendix C 通常输入 log decay。进入一个 chunk 后先做：

```python
g = g.cumsum(-2)
```

和代码一致，chunk 入口处的原始 log gate 先记成 $g^{raw}$，执行 `g = g.cumsum(-2)` 后，后文的 $g$ 都表示累计 log gate：

$$
g^{raw}_r=\log\alpha_r
$$

$$
g_r=\sum_{j=1}^{r}g^{raw}_j
$$

$$
\exp(g_r)=\prod_{j=1}^{r}\alpha_j
$$

其中：

$$
g_r,\exp(g_r)\in\mathbb{R}^{d_k}
$$

第 $i$ 个 token 的写入传到第 $r$ 个 token，需要 channel-wise 相对衰减：

$$
\rho_{r,i}=\exp(g_r-g_i)\in\mathbb{R}^{d_k}
$$

直觉：

$$
\exp(g_r):\text{ chunk 起点到 }r\text{ 的每个 channel 的 decay}
$$

$$
\rho_{r,i}:\text{ token }i\text{ 的写入传到 }r\text{ 时，每个 channel 还剩多少}
$$

`naive_chunk_kda` 的输入 `g` 已经是 log gate，所以直接 `g = g.cumsum(-2)`。`naive_recurrent_kda` 逐 token 更新，不需要 cumsum，循环里直接用 `g_i.exp()`。

## 3. Chunkwise 主公式

先定义三组和代码表达式对应的 decay 后张量：

$$
q_{\text{abs}}=\exp(g)\odot q
$$

$$
k_{\text{abs}}=\exp(g)\odot k
$$

$$
k_{\text{right},i}=\rho_{C,i}\odot k_i=\exp(g_C-g_i)\odot k_i
$$

KDA 的 chunk 内 query-key score 不是 GDN 那种 scalar gate 形式：

$$
(qk^\top)\odot\Gamma
$$

而是：

$$
Aqk=\operatorname{Tril}\left(q_{\text{abs}}\left(\frac{k}{\exp(g)}\right)^\top\right)
$$

也就是逐元素写成：

$$
Aqk_{r,i}=
\begin{cases}
q_r^\top(\rho_{r,i}\odot k_i),& r\ge i\\
0,& r<i
\end{cases}
$$

chunkwise 主公式是：

$$
\boxed{
v_{\text{delta}}=u-wS_0
}
$$

$$
\boxed{
o=q_{\text{abs}}S_0+Aqk\,v_{\text{delta}}
}
$$

$$
\boxed{
S_C=\operatorname{Diag}(\exp(g_C))S_0+k_{\text{right}}^\top v_{\text{delta}}
}
$$

这三行和 GDN 简化版一一对应：

```text
v_delta: 先算真正要写入/读取的 pseudo-value，代码里叫 v_i
o:       旧 state 贡献 + chunk 内新写入贡献
S_C:     decay 旧 state + 写入当前 chunk
```

注意，$k/\exp(g)$ 是数学写法。代码不会 materialize 这个除法，而是在构造每个 score 时用：

$$
\exp(g_r-g_i)
$$

## 4. A、w、u、v_i 为什么这样定义

单步里：

$$
v_{\text{delta},r}=\beta_r(v_r-\bar S_r^\top k_r)
$$

但 $\bar S_r$ 已经包含前面 token 的写入，所以 $v_{\text{delta},r}$ 依赖 $v_{\text{delta},1},\dots,v_{\text{delta},r-1}$。

把前面的展开代进去，可以得到：

$$
v_{\text{delta},r}
=
\beta_r v_r
-
\beta_r S_0^\top(\exp(g_r)\odot k_r)
-
\beta_r
\sum_{i<r}
v_{\text{delta},i}\,k_r^\top(\rho_{r,i}\odot k_i)
$$

这里出现 key-key interaction：

$$
B_{r,i}=k_r^\top(\rho_{r,i}\odot k_i)
$$

代码里先把这个 key-key interaction 临时放进 `A`：

```python
A[..., i] = torch.einsum('... c d, ... d -> ... c', k * (g - g_i).exp(), k_i)
```

也就是：

$$
A^{raw}_{r,i}=k_r^\top(\exp(g_r-g_i)\odot k_i)
$$

矩阵写法对应论文 Eq. 6。论文通常把这个 raw key-key 矩阵记成 $B$：

$$
B=(\exp(g)\odot k)\left(\frac{k}{\exp(g)}\right)^\top
$$

然后代码继续复用 `A` 这个变量，用 lower-triangular solve 吸收 chunk 内依赖。求完之后的 `A` 才对应论文里的 $M$：

$$
A=
\left[
I+\operatorname{StrictTril}\left(\operatorname{Diag}(\beta)B\right)
\right]^{-1}
\operatorname{Diag}(\beta)
$$

这里的 inverse 不是通用矩阵求逆，而是 unit lower-triangular forward substitution。

因此代码里的三行：

```python
w = A @ (g.exp() * k)
u = A @ v
v_i = u_i - w_i @ S
```

对应数学写法：

$$
w=A(\exp(g)\odot k)
$$

$$
u=Av
$$

$$
v_{\text{delta}}=u-wS_0
$$

直觉：

```text
u:        只看当前 chunk 的 value，会写进去什么
w:        当前 chunk 会从旧 state 里擦掉/扣掉什么
v_delta:  扣掉旧 state 影响后的有效写入 pseudo-value；代码变量名是 v_i
```

为什么 `v_i` 要减 `w_i @ S`？因为 delta rule 不是盲写 $v_r$，而是写：

$$
v_r-\text{old value}
$$

旧 value 来自 chunk 开始前的 state $S_0$，所以要从 `u` 里扣掉 `w @ S0`。

实现里推荐按 Appendix C 的伪代码直接算：

$$
A^{raw}_{r,i}=\operatorname{dot}(k_r\odot\exp(g_r-g_i),k_i)
$$

$$
Aqk_{r,i}=\operatorname{dot}(q_r\odot\exp(g_r-g_i),k_i)
$$

## 5. C=2,d=3 数值例子

取：

$$
C=2,\qquad d_k=d_v=3,\qquad S_0=I_3
$$

沿用 GDN 例子里的 $q,k,v,\beta$：

$$
\beta_1=0.8,\quad \beta_2=0.5
$$

$$
q=
\begin{bmatrix}
1 & 1 & 0.5\\
1 & -1 & 2
\end{bmatrix},
\quad
k=
\begin{bmatrix}
1 & 0 & 0\\
0 & 1 & 0
\end{bmatrix},
\quad
v=
\begin{bmatrix}
2 & 1 & 0.5\\
1 & 3 & -1
\end{bmatrix}
$$

KDA 的 raw channel-wise decay 取：

$$
\exp(g^{raw}_1)=
\begin{bmatrix}
0.5 & 0.8 & 1
\end{bmatrix}
$$

$$
\exp(g^{raw}_2)=
\begin{bmatrix}
0.25 & 0.5 & 0.75
\end{bmatrix}
$$

所以执行 `g = g.cumsum(-2)` 后：

$$
\exp(g_1)=
\begin{bmatrix}
0.5 & 0.8 & 1
\end{bmatrix},
\qquad
\exp(g_2)=
\begin{bmatrix}
0.125 & 0.4 & 0.75
\end{bmatrix}
$$

因为 $k_1\perp k_2$，所以三角解之后的 `A` 等于 $\operatorname{Diag}(\beta)$。absolute-decayed key 是：

$$
k_{\text{abs}}=
\exp(g)\odot k
=
\begin{bmatrix}
0.5 & 0 & 0\\
0 & 0.4 & 0
\end{bmatrix}
$$

于是：

$$
w=A k_{\text{abs}}
=
\begin{bmatrix}
0.4 & 0 & 0\\
0 & 0.2 & 0
\end{bmatrix}
$$

$$
u=Av
=
\begin{bmatrix}
1.6 & 0.8 & 0.4\\
0.5 & 1.5 & -0.5
\end{bmatrix}
$$

$$
v_{\text{delta}}=u-wS_0
=
\begin{bmatrix}
1.2 & 0.8 & 0.4\\
0.5 & 1.3 & -0.5
\end{bmatrix}
$$

旧 state 读取：

$$
q_{\text{abs}}=
\exp(g)\odot q
=
\begin{bmatrix}
0.5 & 0.8 & 0.5\\
0.125 & -0.4 & 1.5
\end{bmatrix}
$$

chunk 内读取矩阵：

$$
Aqk=
\begin{bmatrix}
1 & 0\\
0.25 & -1
\end{bmatrix}
$$

输出：

$$
o=
q_{\text{abs}}S_0+Aqk\,v_{\text{delta}}
=
\begin{bmatrix}
1.7 & 1.6 & 0.9\\
-0.075 & -1.5 & 2.1
\end{bmatrix}
$$

state update 需要：

$$
k_{\text{right}}=
\begin{bmatrix}
0.25 & 0 & 0\\
0 & 1 & 0
\end{bmatrix}
$$

因此：

$$
S_C
=
\operatorname{Diag}(\exp(g_2))S_0+k_{\text{right}}^\top v_{\text{delta}}
=
\begin{bmatrix}
0.425 & 0.2 & 0.1\\
0.5 & 1.7 & -0.5\\
0 & 0 & 0.75
\end{bmatrix}
$$

这和逐 token recurrent reference 完全一致。

## 6. Kernel Checklist

最小实现流程：

```text
1. g = g.cumsum(-2)
2. A_raw[r,i] = dot(k[r] * exp(g[r]-g[i]), k[i])
3. A = inverse_lower(I + strict_lower(diag(beta) @ A_raw)) @ diag(beta)
4. w = A @ (exp(g) * k)
5. u = A @ v
6. v_delta = u - w @ S
7. Aqk[r,i] = dot(q[r] * exp(g[r]-g[i]), k[i]), i <= r
8. o = (exp(g) * q) @ S + Aqk @ v_delta
9. k_right = exp(g_C - g) * k
10. S = exp(g_C)[:, None] * S + k_right.T @ v_delta
```

实现注意：

```text
q 要和论文伪代码一样乘 d_k^{-0.5}，或者和 reference 保持同一约定。
inverse_lower 不是真的求通用逆，而是 unit lower-triangular forward substitution。
k/exp(g) 只是数学写法，实际优先用 exp(g_r-g_i)。
q_abs、k_abs、k_right 都可以不 materialize，在 load 时乘 decay。
代码复用 A：前半段 A 是 A_raw/B，lower-triangular solve 后 A 才是论文里的 M。
代码里的 v_i 是 v_delta，不是原始输入 v。
```

和论文对应：

```text
Eq. 1: 单步 recurrent
Eq. 6: A_raw/B 和 A/M 的 lower-triangular solve
Eq. 7: w = A(exp(g)*k), u = A v
Eq. 8: state update
Eq. 9: o = old state 贡献 + chunk 内贡献
Appendix C Listing 1: g.cumsum 和 exp(g_r-g_i)
```

## 7. 和 GDN 的关系

KDA 可以看成把 GDN 的 scalar decay 推广成 channel-wise decay。

GDN 里：

$$
\Gamma_{r,i}=\exp(g_r-g_i)\in\mathbb{R}
$$

所以：

$$
q_r^\top(\Gamma_{r,i}k_i)=\Gamma_{r,i}(q_r^\top k_i)
$$

它可以先算 $qk^\top$，再乘 $C\times C$ 的 scalar mask。

KDA 里：

$$
\rho_{r,i}=\exp(g_r-g_i)\in\mathbb{R}^{d_k}
$$

所以：

$$
q_r^\top(\rho_{r,i}\odot k_i)
=
\sum_c q_{r,c}\rho_{r,i,c}k_{i,c}
$$

因为每个 channel 的 $\rho_{r,i,c}$ 都可能不同，它不能从求和里提出去。只有当所有 channel 的衰减都相同，KDA 才退化成 GDN 那种 scalar mask。

实现开销上，可以这样简化理解：

```text
GDN:
  exp(g) 是 [C]
  score = (q @ k.T) * Gamma
  state decay 是整张 state 乘一个数

KDA:
  exp(g) 是 [C, dk]
  score = dot(q[r] * exp(g[r]-g[i]), k[i])
  state decay 是每一行乘一个数
```

主复杂度没有质变，构造 `A_raw`/`Aqk` 的主项仍然是 $O(C^2d_k)$。KDA 多出来的主要是 $O(Cd_k)$ 的 channel-wise scaling、`exp` 和中间值管理；换来的是同一个 head 内可以同时有长记忆 channel 和短记忆 channel。
