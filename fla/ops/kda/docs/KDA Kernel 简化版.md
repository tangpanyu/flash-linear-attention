# KDA Kernel 简化版

这份笔记按 `Gated DeltaNet Kernel 简化版.md` 的顺序写，方便逐节对照实现。主线放在 chunk 矩阵计算上，单步公式只用来解释语义：

```text
单步语义 -> chunk 内 decay -> chunk 矩阵主线 -> A/W/U/D 的由来 -> C=2,d=3 数值例子 -> kernel checklist
```

先记住一句话：

```text
KDA 用固定大小的 state S 代替完整 KV cache。
它和 GDN 的主线一样：decay 旧 state，再用 delta rule 擦写。
区别是：GDN 的 decay 是 scalar，KDA 的 decay 是 key/channel-wise vector。
```

本文省略 batch/head/chunk 维度，只看一个 chunk。变量名尽量和 `naive.py` 保持一致。令

$$
q,k,g,w,W,k_{\text{right}}\in\mathbb{R}^{C\times d_k}
$$

$$
v,u,U,D,v_{\text{delta}},o\in\mathbb{R}^{C\times d_v}
$$

$$
S_0,S_C\in\mathbb{R}^{d_k\times d_v}
$$

$$
\beta\in\mathbb{R}^{C},\qquad
A,Akk,B,E,Aqk,L\in\mathbb{R}^{C\times C}
$$

其中：

```text
g:             log-space decay。chunk 代码里会被原地概念上改成 cumsum 后的累计 log decay。
A:             本文和 naive.py 的合并记号，A = L^{-1} Diag(beta)；论文里常记作 M。
Akk:           当前优化代码保存的 L^{-1}；WY kernel 再把 beta 乘到 k/v 上。
B:             未乘 beta 的 raw key-key interaction。
w/W:           naive.py 用 w=A@(exp(g)*k)；优化代码用 w=Akk@(beta*exp(g)*k)。
u/U:           naive.py 用 u=A@v；优化代码用 u=Akk@(beta*v)。
D/v_delta:     本文给有效写入取的名字；代码里在 chunk 循环中写成 v_i = u_i - w_i @ S。
E/Aqk:         chunk 内 query-key 读取矩阵；代码里常叫 Aqk，论文里常记作 E。
k_right:       当前优化代码里叫 kg，对应 (g_last - g).exp() * k。
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

## 3. Chunk 矩阵主线

从 kernel 角度，不要先想单个 token 的转置。把一个 chunk 里的 token 都堆成矩阵：

$$
Q=q,\qquad K=k,\qquad V=v
$$

$$
\Gamma=\exp(g)\in\mathbb{R}^{C\times d_k}
$$

其中第 $r$ 行 $\Gamma_r=\exp(g_r)$ 是从 chunk 起点累乘到位置 $r$ 的 channel-wise decay。

先定义四个矩阵。这里的除法都是 elementwise：

$$
Q_{\gamma}=\Gamma\odot Q\in\mathbb{R}^{C\times d_k}
$$

$$
K_{\gamma}=\Gamma\odot K\in\mathbb{R}^{C\times d_k}
$$

$$
K_{\gamma^{-1}}=K/\Gamma\in\mathbb{R}^{C\times d_k}
$$

$$
K_{\text{right}}=(\Gamma_C/\Gamma)\odot K\in\mathbb{R}^{C\times d_k}
$$

直觉：

```text
Q_gamma:       query 从 chunk 起点传到当前位置时带上的 absolute decay
K_gamma:       key 从 chunk 起点传到当前位置时带上的 absolute decay
K_gamma^{-1}:  用来和 Q_gamma/K_gamma 做矩阵乘法，等价表达 exp(g_r-g_i)
K_right:       当前 chunk 写入传到 chunk 末尾还剩多少
```

注意，$K/\Gamma$ 只是紧凑的数学写法。实际 kernel 里通常不会真的 materialize 除法，而是在构造 score 时用 $\exp(g_r-g_i)$。

### 3.1 三个核心矩阵

第一步，构造 chunk 内 key-key interaction：

$$
B=K_{\gamma}K_{\gamma^{-1}}^\top\in\mathbb{R}^{C\times C}
$$

逐元素就是：

$$
B_{r,i}
=
k_r^\top(\exp(g_r-g_i)\odot k_i)
=
\sum_{c=1}^{d_k}k_{r,c}\exp(g_{r,c}-g_{i,c})k_{i,c}
$$

因为是 causal recurrence，后面只用 $i<r$ 的严格下三角部分。

第二步，用 lower-triangular solve 吸收 chunk 内 delta-rule 依赖：

$$
A=
\left[
I+\operatorname{StrictTril}\left(\operatorname{Diag}(\beta)B\right)
\right]^{-1}
\operatorname{Diag}(\beta)
\in\mathbb{R}^{C\times C}
$$

这个 $A$ 对应论文里的 $M$。`naive_chunk_kda` 会复用变量名 `A`：一开始 `A` 放 raw $B$，
做完三角求解后再把 $\beta$ 合进去。当前优化代码则把 $L^{-1}$ 保存为 `Akk`，由 WY kernel
另外把 $\beta$ 乘到 $k/v$ 上。

第三步，构造 chunk 内 query-key 读取矩阵：

$$
E=\operatorname{Tril}\left(Q_{\gamma}K_{\gamma^{-1}}^\top\right)\in\mathbb{R}^{C\times C}
$$

逐元素就是：

$$
E_{r,i}
=
\begin{cases}
q_r^\top(\exp(g_r-g_i)\odot k_i),& r\ge i\\
0,& r<i
\end{cases}
$$

代码里这个 $E$ 通常叫 `Aqk`。

### 3.2 W/U/D，再输出和更新 state

有了 $A$ 以后，先算两个 chunk 表示：

$$
W=AK_{\gamma}\in\mathbb{R}^{C\times d_k}
$$

$$
U=AV\in\mathbb{R}^{C\times d_v}
$$

再得到真正有效写入，也就是代码里的 `v_i`，这里可以将3个`gemm`转为2个：

$$
\boxed{
D=U-WS_0=A(V-K_{\gamma}S_0)
}
$$

本文后面把这个 $D$ 也叫 $v_{\text{delta}}$：

$$
D=v_{\text{delta}}\in\mathbb{R}^{C\times d_v}
$$

然后输出矩阵一次算完：

$$
\boxed{
O=Q_{\gamma}S_0+ED
}
$$

最后更新 chunk 末尾 state：

$$
\boxed{
S_C=\operatorname{Diag}(\Gamma_C)S_0+K_{\text{right}}^\top D
}
$$

只看 shape，整个 kernel 骨架是：

$$
B:[C,d_k][d_k,C]\to[C,C]
$$

$$
A:[C,C]
$$

$$
W:[C,C][C,d_k]\to[C,d_k]
$$

$$
U:[C,C][C,d_v]\to[C,d_v]
$$

$$
D:[C,d_v]-[C,d_k][d_k,d_v]\to[C,d_v]
$$

$$
O:[C,d_k][d_k,d_v]+[C,C][C,d_v]\to[C,d_v]
$$

$$
S_C:[d_k,d_v]+[d_k,C][C,d_v]\to[d_k,d_v]
$$

这就是矩阵形式的主线：

```text
Gamma = exp(cumsum(g_raw))
Q_gamma = Gamma * Q
K_gamma = Gamma * K
K_inv   = K / Gamma

B = K_gamma @ K_inv.T
A = inverse_lower(I + strict_lower(diag(beta) @ B)) @ diag(beta)

W = A @ K_gamma
U = A @ V
D = U - W @ S0

E = tril(Q_gamma @ K_inv.T)
O = Q_gamma @ S0 + E @ D

K_right = (Gamma[-1] / Gamma) * K
S_C = Gamma[-1][:, None] * S0 + K_right.T @ D
```

和 GDN 简化版的一一对应关系是：

```text
D:   真正要写入/读取的 pseudo-value，KDA 代码里叫 v_i
O:   旧 state 贡献 + chunk 内新写入贡献
S_C: decay 旧 state + 写入当前 chunk
```

KDA 和 GDN 的关键差异只在 score 构造。GDN 的 decay 是 scalar，所以可以写成：

$$
(QK^\top)\odot\Gamma_{\text{scalar}}
$$

KDA 的 decay 是 channel-wise，所以必须把 decay 放进 dot product 里面：

$$
Q_{\gamma}K_{\gamma^{-1}}^\top
$$

也就是每个 score 都是：

$$
\sum_c q_{r,c}\exp(g_{r,c}-g_{i,c})k_{i,c}
$$

## 4. A、W、U、D 为什么这样定义

如果只写 kernel，上面第 3 节已经够用。本节只是把 $A,W,U,D$ 和单步 delta rule 对齐。

单步里有效写入是：

$$
v_{\text{delta},r}=\beta_r(v_r-\bar S_r^\top k_r)
$$

但 $\bar S_r$ 已经包含同一个 chunk 中前面 token 的写入，所以第 $r$ 行 $D_r$ 依赖 $D_1,\dots,D_{r-1}$。把这个依赖展开成矩阵行形式，可以写成：

$$
D_r
=
\beta_r v_r
-
\beta_r (\Gamma_r\odot k_r)^\top S_0
-
\beta_r\sum_{i<r}B_{r,i}D_i
$$

把所有 $r$ 堆起来：

$$
D
=
\operatorname{Diag}(\beta)V
-
\operatorname{Diag}(\beta)K_{\gamma}S_0
-
\operatorname{StrictTril}(\operatorname{Diag}(\beta)B)D
$$

移项：

$$
\left[
I+\operatorname{StrictTril}(\operatorname{Diag}(\beta)B)
\right]D
=
\operatorname{Diag}(\beta)V
-
\operatorname{Diag}(\beta)K_{\gamma}S_0
$$

所以：

$$
D
=
\left[
I+\operatorname{StrictTril}(\operatorname{Diag}(\beta)B)
\right]^{-1}
\operatorname{Diag}(\beta)V
-
\left[
I+\operatorname{StrictTril}(\operatorname{Diag}(\beta)B)
\right]^{-1}
\operatorname{Diag}(\beta)K_{\gamma}S_0
$$

把公共部分命名成 $A$：

$$
A=
\left[
I+\operatorname{StrictTril}(\operatorname{Diag}(\beta)B)
\right]^{-1}
\operatorname{Diag}(\beta)
$$

就得到：

$$
U=AV
$$

$$
W=AK_{\gamma}
$$

$$
D=U-WS_0
$$

这也解释了代码里的三行：

```python
w = A @ (g.exp() * k)
u = A @ v
v_i = u_i - w_i @ S
```

直觉：

```text
U:  当前 chunk 的 value 经过三角解之后，会写进去什么
W:  当前 chunk 的 key/erase 方向经过三角解之后，会从旧 state 读出什么
D:  扣掉旧 state 已有内容后的有效写入；代码变量名是 v_i
```

这里的 inverse 不是通用矩阵求逆，而是 unit lower-triangular forward substitution。实现里推荐按 Appendix C 的方式直接构造元素：

$$
B_{r,i}=\operatorname{dot}(k_r\odot\exp(g_r-g_i),k_i)
$$

$$
E_{r,i}=\operatorname{dot}(q_r\odot\exp(g_r-g_i),k_i)
$$

## 5. C=2,d=3：按 KDA 算子流程走一遍

这一节严格沿着 `chunk_kda_fwd` 的 forward 主路径走：

```text
Step 0  入口预处理
Step 1  chunk-local gate cumsum
Step 2  intra-chunk：同时构造 Aqk 和 raw Akk
Step 3  triangular solve，并生成 WY 中间量 w/u/qg/kg
Step 4  inter-chunk recurrent scan，得到 h/v_new/final_state
Step 5  单独决定哪些值保存、哪些值 backward 重算
Step 6  output kernel 合并 old-state 与 intra-chunk 输出
Step 7  用逐 token recurrent reference 交叉检查
```

这个例子故意取两个非零位置不重叠的 key，以便手算。因此会出现 $B_{2,1}=0$。这是特例，不是 KDA
的一般结论；对普通 dense key，channel-wise decay 可能会让原本正交的 key 不再满足加权正交。

### 5.1 Step 0：入口预处理

取：

$$
C=2,\qquad d_k=d_v=3,\qquad S_0=I_3
$$

$$
\beta=
\begin{bmatrix}
0.8 & 0.5
\end{bmatrix}
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

真实入口还可能做：

```text
q/k:   可选 L2 norm
beta:  可选 sigmoid
g:     可选 gate activation
q:     乘 scale = d_k^{-1/2}
```

为了只突出 chunk 主线，本例假设这些入口操作已经完成。因此这里写出的 $q$ 已经包含 `scale` 的语义，$\beta$
也是激活后的值。

### 5.2 Step 1：计算 chunk-local cumulative gate

raw channel-wise decay 取：

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

算子首先沿 chunk 内 token 维做：

$$
g=\operatorname{cumsum}(g^{raw})
$$

所以：

$$
\exp(g_1)=
\begin{bmatrix}
0.5 & 0.8 & 1
\end{bmatrix}
$$

$$
\exp(g_2)
=
\exp(g^{raw}_1)\odot\exp(g^{raw}_2)
=
\begin{bmatrix}
0.125 & 0.4 & 0.75
\end{bmatrix}
$$

记：

$$
\Gamma=\exp(g)=
\begin{bmatrix}
0.5 & 0.8 & 1\\
0.125 & 0.4 & 0.75
\end{bmatrix}
$$

$$
\Gamma_C=\Gamma_2=
\begin{bmatrix}
0.125 & 0.4 & 0.75
\end{bmatrix}
$$

真实 Triton 路径会把 cumulative gate 转成 log2-space 并使用 `exp2`；这里继续使用 $\exp$，数学含义不变。

### 5.3 Step 2：intra-chunk 同时构造 `Aqk` 和 raw `Akk`

`chunk_kda_fwd_intra` 在同一阶段生成：

```text
Aqk:      query-key causal 读取矩阵，保留 diagonal
Akk_raw:  key-key causal 依赖矩阵，只保留 strict lower
```

为了方便手算，先构造两个辅助矩阵。absolute-decayed key 为：

$$
K_\gamma
=
\Gamma\odot K
=
\begin{bmatrix}
0.5 & 0 & 0\\
0 & 0.4 & 0
\end{bmatrix}
$$

inverse-decayed key 为：

$$
K_{\gamma^{-1}}
=
K/\Gamma
=
\begin{bmatrix}
2 & 0 & 0\\
0 & 2.5 & 0
\end{bmatrix}
$$

这里的 $\gamma^{-1}$ 是对 gate 逐元素取倒数，不是对 $K$ 求矩阵逆：

$$
K_\gamma,K_{\gamma^{-1}}\in\mathbb{R}^{2\times3},
\qquad
K_{\gamma^{-1}}^\top\in\mathbb{R}^{3\times2}
$$

真实 kernel 不会 materialize $K/\Gamma$，而是直接使用 $\exp(g_r-g_i)$；这里写出来只是为了展开矩阵乘法。

#### 5.3.1 构造 raw `Akk`

先看未乘 $\beta$ 的 key-key interaction：

$$
B
=
K_\gamma K_{\gamma^{-1}}^\top
=
\begin{bmatrix}
0.5 & 0 & 0\\
0 & 0.4 & 0
\end{bmatrix}
\begin{bmatrix}
2 & 0\\
0 & 2.5\\
0 & 0
\end{bmatrix}
=
\begin{bmatrix}
1 & 0\\
0 & 1
\end{bmatrix}
$$

对 causal triangular solve，只保留 $i<r$ 的元素。长度 $C=2$ 时，strict lower 中只有 $B_{2,1}$：

$$
\exp(g_2-g_1)
=
\frac{\Gamma_2}{\Gamma_1}
=
\begin{bmatrix}
0.25 & 0.5 & 0.75
\end{bmatrix}
$$

$$
B_{2,1}
=
k_2^\top\left(\exp(g_2-g_1)\odot k_1\right)
=
\begin{bmatrix}0&1&0\end{bmatrix}
\begin{bmatrix}0.25\\0\\0\end{bmatrix}
=
0
$$

代码中进入 triangular solve 前的 raw `Akk` 已经乘了当前行的 $\beta_r$：

$$
Akk^{raw}
=
\operatorname{StrictTril}\left(\operatorname{Diag}(\beta)B\right)
=
\begin{bmatrix}
0 & 0\\
\beta_2B_{2,1} & 0
\end{bmatrix}
=
\begin{bmatrix}
0 & 0\\
0 & 0
\end{bmatrix}
$$

为什么不用其他元素：

```text
diagonal:       triangular system 的 diagonal 由 I 提供
upper triangle: 未来 token 不能影响过去 token
strict lower:   过去 token 的写入对当前 correction 的依赖
```

#### 5.3.2 构造 `Aqk`

带 absolute decay 的 query 为：

$$
Q_\gamma
=
\Gamma\odot Q
=
\begin{bmatrix}
0.5 & 0.8 & 0.5\\
0.125 & -0.4 & 1.5
\end{bmatrix}
$$

先算 dense query-key interaction：

$$
Q_\gamma K_{\gamma^{-1}}^\top
=
\begin{bmatrix}
0.5 & 0.8 & 0.5\\
0.125 & -0.4 & 1.5
\end{bmatrix}
\begin{bmatrix}
2 & 0\\
0 & 2.5\\
0 & 0
\end{bmatrix}
=
\begin{bmatrix}
1 & 2\\
0.25 & -1
\end{bmatrix}
$$

例如左下角为：

$$
Aqk_{2,1}
=
q_2^\top\left(\exp(g_2-g_1)\odot k_1\right)
=
\begin{bmatrix}1&-1&2\end{bmatrix}
\begin{bmatrix}0.25\\0\\0\end{bmatrix}
=
0.25
$$

`Aqk` 保留 diagonal，但清掉未来位置：

$$
Aqk
=
\operatorname{Tril}\left(Q_\gamma K_{\gamma^{-1}}^\top\right)
=
\begin{bmatrix}
1 & 0\\
0.25 & -1
\end{bmatrix}
$$

两个 mask 的区别是：

$$
Akk^{raw}:\ r>i,
\qquad
Aqk:\ r\ge i
$$

### 5.4 Step 3：triangular solve 和 WY 中间量

#### 5.4.1 区分代码的 `Akk` 与数学记号 $A$

代码解 unit lower-triangular 系统：

$$
L=I+Akk^{raw},
\qquad
Akk=L^{-1}
$$

本例中：

$$
L=Akk=
\begin{bmatrix}
1&0\\
0&1
\end{bmatrix}
$$

三角求解的结果只有一个：

$$
Akk=L^{-1}
$$

接下来的区别只是“把 $\beta$ 放在哪里乘”。当前优化代码保存 `Akk`，在 WY kernel 中计算：

$$
W=Akk\left(\operatorname{Diag}(\beta)K_\gamma\right),
\qquad
U=Akk\left(\operatorname{Diag}(\beta)V\right)
$$

`naive.py` 为了写得紧凑，先定义一个合并矩阵：

$$
A=Akk\operatorname{Diag}(\beta)
$$

然后写成：

$$
W=AK_\gamma,
\qquad
U=AV
$$

所以 $A$ 不是另一次三角求解，也不是 `Akk` 的别名；它只是把 $\beta$ 提前合进 `Akk` 后得到的数学简写。

所以本例中：

$$
A
=
\begin{bmatrix}
0.8&0\\
0&0.5
\end{bmatrix}
$$

因此本例虽然 `Akk` 是单位阵，合并了 $\beta$ 的 $A$ 却不是单位阵。两条路径最终算出的 $W/U$
完全一样。

#### 5.4.2 生成 `w` 和 `u`

真实 WY kernel 计算：

$$
W
=
Akk\left(\operatorname{Diag}(\beta)K_\gamma\right)
=
\begin{bmatrix}
0.4&0&0\\
0&0.2&0
\end{bmatrix}
$$

展开第一式：

$$
W
=
\begin{bmatrix}
1&0\\
0&1
\end{bmatrix}
\begin{bmatrix}
0.8&0\\
0&0.5
\end{bmatrix}
\begin{bmatrix}
0.5&0&0\\
0&0.4&0
\end{bmatrix}
$$

同理：

$$
U
=
Akk\left(\operatorname{Diag}(\beta)V\right)
=
\begin{bmatrix}
1.6&0.8&0.4\\
0.5&1.5&-0.5
\end{bmatrix}
$$

它们等价于合并记号下的 $W=AK_\gamma$ 和 $U=AV$。

#### 5.4.3 生成 `qg` 和 `kg`

同一个 WY 阶段还会按需 materialize：

$$
qg=Q_\gamma=
\begin{bmatrix}
0.5&0.8&0.5\\
0.125&-0.4&1.5
\end{bmatrix}
$$

代码里的 `kg` 不是 $K_\gamma$，而是本文的 $K_{\text{right}}$：

$$
kg_i
=
\exp(g_C-g_i)\odot k_i
=
\frac{\Gamma_C}{\Gamma_i}\odot k_i
$$

$$
\frac{\Gamma_C}{\Gamma_1}
=
\begin{bmatrix}
0.25&0.5&0.75
\end{bmatrix},
\qquad
\frac{\Gamma_C}{\Gamma_2}
=
\begin{bmatrix}
1&1&1
\end{bmatrix}
$$

所以：

$$
kg=K_{\text{right}}
=
\begin{bmatrix}
0.25&0&0\\
0&1&0
\end{bmatrix}
$$

也可以写成 $kg=\Gamma_C\odot K_{\gamma^{-1}}$，其中 $\Gamma_C$ 沿 token 行广播。

### 5.5 Step 4：inter-chunk recurrent state scan

`chunk_gated_delta_rule_fwd_h` 接收 `kg/w/u/g`，对 chunk 逐个扫描，产生：

```text
h:           每个 chunk 开始时的 state
v_new:       扣除旧 state 读取后的有效写入
final_state: 最后一个 chunk 更新后的 state；仅在请求时返回
```

这里只有一个 chunk，所以：

$$
h=S_0=I_3
$$

有效写入为：

$$
v_{\text{new}}
=
U-Wh
$$

$$
v_{\text{new}}
=
\begin{bmatrix}
1.6&0.8&0.4\\
0.5&1.5&-0.5
\end{bmatrix}
-
\begin{bmatrix}
0.4&0&0\\
0&0.2&0
\end{bmatrix}
I_3
=
\begin{bmatrix}
1.2&0.8&0.4\\
0.5&1.3&-0.5
\end{bmatrix}
$$

它就是前文的 $D$，也是 `naive_chunk_kda` 循环中的 `v_i`：

$$
D=v_{\text{new}}
$$

旧 state 衰减到 chunk 末尾：

$$
\operatorname{Diag}(\Gamma_C)h
=
\begin{bmatrix}
0.125&0&0\\
0&0.4&0\\
0&0&0.75
\end{bmatrix}
$$

chunk 内有效写入传播到末尾：

$$
kg^\top v_{\text{new}}
=
\begin{bmatrix}
0.25&0\\
0&1\\
0&0
\end{bmatrix}
\begin{bmatrix}
1.2&0.8&0.4\\
0.5&1.3&-0.5
\end{bmatrix}
=
\begin{bmatrix}
0.3&0.2&0.1\\
0.5&1.3&-0.5\\
0&0&0
\end{bmatrix}
$$

所以：

$$
S_C
=
\operatorname{Diag}(\Gamma_C)h+kg^\top v_{\text{new}}
=
\begin{bmatrix}
0.425&0.2&0.1\\
0.5&1.7&-0.5\\
0&0&0.75
\end{bmatrix}
$$

### 5.6 Step 5：单独处理“保存还是重算”

这一步不改变数学结果，只决定 forward 结束后的显存占用。先区分两种“保存”：

```text
forward 阶段间交接：后面的 forward kernel 还要立刻消费
save_for_backward：forward 全部结束后仍保留，供 backward 使用
```

本例新产生的中间量及用途如下：

| 值 | 本例结果 | forward 中的用途 |
|---|---:|---|
| `Aqk` | $\begin{bmatrix}1&0\\0.25&-1\end{bmatrix}$ | output 读取 chunk 内新写入 |
| `Akk` | $I_2$ | 生成 WY；backward 的 triangular 路径 |
| `w` | $W$ | state scan 计算 $v_{\text{new}}$ |
| `u` | $U$ | state scan 计算 $v_{\text{new}}$ |
| `qg` | $Q_\gamma$ | 主要供 backward 的 state 梯度路径使用 |
| `kg` | $K_{\text{right}}$ | state scan 更新 chunk 末尾 state |
| `h` | $S_0$ | output 读取当前 chunk 的起始 state |
| `v_new` | $D$ | output 读取 chunk 内有效写入 |

当前代码的保存策略是：

- `Aqk` 和 `Akk` 会保留给 backward。
- `disable_recompute=False` 是省显存路径：`qg` 在 forward 中不 materialize；`w/u/kg/v_new`
  用完后不保留，通常也不保留 `h`，backward 再重算。若请求 `return_intermediate_states`，`h` 例外。
- `disable_recompute=True` 是省计算路径：保存 `w/u/qg/kg/v_new/h`，backward 直接复用。
- `final_state` 是可选 API 输出，不等于“必须为 backward 保存的中间量”。
- $K_{\gamma^{-1}}$ 和完整 $B$ 只为讲解而 materialize，真实 kernel 不把它们作为独立 tensor 保存。

因此，不要把“数学推导中写出来的量”和“实现必须长期保存的 tensor”混为一谈。

### 5.7 Step 6：output kernel 合并两条输出路径

state scan 完成后，output kernel 执行：

$$
o
=
(\Gamma\odot Q)h+Aqk\,v_{\text{new}}
$$

数学上可以把 $\Gamma\odot Q$ 简写成 $qg$，但当前 forward output kernel 通常直接读取 $q,g$
现场计算，不依赖一个已保存的 `qg` tensor。

第一项读取 chunk 开始时的旧 state：

$$
(\Gamma\odot Q)h
=
\begin{bmatrix}
0.5&0.8&0.5\\
0.125&-0.4&1.5
\end{bmatrix}
$$

第二项读取 chunk 内的新写入：

$$
Aqk\,v_{\text{new}}
=
\begin{bmatrix}
1&0\\
0.25&-1
\end{bmatrix}
\begin{bmatrix}
1.2&0.8&0.4\\
0.5&1.3&-0.5
\end{bmatrix}
=
\begin{bmatrix}
1.2&0.8&0.4\\
-0.2&-1.1&0.6
\end{bmatrix}
$$

两项相加：

$$
o
=
\begin{bmatrix}
1.7&1.6&0.9\\
-0.075&-1.5&2.1
\end{bmatrix}
$$

这与真实 `chunk_kda_fwd` 一致：先运行 state scan 得到 `h/v_new`，再由 output kernel 使用
`q/g/Aqk/h/v_new` 生成最终输出。

### 5.8 Step 7：逐 token recurrent reference 交叉检查

这一步不是 chunk kernel 主流程，而是独立验证。

token 1：

$$
\bar S_1
=
\operatorname{Diag}(\exp(g^{raw}_1))S_0
=
\begin{bmatrix}
0.5&0&0\\
0&0.8&0\\
0&0&1
\end{bmatrix}
$$

$$
v_{\text{delta},1}
=
0.8\left(
\begin{bmatrix}2&1&0.5\end{bmatrix}
-
\begin{bmatrix}0.5&0&0\end{bmatrix}
\right)
=
\begin{bmatrix}1.2&0.8&0.4\end{bmatrix}
$$

$$
S_1
=
\bar S_1+k_1v_{\text{delta},1}^\top
=
\begin{bmatrix}
1.7&0.8&0.4\\
0&0.8&0\\
0&0&1
\end{bmatrix}
$$

$$
o_1=S_1^\top q_1=
\begin{bmatrix}
1.7&1.6&0.9
\end{bmatrix}
$$

token 2：

$$
\bar S_2
=
\operatorname{Diag}(\exp(g^{raw}_2))S_1
=
\begin{bmatrix}
0.425&0.2&0.1\\
0&0.4&0\\
0&0&0.75
\end{bmatrix}
$$

$$
v_{\text{delta},2}
=
0.5\left(
\begin{bmatrix}1&3&-1\end{bmatrix}
-
\begin{bmatrix}0&0.4&0\end{bmatrix}
\right)
=
\begin{bmatrix}0.5&1.3&-0.5\end{bmatrix}
$$

$$
S_2
=
\bar S_2+k_2v_{\text{delta},2}^\top
=
\begin{bmatrix}
0.425&0.2&0.1\\
0.5&1.7&-0.5\\
0&0&0.75
\end{bmatrix}
=
S_C
$$

$$
o_2=S_2^\top q_2=
\begin{bmatrix}
-0.075&-1.5&2.1
\end{bmatrix}
$$

逐 token 得到的两行有效写入就是 $v_{\text{new}}=D$ 的两行，$o_1$、$o_2$ 和 $S_C$ 也都与 chunk
算子结果一致。

## 6. Kernel Checklist

最小实现流程：

```text
0. 入口：可选 q/k norm、beta sigmoid、gate activation、q scale
1. g = chunk_local_cumsum(g)
2. intra: Aqk = tril(relative_decay_qk, diagonal=0)
3. intra: Akk_raw = strict_tril(diag(beta) @ relative_decay_kk)
4. solve: Akk = inverse_lower(I + Akk_raw)
5. WY: w = Akk @ (beta[:, None] * exp(g) * k)
6. WY: u = Akk @ (beta[:, None] * v)
7. WY: kg = exp(g_last - g) * k; qg 按 recompute 策略决定是否 materialize
8. state scan: h = chunk-start state; v_new = u - w @ h
9. state scan: final_state = exp(g_last)[:, None] * h + kg.T @ v_new
10. 单独执行 save/recompute 策略
11. output: o = (exp(g) * q) @ h + Aqk @ v_new
```

实现注意：

```text
q 要和论文伪代码一样乘 d_k^{-0.5}，或者和 reference 保持同一约定。
inverse_lower 不是真的求通用逆，而是 unit lower-triangular forward substitution。
k/exp(g) 只是数学写法，实际优先用 exp(g_r-g_i)。
q_abs、k_abs、k_right 都可以不 materialize，在 load 时乘 decay。
优化代码保存 Akk=L^{-1}；本文合并记号 A=Akk@diag(beta)。
代码里的 w/u/v_new 分别对应本文的 W/U/D；v_new 不是原始输入 v。
代码里的 kg 对应 k_right，不是 exp(g)*k。
```

和论文对应：

```text
Eq. 1: 单步 recurrent
Eq. 6: B 和 A/M 的 lower-triangular solve
Eq. 7: W = A(exp(g)*K), U = A V
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

主复杂度没有质变，构造 `B`/`E` 的主项仍然是 $O(C^2d_k)$。KDA 多出来的主要是 $O(Cd_k)$ 的 channel-wise scaling、`exp` 和中间值管理；换来的是同一个 head 内可以同时有长记忆 channel 和短记忆 channel。
