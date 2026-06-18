# KDA 精度处理

这份文档记录 FLA 中 KDA 为了数值稳定和训练可用性做的精度处理。

核心结论：

KDA 不是全程 fp32。它把 gate、指数、state 累积、三角求解这些敏感部分尽量放在 fp32；大规模矩阵乘仍然尽量走 bf16/fp16 Tensor Core 路径。

## 1. 为什么 KDA 需要额外精度处理

KDA 的 recurrent 形式可以写成：

$$
S_t=\exp(g_t)\odot S_{t-1}+\beta_t k_t\left(v_t-k_t^\top S_{t-1}\right)^\top
$$

其中：

$$
g_t\in\mathbb{R}^{HV\times K}
$$

KDA 的 gate 是 per value-head、per key-channel 的向量 gate。它比 scalar gate 更灵活，但也带来几个数值风险。

这里的 $g_t$ 是 log gate。若把真实 decay 写成：

$$
\alpha_t=\exp(g_t)
$$

那么：

$$
g_t=\log\alpha_t
$$

因为 KDA 中 $g_t\le 0$，所以：

$$
0<\alpha_t\le 1
$$

第一，gate 会在时间维上累积。

如果直接在 normal-space 里维护：

$$
\Gamma_t=\prod_{i\le t}\alpha_i
$$

长序列下很容易下溢。实现里不会直接存 $\Gamma_t$，而是存 log-space cumulative gate。

也就是存：

$$
G_t=\sum_{i\le t}g_i
$$

因为：

$$
\Gamma_t=\exp(G_t)
$$

第二，chunk 内大量出现指数差：

$$
\frac{\Gamma_i}{\Gamma_j}
$$

在 log-space 中它会变成：

$$
\exp(G_i-G_j)
$$

如果 $G_i-G_j$ 太大或太小，就会 overflow 或 underflow。

第三，KDA 有 delta correction：

$$
v_t-k_t^\top S_{t-1}
$$

这里存在两个风险：

- $k_t^\top S_{t-1}$ 是累积 state 上的投影，误差会随时间传递。
- 如果 $v_t$ 和 $k_t^\top S_{t-1}$ 接近，会出现减法抵消。

第四，chunk 化之后需要求解下三角系统：

$$
\left(I+\operatorname{StrictTril}(\operatorname{Diag}(\beta)B)\right)^{-1}
$$

三角求解对舍入误差比较敏感，特别是 diagonal block 内部。

所以 KDA 的实现不能只依赖 bf16 直接跑完；必须把关键路径挑出来做 fp32 或范围控制。

## 2. Q/K L2 Norm

在 `ChunkKDAFunction.forward` 入口，如果启用：

$$
use\_qk\_l2norm\_in\_kernel=True
$$

会先对 $q,k$ 做 L2 norm：

$$
\hat q=\frac{q}{\sqrt{\sum_i q_i^2+\epsilon}}
$$

$$
\hat k=\frac{k}{\sqrt{\sum_i k_i^2+\epsilon}}
$$

对应代码：

$$
q,k \leftarrow l2norm(q),l2norm(k)
$$

`l2norm` kernel 中会把输入转成 fp32 计算平方和和倒数标准差：

$$
rstd=\frac{1}{\sqrt{\sum_i x_i^2+\epsilon}}
$$

然后再把输出写回目标 dtype。

这样做的原因是控制：

$$
q^\top k
$$

和：

$$
k^\top k
$$

的尺度。否则如果 $q,k$ 的范数漂移，chunk 内的 $A_{qk}$、$A_{kk}$ 会变大，后面的下三角求解和 recurrent 更新都会更难稳定。

代价是多一个 norm kernel，并且 backward 需要走 `l2norm_bwd`。

## 3. Gate 激活用 fp32

真实模型里传入 KDA 的 $g$ 通常还是 raw logits：

$$
g^{logit}_{b,t,h,c}
$$

KDA 需要把它变成 log-space decay。

也就是说，kernel 里算出来的不是 $\alpha_t$，而是：

$$
g_t=\log\alpha_t
$$

普通路径：

$$
x=g^{logit}+dt\_bias
$$

$$
g^{raw}=-\exp(A\_log)\operatorname{softplus}(x)
$$

safe gate 路径：

$$
g^{raw}=lower\_bound\cdot\sigma(\exp(A\_log)x)
$$

这两条路径都会在 kernel 里把 raw logits、`dt_bias`、`A_log` 转成 fp32 后再算。

原因是这里同时包含：

- $\exp(A\_log)$
- $\operatorname{softplus}(x)$
- $\sigma(x)$
- 后续 cumsum

这些操作如果直接用 bf16，误差会被时间维累积。

## 4. Gate 存成 log2-space cumulative value

gate 激活后不会直接以 normal-space 乘法形式保存，而是做 chunk-local cumsum：

$$
G_t=\sum_{i=0}^{t}g_i
$$

随后乘上：

$$
RCP\_LN2=\log_2(e)
$$

得到：

$$
g^{out}_t=G_t\log_2(e)
$$

这样后续数学上的：

$$
\exp(G_a-G_b)
$$

可以实现成：

$$
2^{g^{out}_a-g^{out}_b}
$$

也就是代码里的 `exp2`。

这么做有两个好处：

- log-space cumsum 避免直接连乘造成 underflow。
- `exp2` 通常比 `exp` 更适合 GPU kernel 路径。

注意：这不是改变数学含义，只是把底从 $e$ 换成了 $2$。

## 5. safe_gate 的作用和代价

普通 gate 是：

$$
g^{raw}=-\exp(A\_log)\operatorname{softplus}(x)
$$

它只保证：

$$
g^{raw}<0
$$

但没有固定下界。如果某些 token 的 gate 很负，那么 chunk 内的：

$$
2^{g_i-g_j}
$$

可能范围很大。

`safe_gate=True` 时改用：

$$
g^{raw}=lower\_bound\cdot\sigma(\exp(A\_log)x)
$$

因此：

$$
g^{raw}\in[lower\_bound,0)
$$

代码里要求：

$$
-5\le lower\_bound<0
$$

这样每一步 decay 被限制在：

$$
\exp(lower\_bound)\le \exp(g^{raw})<1
$$

后果是：

- 指数范围更可控。
- diagonal sub-chunk 可以走更规则的 Tensor Core 路径。
- 但 gate 表达能力变窄，不再是原来的 $-\exp(A)\operatorname{softplus}(x)$ 形式。

所以 `safe_gate` 是性能和数值范围更友好的一条路径，不是完全等价替换。

## 6. 指数差值使用 reference subtraction

chunk 内经常需要：

$$
\exp_2(g_i-g_j)
$$

直接算这个差值可能范围很大。实现中常选一个 reference，比如当前 sub-chunk 的边界 $g_n$，把它拆成：

$$
\exp_2(g_i-g_j)
=
\exp_2(g_i-g_n)\exp_2(g_n-g_j)
$$

代码里会看到类似：

$$
b\_gq=\exp_2(g_i-g_n)
$$

$$
b\_gk=\exp_2(g_n-g_j)
$$

然后把它们分别乘到 $q/k$ 或 $k/k$ 两侧。

这样数学上不变，但每个指数输入更靠近 0，可以降低 overflow / underflow 风险。

这个处理在 `chunk_intra.py` 里大量出现，是 KDA intra 计算里最重要的数值稳定技巧之一。

## 7. diagonal block 的 Akkd 用 fp32

KDA chunk 内需要构造并求解 lower-triangular 的 $A_{kk}$。

实现里完整的 `Akk` 最终存成输入 dtype：

$$
Akk.dtype=k.dtype
$$

但 diagonal 16x16 block 会先用单独的 fp32 buffer：

$$
Akkd.dtype=float32
$$

原因是 diagonal block 需要做 forward substitution / triangular inverse。这个过程是递推的：

$$
L^{-1}_{i,:}
$$

依赖前面已经算出的行。如果这里用 bf16 存中间结果，误差会在 block 内逐行传播。

因此 diagonal block 先用 fp32 临时保存，之后再写入完整 `Akk`。

后果是：

- 增加一块临时 fp32 buffer。
- 降低 diagonal solve 的局部误差。
- 最终消费时仍然可以回到低精度主路径。

## 8. WY 中 beta 的放置位置

WY 重计算里有：

$$
u=L^{-1}(\operatorname{Diag}(\beta)V)
$$

$$
w=L^{-1}(\operatorname{Diag}(\beta)(\exp(g)\odot K))
$$

代码选择把 $\beta$ 乘到右操作数上：

$$
\beta V
$$

$$
\beta K
$$

再进入 `tl.dot`。

数学上也可以写成：

$$
(L^{-1}\operatorname{Diag}(\beta))V
$$

但在 bf16/Tensor Core 下，这两个写法不会 bitwise 一致。

原因是浮点计算里：

$$
fl(A\cdot fl(\beta V))
\neq
fl(fl(A\beta)\cdot V)
$$

也就是说，$\beta$ 乘在哪一边会改变哪个 operand 先被 bf16 量化。

所以 FLA 当前实现保留原有放置方式，不随意把 $\beta$ 挪到 $A$ 上。相关误差分析见 `KDA WY Beta Scaling 精度分析.md`。

## 9. kg 预先 materialize

KDA 的 gate 是 per-key-channel 的，所以 inter-chunk scan 需要的右侧 key 不是原始 $K$，而是：

$$
K_{\text{right},t}
=
K_t\odot \exp_2(g_{\text{last}}-g_t)
$$

代码里把它 materialize 成 `kg`：

$$
kg_t=K_t\odot 2^{g_{\text{last}}-g_t}
$$

这样后面的 recurrent scan 里直接使用：

$$
kg^\top v_{\text{new}}
$$

而不需要在 scan kernel 里重新处理每个 token 的 per-key gate 比例。

它的意义有两个：

- 保持 inter scan 公式简单：

$$
S_{\text{next}}=DS+kg^\top(U-WS)
$$

- 避免在 state scan 的递推里混入更多指数差值计算。

代价是需要额外 materialize 一个 `kg` tensor。如果 `disable_recompute=False`，forward 结束后可以丢掉，backward 时重算。

## 10. recurrent state 用 fp32 累积

inter-chunk recurrent scan 维护：

$$
S_i\in\mathbb{R}^{K\times V}
$$

kernel 里的寄存器 state 是 fp32：

$$
b\_h.dtype=float32
$$

如果有 `initial_state`，代码也要求它是 fp32：

$$
initial\_state.dtype=float32
$$

`final_state` 也用 fp32 保存。

这是因为 state 是跨 chunk 递推的：

$$
S_{i+1}=D_iS_i+K_i^\top(U_i-W_iS_i)
$$

如果 state 本身用 bf16 累积，误差会跨 chunk 传播。用 fp32 寄存器维护 state，可以把长期递推误差控制住。

但注意：`h` 这个中间 buffer 通常是用 `k.dtype` 分配的。也就是说：

- 寄存器里的 state 累积是 fp32。
- 写到 HBM 的 per-chunk `h` 可能是 bf16。
- 如果 backward 选择 recompute，就会重新从 fp32 `initial_state` 跑一遍 scan。

## 11. beta sigmoid 控制 update 系数范围

入口处如果启用：

$$
use\_beta\_sigmoid\_in\_kernel=True
$$

会先做：

$$
\beta=\sigma(\beta^{raw})
$$

如果：

$$
allow\_neg\_eigval=True
$$

则使用：

$$
\beta=2\sigma(\beta^{raw})
$$

普通路径把 $\beta$ 限制在：

$$
(0,1)
$$

允许负特征值时把范围扩到：

$$
(0,2)
$$

这个处理的目的不是单纯防溢出，而是控制 delta update 的系数尺度：

$$
\beta_t k_t(v_t-k_t^\top S_{t-1})^\top
$$

如果直接用 raw beta，update 强度无界，训练更容易不稳定。

## 12. 为什么不全程 fp32

KDA 的主要计算量来自多个矩阵乘：

$$
QK^\top
$$

$$
K^\top W
$$

$$
AV
$$

$$
AK
$$

这些如果全用 fp32，会明显损失吞吐。FLA 的策略是：

- 标量非线性、cumsum、指数参数、state 累积用 fp32。
- 大的 `tl.dot` 尽量使用低精度 operand 和 Tensor Core。
- 必要时把 fp32 结果 cast 回目标 dtype 写出。

所以它追求的是“关键路径 fp32 + 主体 GEMM 低精度”的折中。

## 13. 这些处理的后果

这些精度处理带来几个直接后果。

第一，数学等价的改写不一定数值等价。

例如：

$$
A(\beta V)
$$

和：

$$
(A\operatorname{Diag}(\beta))V
$$

在实数数学里等价，但在 bf16/Tensor Core 路径里可能有可见误差。

第二，safe gate 和普通 gate 不是同一个模型假设。

safe gate 改变了 gate 的函数形式和范围。它可以换来更稳定的指数范围和更规则的 kernel 路径，但不能认为和普通 gate 完全等价。

第三，保存和 recompute 会影响内存，而不是改变数学公式。

`disable_recompute=False` 时，forward 会丢掉 `w/u/qg/kg/v_new/h` 等中间量，backward 重算它们。这样省显存，但 backward 多算一次。

第四，KDA 的误差分析要看多个指标。

常见需要看：

- max absolute error
- mean absolute error
- relative error
- RMSE
- cosine similarity
- 分位数误差
- 梯度误差
- 训练 loss 是否漂移

单独看 max error 容易被极少数点支配，单独看 mean error又可能掩盖局部尖峰。

## 14. 阅读代码时的检查顺序

看 KDA 精度问题时，建议按这个顺序排查：

1. `chunk.py`

确认是否启用：

$$
use\_qk\_l2norm\_in\_kernel
$$

$$
use\_gate\_in\_kernel
$$

$$
use\_beta\_sigmoid\_in\_kernel
$$

$$
safe\_gate
$$

2. `gate.py`

确认 gate 是普通路径还是 lower-bound 路径，以及输出是否已经是 log2-space cumulative gate。

3. `chunk_intra.py`

看 $Aqk/Akk$ 如何使用 gate difference，是否用了 fp32 `Akkd`。

4. `wy_fast.py`

看 $\beta$ 和 $g$ 乘在了哪个 operand 上，以及是否 materialize `kg`。

5. `common/chunk_delta_h.py`

看 recurrent state 是否用 fp32 累积，`initial_state/final_state/h` 的 dtype 和保存策略是什么。

## 15. 总结

KDA 的精度处理不是一个单点技巧，而是一整套折中：

$$
\text{gate/log/exp/state/solve 用 fp32 或范围控制}
$$

$$
\text{大矩阵乘尽量走低精度 Tensor Core}
$$

$$
\text{中间量能重算就不保存}
$$

这样才能在长序列、向量 gate、chunk 化三角求解、recurrent state 递推同时存在的情况下，保持训练可用，同时不把性能完全打回 fp32。
