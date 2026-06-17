# KDA CP 之中

这份文档保留 CP forward 需要的公式和语义，不展开 kernel 代码。

核心是：每个 rank 的本地序列段对 state 的作用可以写成仿射变换：

$$
S_{\text{rank,out}}=M_{\text{rank}}S_{\text{rank,in}}+H_{\text{rank}}
$$

$M_{\text{rank}}$ 和 $H_{\text{rank}}$ 只由这个 rank 本地 token 决定。

真实的 $S_{\text{rank,in}}$ 来自前面 rank。

## 1. 单个 chunk 的公式

一个 chunk 内，先有：

$$
V_{\text{new}}=U-WS
$$

令：
$$
\bar K^\top=(\Gamma_C/\Gamma)\odot K\in\mathbb{R}^{C\times d_k}
$$
再更新 state：

$$
S_{\text{next}}=DS+\bar K^\top V_{\text{new}}
$$

代入：

$$
S_{\text{next}}
=
DS+\bar K^\top(U-WS)
$$

展开：

$$
S_{\text{next}}
=
DS+\bar K^\top U-\bar K^\top WS
$$

整理成关于 $S$ 的仿射形式：

$$
S_{\text{next}}
=
(D-\bar K^\top W)S+\bar K^\top U
$$

所以单个 chunk 的：

$$
M_{\text{chunk}}=D-\bar K^\top W
$$

$$
H_{\text{chunk}}=\bar K^\top U
$$

也就是：

$$
S_{\text{next}}=M_{\text{chunk}}S+H_{\text{chunk}}
$$

其中 $D$ 是 gate 对旧 state 的衰减。KDA 的 gk 作用在 key 维度上：

$$
D=\operatorname{Diag}(\exp2(gk_{\text{last}}))
$$

所以旧 state 进入当前 chunk 后，确实要先经过：

$$
DS
$$

## 2. 多个 chunk 合成一个 rank

假设一个 rank 内有 3 个 chunk。

第 0 个 chunk：

$$
S_1=M_0S_0+H_0
$$

第 1 个 chunk：

$$
S_2=M_1S_1+H_1
$$

第 2 个 chunk：

$$
S_3=M_2S_2+H_2
$$

展开：

$$
S_2=M_1(M_0S_0+H_0)+H_1
$$

$$
S_2=M_1M_0S_0+M_1H_0+H_1
$$

继续：

$$
S_3=M_2(M_1M_0S_0+M_1H_0+H_1)+H_2
$$

$$
S_3=M_2M_1M_0S_0+M_2M_1H_0+M_2H_1+H_2
$$

所以整个 rank 可以写成：

$$
S_3=M_{\text{rank}}S_0+H_{\text{rank}}
$$

其中：

$$
M_{\text{rank}}=M_2M_1M_0
$$

$$
H_{\text{rank}}=M_2M_1H_0+M_2H_1+H_2
$$

因此：

$$
H_{\text{rank}}\neq H_0+H_1+H_2
$$

前面 chunk 生成的 $H$ 会继续被后面 chunk 的 $M$ 作用。

## 3. 为什么令 $S_{\text{rank,in}}=0$ 可以求 $H_{\text{rank}}$

rank 级公式是：

$$
S_{\text{rank,out}}
=
M_{\text{rank}}S_{\text{rank,in}}+H_{\text{rank}}
$$

令：

$$
S_{\text{rank,in}}=0
$$

得到：

$$
S_{\text{rank,out}}=H_{\text{rank}}
$$

所以当把本 rank 的入口 state 临时设为 0，并跑完整个 rank，本质上是在求：

$$
H_{\text{rank}}
$$

这只是为了求仿射变换的常数项。

真实 forward 里，只有全局第一个 rank 的入口 state 是 0。

非第一个 rank 的真实入口 state 不是 0。

## 4. 为什么求 $H$ 时仍然有 $WH$

设一个 rank 有两个 chunk。

求 $H_{\text{rank}}$ 时，先令：

$$
S_0=0
$$

第 0 个 chunk：

$$
S_1=D_0S_0+\bar K_0^\top(U_0-W_0S_0)
$$

代入 $S_0=0$：

$$
S_1=\bar K_0^\top U_0
$$

所以：

$$
S_1=H_0
$$

第 1 个 chunk 的输入不是 0，而是上一 chunk 的输出：

$$
S_1=H_0
$$

因此：

$$
S_2=D_1H_0+\bar K_1^\top(U_1-W_1H_0)
$$

这就是代码实际采用的计算顺序。

先用已有 state 修正 value：

$$
U_1-W_1H_0
$$

再乘当前 chunk 的 key：

$$
\bar K_1^\top(U_1-W_1H_0)
$$

同时旧 state 经过 gate 衰减：

$$
D_1H_0
$$

最后相加：

$$
S_2=D_1H_0+\bar K_1^\top(U_1-W_1H_0)
$$

展开：

$$
S_2=D_1H_0+\bar K_1^\top U_1-\bar K_1^\top W_1H_0
$$

整理：

$$
S_2=(D_1-\bar K_1^\top W_1)H_0+\bar K_1^\top U_1
$$

也就是：

$$
S_2=M_1H_0+H_1
$$

其中：

$$
M_1=D_1-\bar K_1^\top W_1
$$

$$
H_1=\bar K_1^\top U_1
$$

所以求 $H_{\text{rank}}$ 时，虽然 rank 的初始输入设成了 0，但从第二个 chunk 开始，输入 state 已经不是 0。

它是前面 chunk 生成的 $H$。

因此后续 chunk 必须包含：

$$
W_1H_0
$$

也必须包含：

$$
D_1H_0
$$

它们合起来就是：

$$
M_1H_0
$$

这就是 $WH$ 出现在求 $H_{\text{rank}}$ 过程里的原因。

## 5. rank 之间如何合成真实入口 state

每个 rank 先独立得到自己的：

$$
(M_r,H_r)
$$

rank0 的真实入口 state：

$$
S_{0,\text{in}}=0
$$

rank1 的真实入口 state：

$$
S_{1,\text{in}}=M_0S_{0,\text{in}}+H_0
$$

所以：

$$
S_{1,\text{in}}=H_0
$$

rank2 的真实入口 state：

$$
S_{2,\text{in}}=M_1S_{1,\text{in}}+H_1
$$

代入 $S_{1,\text{in}}=H_0$：

$$
S_{2,\text{in}}=M_1H_0+H_1
$$

rank3 的真实入口 state：

$$
S_{3,\text{in}}=M_2S_{2,\text{in}}+H_2
$$

代入：

$$
S_{3,\text{in}}=M_2(M_1H_0+H_1)+H_2
$$

也就是：

$$
S_{3,\text{in}}=M_2M_1H_0+M_2H_1+H_2
$$

一般地：

$$
S_{r,\text{in}}
=
M_{r-1}S_{r-1,\text{in}}+H_{r-1}
$$

其中：

$$
S_{0,\text{in}}=0
$$

这一步就是 rank 间的 scan。

## 6. CP forward 的两阶段

CP forward 可以分成两阶段。

第一阶段：每个 rank 总结自己的本地序列段：

$$
\text{local summary}_r=(M_r,H_r)
$$

然后把所有 rank 的 summary 聚合起来。

第二阶段：根据前面 rank 的 summary，算当前 rank 的真实入口 state：

$$
S_{r,\text{in}}
=
F_{r-1}\circ F_{r-2}\circ\cdots\circ F_0(0)
$$

其中：

$$
F_i(S)=M_iS+H_i
$$

得到真实 $S_{r,\text{in}}$ 后，当前 rank 再用自己的 token 正常计算本地输出。

所以：

$$
H_r=F_r(0)
$$

只是 summary 的常数项。

真实 forward 使用的是：

$$
F_r(S_{r,\text{in}})
$$

也就是：

$$
S_{r,\text{out}}=M_rS_{r,\text{in}}+H_r
$$

## 7. $hm$ 的数学含义

CP 中聚合的对象可以理解成：

$$
hm_r=(H_r,M_r)
$$

其中 $H_r$ 是本 rank 从 0 state 开始跑完整段后的输出：

$$
H_r=F_r(0)
$$

$M_r$ 是本 rank 对输入 state 的线性作用：

$$
F_r(S)-F_r(0)=M_rS
$$

所以只要拿到所有 rank 的 $hm_r$，就能合成任意 rank 的真实入口 state。

## 8. `cu_seqlens[-2:]` 的意义

变长序列下，本地 rank 可能包含多个 sequence 片段。

对 CP 跨 rank 传播来说，真正会影响下一个 rank 的，是当前 rank 最后一个仍会向后延续的片段。

因此 pre-process 总结的是本 rank 末尾片段对应的：

$$
(M_{\text{tail}},H_{\text{tail}})
$$

前面的本地片段仍然会在本 rank 内正常计算输出，但它们不需要作为跨 rank summary 传给后面的 rank。

如果某个 sequence 在本 rank 内已经结束，它不会成为下一个 rank 的入口 state 来源。

## 9. state、prefill、training、decode

training 中，中间 state 需要服务 backward。

prefill 中，如果后面要接 decode，真正需要保留的是 prompt 结束处的最终 state：

$$
S_{\text{prompt,end}}
$$

不是每个 rank 的所有中间 state。

decode 每步通常只有一个 token：

$$
T=1
$$

此时 sequence 维 CP 基本没有并行度。

decode 更自然的并行方式通常是 TP、PP 或 batch parallel。

所以 KDA 这类 LA 结构更适合：

$$
\text{prefill 用 CP}
$$

再切到：

$$
\text{decode 用 TP/PP/batch parallel}
$$

## 10. 最容易混淆的一句话

设 $S_{\text{rank,in}}=0$，是在求：

$$
H_{\text{rank}}
$$

不是说真实 forward 中所有 rank 的入口 state 都是 0。

真实 forward 中：

$$
S_{\text{rank,out}}
=
M_{\text{rank}}S_{\text{rank,in}}+H_{\text{rank}}
$$

其中只有 rank0 的：

$$
S_{0,\text{in}}=0
$$

其他 rank 的：

$$
S_{r,\text{in}}\neq0
$$
