# KDA WY Beta Scaling 精度分析

这份笔记记录 `recompute_w_u_fwd_kda_kernel` 里一个看起来很自然的代数变换：把 `beta` 从右操作数 `v/k` 上移到矩阵 `A` 的列上。数学上它们等价，但 bf16/Tensor Core 路径下并不会精确对齐。这是一个很好的例子：优化 GPU kernel 时，不能只看代数等式，还要看低精度输入的舍入位置。

最重要的结论先写在前面：

```text
两种写法的误差来自不同的 bf16 舍入点。

旧写法先量化 beta * V / beta * K。
新写法先量化 A * beta。

进入 tl.dot 的低精度 operand 不同，所以结果不保证一致。
```

## 1. 背景

`chunk_kda_fwd_intra` 前面会先构造 chunk 内的 lower-triangular inverse。代码里传给 `recompute_w_u_fwd_kda_kernel` 的 `A` 实际是：

$$
L^{-1}
$$

其中：

$$
L=I+\operatorname{StrictTril}(\operatorname{Diag}(\beta)B)
$$

而一些推导文档里常把完整矩阵写成：

$$
A_{\text{math}}
=
L^{-1}\operatorname{Diag}(\beta)
$$

所以要注意：代码里的 `A` 不是 $A_{\text{math}}$，而是 $L^{-1}$。`beta` 没丢，它在 `recompute_w_u_fwd_kda_kernel` 里乘到右操作数上。

当前 kernel 的 forward 写法是：

```python
b_vb = b_v * b_b[:, None]
b_u = tl.dot(b_A, b_vb)

b_kb = b_k * b_b[:, None]
b_kb *= exp2(b_gk)
b_w = tl.dot(b_A, b_kb)
```

数学上对应：

$$
u=L^{-1}(\operatorname{Diag}(\beta)V)
$$

$$
w=L^{-1}(\operatorname{Diag}(\beta)(\exp(g)\odot K))
$$

## 2. 想研究的变换

因为：

$$
L^{-1}\operatorname{Diag}(\beta)V
=
(L^{-1}\operatorname{Diag}(\beta))V
$$

所以可以把 `beta` 临时乘到 `A` 的列上：

```python
b_A = tl.load(p_A, boundary_check=(0, 1))
b_A = (b_A * b_b[None, :]).to(b_A.dtype)

b_u = tl.dot(b_A, b_v)
b_w = tl.dot(b_A, b_k * exp2(b_gk))
```

这里的 `b_b[None, :]` 是 column scaling。注意不能写成 `b_b[:, None]`，那会 scale `A` 的行，语义是错的。

这个变换不需要写回 `A`，所以不会改变前一个 kernel 产生的 `Akk`，也不会直接改变 backward 看到的存储表示。它只是 `recompute_w_u_fwd_kda_kernel` 内部的局部等价变换。

## 3. 为什么它可能更快

以常见形状 `BT=64, K=128, V=128` 为例。

当前写法需要做：

```text
beta * v: 64 * 128
beta * k: 64 * 128
```

合计大约：

```text
64 * (128 + 128) = 16384
```

如果改成 column-scaled `A`，只需要：

```text
A * beta[None, :]: 64 * 64 = 4096
```

所以从标量乘法数量看，新写法少一些。主计算仍然是两个 `tl.dot`，这个优化不改变主要 GEMM 规模，但可能减少一些右操作数准备开销。

## 4. 为什么数学等价但数值不完全等价

核心原因不是 KDA 公式本身，而是浮点计算的基本事实：

```text
浮点乘法不是实数乘法。每一步会舍入。
```

用 `fl(...)` 表示一次浮点舍入。实数里有：

$$
A(\beta V)=(A\operatorname{Diag}(\beta))V
$$

但 kernel 里跑的不是这个式子，而更接近下面两个式子。

旧写法：

$$
u_{\text{old}}
=
\operatorname{dot}\left(
A,\,
\operatorname{fl}_{bf16}(\beta V)
\right)
$$

新写法：

$$
u_{\text{new}}
=
\operatorname{dot}\left(
\operatorname{fl}_{bf16}(A\operatorname{Diag}(\beta)),\,
V
\right)
$$

这两个式子已经不一样了。差别在于：旧写法把 `beta` 乘到右操作数 `V/K` 上，然后这个右操作数被量化；新写法把 `beta` 乘到左操作数 `A` 上，然后左操作数被量化。

也就是说，二者进入 `tl.dot` 的低精度输入不同：

```text
old dot inputs:
  left  = A
  right = round_bf16(beta * V)

new dot inputs:
  left  = round_bf16(A * beta[None, :])
  right = V
```

只要 `round_bf16(beta * V)` 和 `round_bf16(A * beta)` 的舍入误差不一样，dot 的结果就会不一样。

### 4.1 一个最小标量例子

实数里：

$$
a(\beta v)=(a\beta)v
$$

浮点里实际是：

$$
y_1=\operatorname{fl}(a \cdot \operatorname{fl}(\beta v))
$$

$$
y_2=\operatorname{fl}(\operatorname{fl}(a\beta)\cdot v)
$$

如果 `fl(beta * v)` 的舍入方向和 `fl(a * beta)` 的舍入方向不同，`y_1` 和 `y_2` 就会不同。这和结合律、分配律在浮点里不严格成立是同一个问题。

### 4.2 为什么 bf16 下更明显

bf16 的有效尾数只有 7 bits。它的相对精度大约是：

$$
2^{-7}\approx 0.0078125
$$

所以量级在 1 附近的数，bf16 相邻可表示值之间的间隔大约就是 `0.0078125` 或它的倍数。我们看到的：

```text
max_abs = 0.015625
```

正好是 `2 * 0.0078125`。这说明误差很可能来自 bf16 operand 量化，而不是公式写错。

### 4.3 为什么 dot 会放大这个差异

对一个输出元素，dot 做的是：

$$
y_i=\sum_j A_{i,j}X_j
$$

旧写法和新写法的输入差异会进入这个求和：

$$
y_{\text{old}}-y_{\text{new}}
=
\sum_j
\left(
A_{i,j}\operatorname{fl}(\beta_jV_j)
-
\operatorname{fl}(A_{i,j}\beta_j)V_j
\right)
$$

每个 `j` 上的差异可能有正有负，平均误差可能不大；但某些行、某些 channel 上如果误差同向累积，就会出现较大的 `max_abs`。这就是为什么要同时看 `max_abs`、`mean_abs` 和分位数，而不能只看一个指标。

### 4.4 Tensor Core 路径要特别小心

`tl.dot` 通常会走 Tensor Core。Tensor Core 常见模式是：

```text
低精度输入 * 低精度输入 -> 较高精度累加
```

累加可能是 fp32，但输入 operand 已经是 bf16/fp16 了。因此“进入 dot 之前先把谁乘上 beta”会决定哪个 operand 先被 bf16 量化。只要量化位置改变，结果就不保证 bitwise 相同。

所以这个问题的本质是：

```text
实数代数重排改变了低精度 kernel 的舍入边界。
```

这也是精度对齐里最常见的坑之一。

## 5. 我们怎么分析

这次不是直接改正式 kernel，而是新增了一个独立 example：

```text
examples/kda_wy_beta_scale_equiv.py
```

这个 example 做了四件事：

1. 构造随机 `q/k/v/beta/gk` 和一个模拟的 unit-lower `A`。
2. 调用正式 `recompute_w_u_fwd`，得到旧 Triton kernel 输出。
3. 在 example 内写一个 column-scaled-A Triton kernel，得到新 Triton 写法输出。
4. 用 PyTorch reference 分别复现旧公式和新公式，确认差异来自舍入位置，而不是 kernel 写错。

关键对照是：

```text
original_kernel_vs_column_kernel_w
original_kernel_vs_column_kernel_u
```

它们直接比较两种 Triton 写法，不是只比较 PyTorch 公式。

## 6. 当前实验结果

命令：

```bash
python examples/kda_wy_beta_scale_equiv.py
```

形状：

```text
B=1, T=128, H=4, HV=4, K=128, V=128, BT=64
```

一次 RTX 5060 Ti / sm_120 上的输出：

```text
original_kernel_vs_column_kernel_w:
  max_abs=0.015625, mean_abs=0.000803, max_rel=1.858300, mean_rel=0.006914

original_kernel_vs_column_kernel_u:
  max_abs=0.015625, mean_abs=0.000799, max_rel=1.887500, mean_rel=0.005614

original_kernel_vs_column_kernel_qg:
  max_abs=0.000000, mean_abs=0.000000

original_kernel_vs_column_kernel_kg:
  max_abs=0.000000, mean_abs=0.000000
```

`qg` 和 `kg` 不依赖这个 beta placement，所以它们完全一致。这是一个 negative control，说明 example 里两个 kernel 的其它路径没有被无意改坏。

`w/u` 的 `max_abs=0.015625` 不算小。它很像 bf16 在当前数值量级上的一个量化台阶。`mean_abs` 大约是 `8e-4`，说明大多数元素差异没到 max 那么大。

`max_rel` 很大，主要是因为有些元素本身接近 0。相对误差的分母很小时，一个很小的绝对误差也会产生很大的 relative error。因此不能只看 `max_rel`，还要看 `mean_abs`、`mean_rel`、分位数，以及端到端输出/梯度误差。

## 7. 这个误差说明什么

当前结果说明：

```text
这个变换数学上正确，但不是精度无感的替换。
```

它不适合被当成“完全等价重排”直接合入。要继续研究，需要回答两个问题：

1. 性能收益是否真实存在？
2. 端到端数值误差是否在 KDA 测试容忍范围内？

只有局部 `w/u` 差异还不够。`w/u` 后面会进入 recurrent state update 和 output kernel，误差可能被放大，也可能被后续计算稀释。

## 8. 推荐的后续分析流程

### 8.1 局部 kernel 误差

先用 example 扫形状：

```bash
python examples/kda_wy_beta_scale_equiv.py --T 64 --K 64 --V 64
python examples/kda_wy_beta_scale_equiv.py --T 128 --K 128 --V 128
python examples/kda_wy_beta_scale_equiv.py --BT 32
```

需要观察：

```text
max_abs
mean_abs
max_rel
mean_rel
```

最好再加分位数：

```text
p50 / p90 / p99 / p99.9 absolute error
```

因为 max 往往来自极少数元素。

### 8.2 使用真实 KDA 中间量

当前 example 的 `A` 是随机模拟的 unit-lower inverse。下一步应该从真实 KDA forward 中拿到：

```text
Akk
beta
gk
k
v
```

然后只替换 `recompute_w_u_fwd_kda_kernel`，比较真实分布下的 `w/u` 差异。真实 `Akk` 的数值分布可能比随机矩阵更温和，也可能更尖锐。

### 8.3 端到端 forward

对比完整 `chunk_kda` 输出：

```text
old kernel output o
column-scaled-A kernel output o
```

至少要覆盖：

```text
safe_gate=False
safe_gate=True
chunk_size=32/64
dtype=bf16/fp16
cu_seqlens fixed/varlen
```

### 8.4 backward 和训练敏感性

虽然这个局部变换不改变存储的 `Akk`，但 forward 的 `w/u` 变了，后续 loss 和 backward 输入都会变。因此还要比较：

```text
dq, dk, dv, dbeta, dg
```

如果梯度误差明显大于 forward，说明这个优化不适合默认路径。

### 8.5 性能基准

如果数值可接受，再 benchmark。不要只看局部 kernel 时间，也要看完整 KDA op 时间：

```text
recompute_w_u_fwd_kda_kernel only
chunk_kda forward
chunk_kda forward+backward
```

因为这个改动省的是 RHS 准备乘法，不一定能改变主 dot 的瓶颈。

## 9. 判断标准

这个优化值得继续研究，但要满足：

1. 局部 `w/u` 差异在已有测试容忍度内。
2. 完整 `chunk_kda` forward 和 backward 都通过 reference。
3. 性能有稳定收益，不能只是单个 shape 有小幅波动。
4. 不让寄存器压力明显上升。`b_A` 已经是 `[BT, BT]`，虽然覆盖写 `b_A` 理论上不增加 fragment 数量，但实际编译器可能因为 live range 改变产生不同 register allocation。

## 10. 当前结论

这个点有研究价值，因为它揭示了一个常见 kernel 优化陷阱：

```text
代数等价 != 低精度 kernel 输出完全等价
```

在 fp32 里，这个变换几乎可以认为是安全重排；在 bf16/Tensor Core 路径里，乘法放在哪一侧会改变输入 operand 的量化位置，最终 `w/u` 出现 `0.015625` 级别的 max abs 差异。

所以当前建议是：

```text
先保留正式 kernel 不改；
把 column-scaled-A 作为候选实现继续做局部、真实中间量、端到端和性能实验。
```
