---
name: read-cuda-triton-kernel
description: Read and explain existing CUDA, Triton, CUDA C++, or Python-wrapped GPU kernels by reconstructing the execution model, tile ownership, memory movement, loop dependencies, resource limits, and likely bottlenecks. Use when asked to understand, review, debug, or explain a kernel before optimizing or rewriting it, especially for FLA ops, attention kernels, GEMM-like kernels, reductions, scans, online softmax, causal masking, or block-sparse/irregular tiling.
---

# Read and Explain CUDA/Triton Kernel

## Purpose

把已有 CUDA/Triton kernel 解读成执行模型，而不是先优化或重写。目标是回答“这个 kernel 如何把 program/block/warp/thread 映射到数据，如何搬运和复用数据，哪里最可能慢或错”。

## First Principles

- 不平均解释所有代码，只抓主数据流、tile ownership、关键循环、边界语义和性能约束。
- 先解释 tile ownership，再解释指令细节。
- 不默认提出改写方案；只有用户要求优化或发现明确 bug 时，才给出修改建议。
- 看到 `num_stages`、pipeline stage、`cp.async`、Triton `num_stages` 时，必须说明它是不是 pipeline depth，而不是算法维度。
- 看到 `BK`、`BLOCK_K`、`BT`、`BK/V` 这类 K 方向块大小时，必须说明它影响 K-loop 粒度、smem/寄存器占用、reuse、mma dispatch 或向量化粒度。
- 看到 `mask`、`predicate`、`where`、`boundary_check`、`tl.load(..., mask=...)` 时，必须区分它服务于边界保护，还是稀疏/causal/local attention 等算法语义。

## Reading Workflow

1. Locate the kernel entry point and launcher.
   - For Triton, read the `@triton.jit` function, call wrapper, `grid`, `num_warps`, `num_stages`, and `constexpr` parameters.
   - For CUDA C++, read `<<<grid, block, smem, stream>>>`, kernel signature, template parameters, launch bounds, and any shared-memory declarations.
   - In FLA ops, also check the Python wrapper, shape conventions, chunking names, and nearby naive/reference implementation when available.

2. Recover tensor shapes and strides before reading arithmetic.
   - Write down logical dimensions such as batch `B`, heads `H`, sequence/chunk `T/C`, key/value/head dim `K/V`, and output dimensions.
   - Decode layout from pointer arithmetic, not variable names alone.
   - Identify whether strides are element strides or byte strides.

3. Decode grid ownership.
   - Map each `program_id(axis)` or `blockIdx.{x,y,z}` to one output tile.
   - State which output elements a program/block owns and whether ownership is unique, split-K, atomic, or reduction-based.
   - If the kernel has persistent scheduling, grouped ordering, swizzling, or remapped program IDs, explain both the physical launch ID and logical tile ID.

4. Decode block/warp/thread ownership.
   - For CUDA, state block size, number of warps, warp roles, lane mapping, and per-thread register fragments.
   - For Triton, translate `tl.arange`, block tensors, and masks into conceptual lanes/elements; do not overclaim exact hardware lane mapping unless the code uses warp-level primitives or inline asm.
   - For MMA kernels, identify CTA tile, warp tile, MMA fragment shape, and accumulator fragment layout.

5. Decode the algorithm.
   - Write the mathematical recurrence or reduction in compact form.
   - Mark loop-carried state such as running max/sum, recurrent hidden state, prefix scan, accumulated dot product, or carry between chunks.
   - Distinguish online/streaming logic from ordinary tiled reduction.

6. Decode memory movement and resources.
   - Explain global loads/stores, coalescing assumptions, smem layout, register fragments, vectorization, `ldmatrix`, `mma`, `cp.async`, barriers, and cache modifiers when present.
   - Estimate resource pressure qualitatively or quantitatively: smem per block, registers per thread, occupancy, memory traffic, arithmetic intensity.

7. Judge likely bottleneck and suspicious points.
   - Use the algorithm and memory traffic to classify compute-bound, memory-bound, or latency/synchronization-bound.
   - Call out concrete risks: wrong bounds, bad coalescing, bank conflicts, register spill, over-deep pipeline, too-small/too-large `BK`, atomics, divergent causal masks, or excessive recomputation.

## Required Output Order

When answering a kernel-reading request, use this order unless the user asks for a shorter answer:

1. 结论：这个 kernel 在算什么，主瓶颈大概率是什么
2. Grid 解读：`program_id` / `blockIdx` 如何映射到输出 tile
3. Block/Warp 解读：一个 block 有多少 warp，每个 warp 负责哪块数据
4. 超参数解读：`BM`/`BN`/`BK`、`num_warps`、`num_stages`、`BLOCK_SIZE` 分别约束什么
5. 算法解读：数学公式、循环依赖、online/streaming/reduction 逻辑
6. 实现解读：global load、smem layout、寄存器 fragment、mma/ldmatrix/cp.async
7. 资源约束：smem、register、occupancy、memory traffic、AI
8. 最小伪代码：只保留主循环和关键数据流
9. 性能判断：compute-bound / memory-bound / latency-bound
10. 可疑点：越界、bank conflict、coalescing、spill、stage 过深、`BK` 不合适

For small kernels, combine adjacent sections, but keep the same logical order.

## Interpretation Rules

### Grid and Tile Ownership

- Start from the output tensor. Ask: “Which program/block writes which output tile?”
- Derive tile coordinates from launch axes:
  - Triton: `pid_m = tl.program_id(0)`, `pid_n = tl.program_id(1)`, grouped or swizzled IDs.
  - CUDA: `blockIdx.x/y/z`, `threadIdx.x/y/z`.
- Identify whether tiles cover output directly, cover an intermediate buffer, or participate in a multi-kernel pipeline.
- If multiple programs contribute to the same output, identify the reduction mechanism: atomics, separate reduction kernel, split-K, prefix scan, or write to scratch.

### Block, Warp, and Lane Roles

- CUDA:
  - Compute `warps_per_block = blockDim.x * blockDim.y * blockDim.z / 32`.
  - Map warp ID and lane ID if the code uses `threadIdx`, `warp_id`, `lane_id`, cooperative groups, `__shfl_*`, or warp reductions.
  - Explain which data each warp loads, computes, reduces, and stores.
- Triton:
  - Treat block tensors as vectorized program-local work.
  - Explain `tl.arange` index grids as element ownership inside a program.
  - Use `num_warps` as a scheduling/resource hint unless code structure makes warp partitioning explicit.

### Hyperparameters

- `BM`, `BN`: output tile shape along M/N-like dimensions; control reuse, accumulator size, store granularity, and edge-mask cost.
- `BK`: reduction or streaming step size; controls K-loop granularity, smem footprint, register fragments, data reuse, vectorization/MMA dispatch count, and tail handling.
- `BLOCK_SIZE`: decode from usage, not name. It may mean tile length, reduction width, vector length, chunk size, or number of elements per program.
- `num_warps`: Triton scheduling/parallelism knob; too low can underutilize compute, too high can increase register pressure or reduce occupancy.
- `num_stages`: pipeline depth for software pipelining/prefetching in Triton or async-copy buffering in CUDA-style kernels. Do not describe it as an algorithmic loop count unless the code explicitly uses it that way.

### Masks and Predicates

- Boundary masks protect out-of-range tensor access at ragged tile edges, e.g. `offs_m < M`, `offs_n < N`, `offs_k < K`.
- Semantic masks implement algorithm behavior, e.g. causal attention, sliding windows, block sparsity, padding tokens, or segment boundaries.
- A single mask can combine both; separate the terms and explain each one.
- For masked loads, state the fill value and why it is correct: zero for additive reductions, `-inf` for max/softmax logits, identity for recurrence when applicable.

### Algorithm and Math

- Express the main computation with the smallest useful formula. For example:

  $$C_{m,n} = \sum_k A_{m,k} B_{k,n}$$

- For online softmax, identify running max and denominator updates:

  $$m_i = \max(m_{i-1}, x_i), \quad l_i = l_{i-1} e^{m_{i-1}-m_i} + e^{x_i-m_i}$$

- For recurrent or scan kernels, name the state and specify whether the loop is sequential across time/chunks or parallelized by associative scan.
- For backward kernels, state which gradients are produced and where reductions over batch/head/time/channel occur.

### Implementation Details

- Global memory:
  - State whether loads/stores are contiguous, strided, broadcasted, gathered, or repeated.
  - Check vectorized loads/stores for alignment assumptions.
- Shared memory:
  - State logical layout, padding, double buffering, and whether accesses risk bank conflicts.
  - For CUDA, pair every producer/consumer phase with the required barrier or async wait.
- Registers:
  - Identify accumulator fragments, temporary vectors, per-thread arrays, and loop-carried state.
  - Watch for large block tensors in Triton or large per-thread arrays in CUDA that may spill.
- Tensor core path:
  - Identify `mma.sync`, `wmma`, `tl.dot`, or `tl.dot_scaled`.
  - State input fragment shape, accumulator dtype, and whether `BK` aligns with MMA K.
- Pipeline:
  - Explain what data is prefetched, how many stages/buffers exist, and where waits/barriers happen.

## Minimal Pseudocode

Keep pseudocode shorter than the real kernel. Preserve only ownership, the main loop, masks, state updates, and stores.

Example shape:

```text
pid -> output tile coordinates
initialize accum/state
for k_block or time_block:
    load needed input tile(s) with boundary/semantic masks
    update accum/state
store owned output tile with output boundary mask
```

## Performance Judgment

- Compute-bound: high arithmetic intensity, substantial MMA/FMA work per byte, good tile reuse, few synchronizations.
- Memory-bound: low reuse, large streaming reads/writes, small reduction dimension, gather/scatter, or output-dominated traffic.
- Latency-bound: many tiny tiles, serial loop-carried dependencies, atomics, synchronization-heavy reductions, or insufficient active warps/programs.
- When possible, estimate arithmetic intensity as useful FLOPs divided by bytes moved from global memory. If exact counting is too costly, provide a directional estimate and state the assumption.

## Suspicious Point Checklist

- Bounds: every load/store at tile edges is masked or otherwise proven in range.
- Mask semantics: boundary and causal/sparse/padding masks are not accidentally conflated.
- Coalescing: contiguous lanes/program elements access contiguous memory where expected.
- Bank conflicts: smem strides do not create severe 32-bank conflicts in hot loops.
- Spill risk: accumulator/block tensor size does not imply excessive registers.
- Occupancy: smem, registers, and `num_warps` allow enough active CTAs/programs.
- Pipeline depth: `num_stages`/double buffering hides latency without consuming too much smem/registers.
- `BK`: not so small that MMA/loop overhead dominates, and not so large that smem/register pressure or tail waste dominates.
- Reductions: atomics or split reductions are deterministic enough for the expected numeric tolerance.
- Dtypes: accumulation dtype and conversions match numerical requirements.
