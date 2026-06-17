# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import argparse
import os
import sys
from pathlib import Path

# Set these before importing Triton so the knobs are picked up in this process.
os.environ.setdefault("TRITON_ALWAYS_COMPILE", "1")
os.environ.setdefault("TRITON_KERNEL_DUMP", "1")
os.environ.setdefault("TRITON_DUMP_PTXAS_LOG", "1")
os.environ.setdefault("TRITON_DUMP_DIR", "/tmp/kda_inter_solve_dump")
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/kda_inter_solve_cache")
os.environ.setdefault("TRITON_PRINT_AUTOTUNING", "1")

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect register usage for chunk_kda_fwd_kernel_inter_solve_fused."
    )
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--T", type=int, default=64)
    parser.add_argument("--H", type=int, default=4)
    parser.add_argument("--HV", type=int, default=4)
    parser.add_argument("--K", type=int, default=128)
    parser.add_argument("--BT", type=int, default=64, choices=(32, 64))
    parser.add_argument("--BC", type=int, default=16)
    parser.add_argument("--safe-gate", action="store_true")
    parser.add_argument("--bk", type=int, choices=(32, 64), default=None)
    parser.add_argument("--num-warps", type=int, choices=(1, 2, 4), default=None)
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Force and report every BK/num_warps candidate instead of the autotuned choice.",
    )
    return parser.parse_args()


def make_inputs(args):
    device = "cuda"
    dtype = torch.bfloat16

    q = torch.randn(args.B, args.T, args.H, args.K, device=device, dtype=dtype)
    k = torch.randn(args.B, args.T, args.H, args.K, device=device, dtype=dtype)
    g = torch.randn(args.B, args.T, args.HV, args.K, device=device, dtype=torch.float32)
    beta = torch.rand(args.B, args.T, args.HV, device=device, dtype=torch.float32)

    Aqk = torch.empty(args.B, args.T, args.HV, args.BT, device=device, dtype=dtype)
    Akk = torch.zeros(args.B, args.T, args.HV, args.BT, device=device, dtype=dtype)
    Akkd = torch.zeros(args.B, args.T, args.HV, args.BC, device=device, dtype=torch.float32)
    return q, k, g, beta, Aqk, Akkd, Akk


def get_autotuner(kernel):
    current = kernel
    while not hasattr(current, "configs") or not hasattr(current, "cache"):
        if not hasattr(current, "fn"):
            raise TypeError(f"Could not find Triton autotuner inside {type(kernel)!r}.")
        current = current.fn
    return current


def set_single_config(kernel, triton, bk, num_warps):
    autotuner = get_autotuner(kernel)
    autotuner.configs = [triton.Config({"BK": bk}, num_warps=num_warps)]
    autotuner.cache.clear()


def run_kernel(args, kernel, triton, bk=None, num_warps=None):
    if bk is not None or num_warps is not None:
        if bk is None or num_warps is None:
            raise ValueError("--bk and --num-warps must be passed together.")
        set_single_config(kernel, triton, bk, num_warps)

    q, k, g, beta, Aqk, Akkd, Akk = make_inputs(args)
    NT = triton.cdiv(args.T, args.BT)
    NC = triton.cdiv(args.BT, args.BC)
    grid = (NT, args.B * args.HV)

    compiled = kernel[grid](
        q=q,
        k=k,
        g=g,
        beta=beta,
        Aqk=Aqk,
        Akkd=Akkd,
        Akk=Akk,
        scale=1.0,
        cu_seqlens=None,
        chunk_indices=None,
        T=args.T,
        H=args.H,
        HV=args.HV,
        K=args.K,
        BT=args.BT,
        BC=args.BC,
        NC=NC,
        USE_SAFE_GATE=args.safe_gate,
    )
    torch.cuda.synchronize()
    return compiled


def print_kernel_info(compiled, label):
    print(label)
    print(f"  name: {compiled.name}")
    print(f"  n_regs: {compiled.n_regs}")
    print(f"  n_spills: {compiled.n_spills}")
    print(f"  shared: {compiled.metadata.shared}")
    print(f"  num_warps: {compiled.metadata.num_warps}")
    print(f"  num_stages: {compiled.metadata.num_stages}")


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires CUDA.")
    if args.HV % args.H != 0:
        raise ValueError(f"HV must be divisible by H, got H={args.H}, HV={args.HV}.")
    if (args.bk is None) != (args.num_warps is None):
        raise ValueError("--bk and --num-warps must be passed together.")

    torch.manual_seed(0)

    import triton

    from fla.ops.kda.chunk_intra import chunk_kda_fwd_kernel_inter_solve_fused

    print(f"TRITON_DUMP_DIR: {os.environ['TRITON_DUMP_DIR']}")
    print(f"TRITON_CACHE_DIR: {os.environ['TRITON_CACHE_DIR']}")
    print(
        "shape: "
        f"B={args.B}, T={args.T}, H={args.H}, HV={args.HV}, K={args.K}, "
        f"BT={args.BT}, BC={args.BC}, safe_gate={args.safe_gate}"
    )

    if args.sweep:
        for bk in (32, 64):
            for num_warps in (1, 2, 4):
                compiled = run_kernel(
                    args,
                    chunk_kda_fwd_kernel_inter_solve_fused,
                    triton,
                    bk=bk,
                    num_warps=num_warps,
                )
                print_kernel_info(compiled, f"BK={bk}, num_warps={num_warps}")
        return

    compiled = run_kernel(
        args,
        chunk_kda_fwd_kernel_inter_solve_fused,
        triton,
        bk=args.bk,
        num_warps=args.num_warps,
    )
    label = "autotuned config" if args.bk is None else f"BK={args.bk}, num_warps={args.num_warps}"
    print_kernel_info(compiled, label)
    if args.bk is None:
        autotuner = get_autotuner(chunk_kda_fwd_kernel_inter_solve_fused)
        print(f"  best_config: {autotuner.best_config}")


if __name__ == "__main__":
    main()
