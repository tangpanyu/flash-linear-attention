# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import argparse
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@triton.jit(do_not_specialize=["T"])
def recompute_w_u_fwd_kda_kernel_column_scaled_A(
    q,
    k,
    qg,
    kg,
    v,
    beta,
    w,
    u,
    A,
    gk,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)
    bos = i_b * T

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    qg += (bos * HV + i_hv) * K
    kg += (bos * HV + i_hv) * K
    v += (bos * HV + i_hv) * V
    u += (bos * HV + i_hv) * V
    w += (bos * HV + i_hv) * K
    gk += (bos * HV + i_hv) * K
    beta += bos * HV + i_hv
    A += (bos * HV + i_hv) * BT

    p_b = tl.make_block_ptr(beta, (T,), (HV,), (i_t * BT,), (BT,), (0,))
    b_b = tl.load(p_b, boundary_check=(0,))

    p_A = tl.make_block_ptr(A, (T, BT), (HV*BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_A = (b_A * b_b[None, :]).to(b_A.dtype)

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v, (T, V), (HV*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_u = tl.make_block_ptr(u, (T, V), (HV*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_u = tl.dot(b_A, b_v)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

    for i_k in range(tl.cdiv(K, BK)):
        p_w = tl.make_block_ptr(w, (T, K), (HV*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))

        p_gk = tl.make_block_ptr(gk, (T, K), (HV*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_gk = tl.load(p_gk, boundary_check=(0, 1)).to(tl.float32)
        b_kg_exp = (b_k * tl.math.exp2(b_gk)).to(b_k.dtype)

        p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_qg = tl.make_block_ptr(qg, (T, K), (HV*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_qg = b_q * tl.math.exp2(b_gk)
        tl.store(p_qg, b_qg.to(p_qg.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min(i_t * BT + BT, T) - 1
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        b_gn = tl.load(gk + last_idx * HV*K + o_k, mask=m_k, other=0.).to(tl.float32)
        b_kg = b_k * tl.where((i_t * BT + tl.arange(0, BT) < T)[:, None], tl.math.exp2(b_gn[None, :] - b_gk), 0)
        p_kg = tl.make_block_ptr(kg, (T, K), (HV*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        tl.store(p_kg, b_kg.to(p_kg.dtype.element_ty), boundary_check=(0, 1))

        b_w = tl.dot(b_A, b_kg_exp)
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check beta scaling equivalence in recompute_w_u_fwd_kda_kernel."
    )
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--T", type=int, default=128)
    parser.add_argument("--H", type=int, default=4)
    parser.add_argument("--HV", type=int, default=4)
    parser.add_argument("--K", type=int, default=128)
    parser.add_argument("--V", type=int, default=128)
    parser.add_argument("--BT", type=int, default=64, choices=(32, 64))
    parser.add_argument("--atol", type=float, default=8e-2)
    parser.add_argument("--rtol", type=float, default=8e-2)
    return parser.parse_args()


def make_unit_lower_A(B, T, HV, BT, device, dtype):
    A = torch.zeros(B, T, HV, BT, device=device, dtype=dtype)
    eye = torch.eye(BT, device=device, dtype=torch.float32)
    for start in range(0, T, BT):
        C = min(BT, T - start)
        lower = torch.tril(0.05 * torch.randn(B, HV, C, C, device=device), diagonal=-1)
        lower = lower + eye[:C, :C]
        A[:, start:start + C, :, :C] = lower.transpose(1, 2).to(dtype)
    return A


def make_inputs(args):
    device = "cuda"
    dtype = torch.bfloat16

    q = torch.randn(args.B, args.T, args.H, args.K, device=device, dtype=dtype)
    k = torch.randn(args.B, args.T, args.H, args.K, device=device, dtype=dtype)
    v = torch.randn(args.B, args.T, args.HV, args.V, device=device, dtype=dtype)
    beta = torch.rand(args.B, args.T, args.HV, device=device, dtype=torch.float32)

    # Keep the cumulative gate range moderate so this example measures beta placement,
    # not exponent overflow behavior.
    g_step = -0.01 * torch.rand(args.B, args.T, args.HV, args.K, device=device)
    gk = torch.cumsum(g_step, dim=1)
    A = make_unit_lower_A(args.B, args.T, args.HV, args.BT, device, dtype)
    return q, k, v, beta, A, gk


def wy_reference(q, k, v, beta, A, gk, BT, scale_A_columns):
    B, T, H, K = k.shape
    HV, V = v.shape[2], v.shape[-1]
    G = HV // H
    dtype = k.dtype

    w = torch.empty(B, T, HV, K, device=k.device, dtype=dtype)
    u = torch.empty(B, T, HV, V, device=k.device, dtype=dtype)
    qg = torch.empty(B, T, HV, K, device=k.device, dtype=dtype)
    kg = torch.empty(B, T, HV, K, device=k.device, dtype=dtype)

    for b in range(B):
        for hv in range(HV):
            h = hv // G
            for start in range(0, T, BT):
                end = min(start + BT, T)
                C = end - start

                b_A = A[b, start:end, hv, :C].float()
                b_beta = beta[b, start:end, hv].float()
                b_v = v[b, start:end, hv].float()
                b_k = k[b, start:end, h].float()
                b_q = q[b, start:end, h].float()
                b_g = gk[b, start:end, hv].float()

                if scale_A_columns:
                    b_A = (b_A * b_beta[None, :]).to(dtype).float()
                    b_u = b_A @ b_v
                    b_w = b_A @ (b_k * torch.exp2(b_g)).to(dtype).float()
                else:
                    b_vb = (b_v * b_beta[:, None]).to(dtype).float()
                    b_kb = (b_k * b_beta[:, None] * torch.exp2(b_g)).to(dtype).float()
                    b_u = b_A @ b_vb
                    b_w = b_A @ b_kb

                b_qg = b_q * torch.exp2(b_g)
                b_gn = b_g[-1:]
                b_kg = b_k * torch.exp2(b_gn - b_g)

                u[b, start:end, hv] = b_u.to(dtype)
                w[b, start:end, hv] = b_w.to(dtype)
                qg[b, start:end, hv] = b_qg.to(dtype)
                kg[b, start:end, hv] = b_kg.to(dtype)

    return w, u, qg, kg


def error_stats(a, b):
    diff = (a.float() - b.float()).abs()
    denom = torch.maximum(a.float().abs(), b.float().abs()).clamp_min(1e-6)
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "max_rel": (diff / denom).max().item(),
        "mean_rel": (diff / denom).mean().item(),
    }


def print_error_stats(name, a, b):
    stats = error_stats(a, b)
    print(
        f"{name}: "
        f"max_abs={stats['max_abs']:.6f}, "
        f"mean_abs={stats['mean_abs']:.6f}, "
        f"max_rel={stats['max_rel']:.6f}, "
        f"mean_rel={stats['mean_rel']:.6f}"
    )


def recompute_w_u_fwd_column_scaled_A(q, k, v, beta, A, gk):
    B, T, H, K = k.shape
    HV, V = v.shape[2], v.shape[-1]
    BT = A.shape[-1]
    BK = 64
    BV = 64
    NT = triton.cdiv(T, BT)

    w = torch.empty(B, T, HV, K, device=k.device, dtype=k.dtype)
    u = torch.empty_like(v)
    qg = torch.empty(B, T, HV, K, device=k.device, dtype=k.dtype)
    kg = torch.empty(B, T, HV, K, device=k.device, dtype=k.dtype)
    recompute_w_u_fwd_kda_kernel_column_scaled_A[(NT, B * HV)](
        q=q,
        k=k,
        qg=qg,
        kg=kg,
        v=v,
        beta=beta,
        w=w,
        u=u,
        A=A,
        gk=gk,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        num_warps=4,
        num_stages=3,
    )
    return w, u, qg, kg


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires CUDA.")
    if args.HV % args.H != 0:
        raise ValueError(f"HV must be divisible by H, got H={args.H}, HV={args.HV}.")

    torch.manual_seed(0)
    print(
        "shape: "
        f"B={args.B}, T={args.T}, H={args.H}, HV={args.HV}, "
        f"K={args.K}, V={args.V}, BT={args.BT}",
        flush=True,
    )
    print("importing kernel...", flush=True)
    from fla.ops.kda.wy_fast import recompute_w_u_fwd

    print("creating inputs...", flush=True)
    q, k, v, beta, A, gk = make_inputs(args)

    print("running Triton recompute_w_u_fwd...", flush=True)
    w, u, qg, kg = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A, gk=gk, q=q)
    torch.cuda.synchronize()

    print("running Triton column-scaled-A variant...", flush=True)
    w_col, u_col, qg_col, kg_col = recompute_w_u_fwd_column_scaled_A(q=q, k=k, v=v, beta=beta, A=A, gk=gk)
    torch.cuda.synchronize()

    print("running PyTorch old-form reference...", flush=True)
    w_old, u_old, qg_ref, kg_ref = wy_reference(q, k, v, beta, A, gk, args.BT, scale_A_columns=False)
    print("running PyTorch column-scaled-A reference...", flush=True)
    w_new, u_new, _, _ = wy_reference(q, k, v, beta, A, gk, args.BT, scale_A_columns=True)

    print_error_stats("original_kernel_vs_column_kernel_w", w, w_col)
    print_error_stats("original_kernel_vs_column_kernel_u", u, u_col)
    print_error_stats("original_kernel_vs_column_kernel_qg", qg, qg_col)
    print_error_stats("original_kernel_vs_column_kernel_kg", kg, kg_col)
    print_error_stats("kernel_vs_old_w", w, w_old)
    print_error_stats("kernel_vs_old_u", u, u_old)
    print_error_stats("column_kernel_vs_column_ref_w", w_col, w_new)
    print_error_stats("column_kernel_vs_column_ref_u", u_col, u_new)
    print_error_stats("old_vs_column_scaled_w", w_old, w_new)
    print_error_stats("old_vs_column_scaled_u", u_old, u_new)
    print_error_stats("kernel_vs_ref_qg", qg, qg_ref)
    print_error_stats("kernel_vs_ref_kg", kg, kg_ref)

    torch.testing.assert_close(w, w_col, atol=args.atol, rtol=args.rtol)
    torch.testing.assert_close(u, u_col, atol=args.atol, rtol=args.rtol)
    torch.testing.assert_close(qg, qg_col, atol=args.atol, rtol=args.rtol)
    torch.testing.assert_close(kg, kg_col, atol=args.atol, rtol=args.rtol)
    torch.testing.assert_close(w, w_old, atol=args.atol, rtol=args.rtol)
    torch.testing.assert_close(u, u_old, atol=args.atol, rtol=args.rtol)
    torch.testing.assert_close(w_col, w_new, atol=args.atol, rtol=args.rtol)
    torch.testing.assert_close(u_col, u_new, atol=args.atol, rtol=args.rtol)
    torch.testing.assert_close(w_old, w_new, atol=args.atol, rtol=args.rtol)
    torch.testing.assert_close(u_old, u_new, atol=args.atol, rtol=args.rtol)
    torch.testing.assert_close(qg, qg_ref, atol=args.atol, rtol=args.rtol)
    torch.testing.assert_close(kg, kg_ref, atol=args.atol, rtol=args.rtol)
    print("all checks passed")


if __name__ == "__main__":
    main()
