import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fla.ops.kda import chunk_kda


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires CUDA.")

    torch.manual_seed(0)

    seq_lens = [70, 33, 101]
    cu_seqlens_cpu = torch.tensor([0, *torch.tensor(seq_lens).cumsum(0).tolist()], dtype=torch.long)
    cu_seqlens = cu_seqlens_cpu.cuda()
    total_tokens = int(cu_seqlens_cpu[-1])
    num_sequences = len(seq_lens)

    # In cu_seqlens mode, physical B must be 1 and all sequences are packed on T.
    B = 1
    H = 16
    HV = 16
    K = 128
    V = 128
    device = "cuda"
    dtype = torch.bfloat16

    q = torch.randn(B, total_tokens, H, K, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, total_tokens, H, K, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, total_tokens, HV, V, device=device, dtype=dtype, requires_grad=True)

    # Raw projection outputs. chunk_kda will transform them internally.
    g_raw = torch.randn(B, total_tokens, HV, K, device=device, dtype=dtype, requires_grad=True)
    beta_raw = torch.randn(B, total_tokens, HV, device=device, dtype=dtype, requires_grad=True)

    A_log = torch.zeros(HV, device=device, dtype=torch.float32, requires_grad=True)
    dt_bias = torch.randn(HV * K, device=device, dtype=torch.float32, requires_grad=True)

    # There is one recurrent state per logical sequence, not per physical batch.
    initial_state = torch.zeros(
        num_sequences,
        HV,
        K,
        V,
        device=device,
        dtype=torch.float32,
    )

    o, final_state = chunk_kda(
        q=q,
        k=k,
        v=v,
        g=g_raw,
        beta=beta_raw,
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=True,
        lower_bound=-5.0,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        chunk_size=64,
    )

    loss = o.float().square().mean()
    loss.backward()

    print(f"seq_lens: {seq_lens}")
    print(f"cu_seqlens_cpu: {cu_seqlens_cpu.tolist()}")
    print(f"q.shape: {tuple(q.shape)}")
    print(f"o.shape: {tuple(o.shape)}")
    print(f"initial_state.shape: {tuple(initial_state.shape)}")
    print(f"g_raw.shape: {tuple(g_raw.shape)}")
    print(f"final_state.shape: {tuple(final_state.shape)}")
    print(f"loss: {loss.item():.6f}")
    print(f"g_raw.grad is None: {g_raw.grad is None}")
    print(f"A_log.grad is None: {A_log.grad is None}")
    print(f"dt_bias.grad is None: {dt_bias.grad is None}")


if __name__ == "__main__":
    main()
