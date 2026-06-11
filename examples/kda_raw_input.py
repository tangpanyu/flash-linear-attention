import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fla.ops.kda import chunk_kda


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires CUDA.")

    torch.manual_seed(0)

    B, T = 1, 128
    H = 16
    HV = 16
    K = 128
    V = 128
    device = "cuda"
    dtype = torch.bfloat16

    q = torch.randn(B, T, H, K, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, T, HV, V, device=device, dtype=dtype, requires_grad=True)

    # Raw projection outputs. chunk_kda will transform them internally.
    g_raw = torch.randn(B, T, HV, K, device=device, dtype=dtype, requires_grad=True)
    beta_raw = torch.randn(B, T, HV, device=device, dtype=dtype, requires_grad=True)

    A_log = torch.zeros(HV, device=device, dtype=torch.float32, requires_grad=True)
    dt_bias = torch.randn(HV * K, device=device, dtype=torch.float32, requires_grad=True)

    o, final_state = chunk_kda(
        q=q,
        k=k,
        v=v,
        g=g_raw,
        beta=beta_raw,
        A_log=A_log,
        dt_bias=dt_bias,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=True,
        lower_bound=-5.0,
        output_final_state=True,
        chunk_size=64,
    )

    loss = o.float().square().mean()
    loss.backward()

    print(f"o.shape: {tuple(o.shape)}")
    print(f"final_state.shape: {tuple(final_state.shape)}")
    print(f"loss: {loss.item():.6f}")
    print(f"g_raw.grad is None: {g_raw.grad is None}")
    print(f"A_log.grad is None: {A_log.grad is None}")
    print(f"dt_bias.grad is None: {dt_bias.grad is None}")


if __name__ == "__main__":
    main()
