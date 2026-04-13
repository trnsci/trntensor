"""
DF-MP2 energy via tensor contractions (trntensor.einsum).

Same calculation as trnblas/examples/df_mp2.py, but expressed naturally
as einsum contractions instead of GEMM decomposition.

Compare:
    trnblas: T = B[i] @ B[j].T          # loop over i,j pairs
    trntensor: T = einsum("ap,bp->ab", B[i], B[j])  # same but clearer intent

The einsum planner can detect these patterns and fuse them.

Usage:
    python examples/df_mp2_einsum.py --demo
"""

import argparse
import time

import torch

import trntensor


def df_mp2_energy(B: torch.Tensor, eps_occ: torch.Tensor, eps_vir: torch.Tensor) -> float:
    """DF-MP2 energy from DF coefficients B_ia^P.

    B: (nocc, nvir, naux)
    eps_occ: (nocc,)
    eps_vir: (nvir,)
    """
    nocc = B.shape[0]

    e_mp2 = 0.0
    for i in range(nocc):
        for j in range(nocc):
            # T_ab = Σ_P B_ia^P B_jb^P
            T = trntensor.einsum("ap,bp->ab", B[i], B[j])

            # Energy denominators
            denom = eps_occ[i] + eps_occ[j] - eps_vir.unsqueeze(1) - eps_vir.unsqueeze(0)

            # MP2 energy: (2T - T^T) * T / denom
            e_mp2 += (T * (2 * T - T.T) / denom).sum().item()

    return e_mp2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--nocc", type=int, default=5)
    parser.add_argument("--nvir", type=int, default=19)
    parser.add_argument("--naux", type=int, default=72)
    args = parser.parse_args()

    if args.demo:
        args.nocc, args.nvir, args.naux = 5, 19, 72

    print("DF-MP2 via einsum:")
    print(f"  nocc={args.nocc}, nvir={args.nvir}, naux={args.naux}")
    print("  Contraction: T_ab = einsum('ap,bp->ab', B[i], B[j])")

    torch.manual_seed(42)
    B = torch.randn(args.nocc, args.nvir, args.naux) * 0.1
    eps_occ = -torch.sort(torch.rand(args.nocc))[0] - 0.5
    eps_vir = torch.sort(torch.rand(args.nvir))[0] + 0.1

    # Check that the planner detects matmul
    plan = trntensor.plan_contraction("ap,bp->ab", B[0], B[1])
    print(f"  Planner strategy: {plan.strategy}")
    flops_per_pair = trntensor.estimate_flops("ap,bp->ab", B[0], B[1])
    total_flops = flops_per_pair * args.nocc * args.nocc
    print(f"  Estimated FLOPs: {total_flops:,.0f}")

    t0 = time.perf_counter()
    e_mp2 = df_mp2_energy(B, eps_occ, eps_vir)
    elapsed = time.perf_counter() - t0

    print(f"\n  E_MP2 = {e_mp2:.10f} (synthetic)")
    print(f"  Time:  {elapsed:.3f}s")


if __name__ == "__main__":
    main()
