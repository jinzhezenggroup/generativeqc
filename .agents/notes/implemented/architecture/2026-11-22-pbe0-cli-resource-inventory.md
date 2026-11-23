# Decision: reuse the native CPU KS capacity for PBE0 CLI dry runs

Status: implemented
Date: 2026-11-22

CPU conventional LDA/PBE and PBE0 use the same shape-based host matrix and
full ERI provider capacity. The shared native planner already accounts for the
first-AO-derivative cache used by PBE0. Route canonical CPU PBE0 RKS/UKS to
that planner instead of adding a hybrid-specific estimator. CPU named
PBE0-D3(BJ) RKS uses the exact generated total-method record and the native
D3 capacity query; add the retained per-item correction owners and enforce
both the equal-share correction cap and the aggregate host cap.

Do not use the PBE0 carrier enum to authorize another hybrid or a correction.
CUDA hybrid arenas, forces, Hessians, DF and incremental J/K are outside this
inventory. A feasible dry run bounds the reported owners; it is not provider
admission or a cap on process RSS.

The CLI tests compare batch capacities to the native KS fleet ABI and the
native D3 resource ABI, including exact and one-byte-short limits. The
installed CLI reported feasible bare RKS, bare UKS and corrected RKS plans
and ran the corrected energy with Python and the compiler absent from the
runtime search path. See `docs/user/native_cli.md` for the current contract.

Refs #2227 (R2/R5).
