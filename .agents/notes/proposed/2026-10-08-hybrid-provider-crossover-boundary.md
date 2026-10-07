# Decision: label the supported hybrid provider arms before timing

Status: proposed
Date: 2026-10-08

## Problem

#2054 asks for a Direct, DF-J plus exact-K, and full DF-JK complete-endpoint
comparison. The current public KS selector exposes conventional or fitted
providers, while the CUDA KS admission rejects fitted Coulomb with exact
exchange. A timing script that silently routes this third arm would compare
different approximations under one name.

## Decision

The #2054 pilot records explicit provider proof after native execution.
Direct J/K and full DF-JK can enter a source-matched ABBA E+force pilot. The
DF-J plus exact-K arm is `UNSUPPORTED` until the production provider contract
actually admits it. No production selector or provider is changed by this
benchmark. Each supported arm has its own PySCF oracle using the frozen grid
and matching auxiliary identity.

## Rejected alternatives

Treating full DF-JK as DF-J plus exact-K, inferring provider identity from a
requested flag, or lowering tolerances after a failed attempt would make the
comparison irreproducible. Allocation ledgers cannot stand in for measured
peak memory.

## Revisit when

The public KS provider contract admits mixed J/K approximations with live
proof, complete forces and an independent matched oracle, or a reviewed
source-matched device campaign covers the full #2054 case and resource matrix.

## References

Issue #2054; `python/generativeqc/_ks_snapshot.py` provider proof;
`src/dft/cuda_ks.cpp` mixed-provider admission;
`benchmarks/results/hybrid-provider-crossover-2054/README.md`.
