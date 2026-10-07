# Decision: CUDA MINAO borrows its target's idle GPU eigensolver

Status: implemented
Date: 2026-10-07

Default selection is subsequently superseded by
[the guarded CPU/CUDA promotion](2026-10-07-minao-default.md). The measurements
below retain their frozen pre-promotion source/binary identity.

## Problem

The opt-in MINAO provider from #2068 constructs no preliminary Fock, but its
overlap, ensemble construction and strict admission called independent CPU
Jacobi decompositions even for CUDA targets. Strict CUDA initial-density upload
also repeated CPU decomposition. The reference eigensolver scans the whole
upper triangle for every pivot, so iterations saved alone cannot establish a
cheap seed or net cold benefit.

## Decision

Pass a borrowed `EigenOperation` through the method owner, MINAO preparation,
occupation construction and unchanged shared validator. CUDA supplies its
existing synchronous `seed_eigen_operation`, using the charged idle solver
workspace and dead DIIS buffers. Strict CUDA upload uses the same operation.
CPU callers keep the default independent reference implementation. MINAO never
retains the callback and allocates no separate GPU owner or solver workspace.
Strict RKS upload checks the total electron count of its spin-summed density,
not the half-sized occupied-orbital count; the real-device regression exposes
the previous mismatch without changing the shared validator or its tolerances.

Native rectangular overlap, host matrix products and bounded occupation
projection remain host operations. This moves decompositions to the GPU; it
does not claim fully resident projection or eliminate transfers. Construction,
transfers, validation, target attempts and forces all remain charged to the
complete cold endpoint.

## Invariants

- Preserve pinned ANO projection, electron count and ensemble constraints.
- Preserve the overlap singularity cutoff, shared seed gate, target scientific
  settings and independent final-state/derivative gates.
- Check the narrow numeric cap before backend invocation. Retain conservative
  host inventory and already charged CUDA solver resources.
- Preserve warm/checkpoint precedence and one fresh Hcore target retry.
- A failing GPU operation does not silently retry CPU MINAO Jacobi.
- Keep `initial_guess=None` as the default while measuring profitability.

## Evidence

Native callback tests require all four preparation decompositions to use the
injected operation and preserve the fixture density. A real-device case checks
zero reference decompositions during preparation and strict upload, transfer
accounting, independent CPU PBE0 energy/density and budget preflight. Existing
retry control tests protect failure bounds and warm precedence.

Complete profitability is measured separately on frozen-source, same-binary
Hcore/MINAO CUDA cold endpoints with independent GPU4PySCF energy/force gates.
Retain source/binary identities and negative samples with any speedup claim;
test design alone is not a performance win or default qualification.

The 2026-10-07 n1 RTX 5090 run uses frozen parent
`2b0feff1d5577e58f3a44fbecbc1974be92df4e3` plus this repair, one identical
binary for both arms, strict FP64 exact PBE0/def2-SVP, and three interleaved
fresh-owner E+force samples per arm. Complete cold medians are:

| Domain | Hcore seconds | GPU MINAO seconds | Wall reduction |
| --- | ---: | ---: | ---: |
| 48-atom water / 384 AOs | 99.047145 | 86.841994 | 12.32% |
| 96-atom water / 768 AOs | 211.832632 | 181.389462 | 14.37% |
| 16-atom peroxide holdout / 152 AOs | 40.162046 | 35.969546 | 10.44% |

MINAO preparation medians are 0.318152, 2.337477 and 0.034897 seconds,
respectively. Every cold sample converges and passes independent E/force gates;
all seeded attempts report zero preliminary Fock builds, one target attempt
and a complete work census. Warm replay skips preparation. Maximum paired
native density difference is 3.10e-11. The repaired source passes 178 focused
Python tests and the CPU-preliminary/native-GPU contract executables.

Raw records, densities, harnesses and source/binary identities are retained in
`/data/jzzeng/qc-2068-minao-profit-20261007` on n1, mirrored in
`/home/jzzeng/codes/qc-branch-audit-20260922/evidence-2068`. These finite-domain
samples do not qualify blanket CPU/HF/UKS/DF defaults or whole-process peak
resources. No default changes or molecule/GPU-name selection are introduced.
Raw JSON records and the frozen patch are also published in the
[qualification bundle](../../../../benchmarks/results/minao-default-20261007/README.md).

## Rejected Alternatives

Do not relax occupation validation, replace MINAO with preliminary SCF, create
an unrelated solver owner, or seed the native target from another engine.
Do not promote a default solely from iterations saved.

## References

#2068, #2052; the prior MINAO ensemble-admission note retains numerical
construction rationale and conservative numeric-cap inventory.
