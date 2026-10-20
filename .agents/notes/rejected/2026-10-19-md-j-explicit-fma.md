# Rejected promotion: local explicit FP64 FMA in the MD value recipe

Status: rejected for production promotion; numerically qualified candidate retained
Date: 2026-10-19

## Why it was tested

The continued #2609 source audit found another real difference: our CUDA build
uses `--fmad=false`, while the local pinned upstream MD target permits NVCC's
default implicit fusion. The installed wheel's flags are not inferred from that
source build. On n4:1181, the upstream generic single-DM kernel translation unit
alone was rebuilt with implicit FMA disabled. Complete same-density r2 J changed
from0.683916s to0.884015s. Its matrix changes by at most1.42e-14. All other
objects, helpers, screening, task schedule and compiler language mode remained
unchanged. This is a source-matched ablation, not a production precision change.

n4:1182 repeats at1e-22 nominal upstream screening: generic1.251107s,
no-FMA1.585252s, max matrix difference1.78e-14. The latter is close to native
primitive1.561526s, but upstream heuristic1e-22 and native cumulative1e-12
omission are NOT equivalent scientific error contracts. Do not use this as a
proof that the remaining native cost is necessary for accuracy or fully explained.

## Candidate and independent validation

Use existing `Graph.materialization_plan(...fusion=AlgebraFusion.FMA)` in the
compiler's MD radial stages, explicit FP64 FMA in forward/reverse Hermite dots,
and explicit FMA in the native bra accumulation. Do not change global flags,
Boys, geometry/E transforms, or outward-rounded screening certificates.

The initial host run has68 cross-recurrence comparison failures: the old gate
was relative to a nearly cancelled separately rounded result. A retained
450-case80-digit diagnosis shows zero independent-gate failures in either
algorithm. Both worst independent errors are3.74e-13 of the absolute-term bound;
fused-versus-separate difference is at most3.23e-16 of that bound. The candidate
compared cross algorithms at the same2e-14 coefficient but against the independent
absolute-term bound, and retained the original3e-12 high-precision gate on BOTH.
All816 host cases then passed. This candidate test change was reverted with the
candidate; no numerical test weakening was retained to force a promotion.

n2:3482 has90 GPU passes, one unavailable fully pinned package deselection;
the two actual kernel/capture/tail cases pass each of memcheck, initcheck,
racecheck and synccheck, zero errors and race zero warnings/hazards. These are
small sanitizer systems, not1088-AO full sanitizer coverage.

The first GPU deployment retained stale stationary-force contract manifests.
n2:3481 failed two UKS force cases, and n4:1183's133.30s result had no forces and
failed its gate. Neither is a numerical or performance result. Genuine CMake
manifest targets regenerated identities while verifying actual emitted sources
and library hashes; no identity was manually overridden. The complete corrected
endpoint is n4:1184 below.

## No demonstrated performance benefit

| Complete operation (s) | Retained separate arithmetic | Explicit local FMA |
|---|---:|---:|
| Fixed-D r2SCAN-3c J, n=5 median | 1.561526 | 1.556471 |
| Fixed-D PBE0 J, n=5 median | 0.487096 | 0.485576 |
| r2SCAN-3c cold E/F, n=1 | 157.723791 | 157.698189 |
| r2SCAN-3c warm E/F | 26.050681 (n=5 median) | 25.926579 (n=1) |

Both cold endpoints have24Fock and pass the original independent E/F gates.
The0.026s cold difference is not a demonstrated speedup. The upstream FMA
ablation did not predict native recipe economics. Preserve the qualified
separate arithmetic and its exact source/binary identity instead of retaining
an additional numerical change with no endpoint benefit. The candidate was
archived and reverted before completion; the final production path does NOT
include these explicit FMA edits.

Candidate source identity:
`7f3e9973366cff5634364011241a0b6fb6e91df009a10bcf6fde405451a1509f`.
Candidate main library:
`0b218f5ed53425738616e7f848d82f094e3abac5e8e8539eeabf77fe7cd7a292`.
Evidence/code: `.artifacts/pr2609/fma/` (including `candidate/`, original
failures, high-precision diagnosis, actual build/cache commands and corrected
Slurm receipts), `.artifacts/pr2609/audit/n4/{1181,1182}`. Immutable candidate
source/runtime snapshots remain at `/data/jzzeng/qc-pr2609-fma-20261019` on n2/n4.

## Revisit

A different native layout/schedule makes its contractions compute-bound, or a
separately identified full-translation-unit arithmetic recipe establishes
independent scientific and complete endpoint gains. This test only rejects
promotion of the measured local FMA candidate, not all possible uses of FMA.

[Full source audit](../implemented/performance/2026-10-19-gpu4pyscf-cold-source-audit.md)
and [retained spatial execution](../implemented/performance/2026-10-19-xc-spatial-grid-order.md).

Final restoration was rebuilt with the verified compiler cache. Both the source
identity and main-library SHA256 exactly match the qualified spatial receipt
n4:1178 (`.artifacts/pr2609/spatial/final-identity.json`). All six local stationary
AOT manifests were regenerated through their genuine CMake targets for the
restored compiler closure, so no rejected-candidate force identity remains.
