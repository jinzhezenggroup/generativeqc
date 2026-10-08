# Decision record: HF/SCF decomposition and acceptance provenance

Status: implemented (historical architecture decision; follow-up runtime issues are separate)
Date: 2026-10-09
Historical work: [#240](https://github.com/jinzhezenggroup/generativeqc/issues/240), closed 2026-09-22

## Problem

The original `rhf.cpp`, `cuda_rhf.cu` and
`cuda_density_fitting.cu` combined reference mathematics, SCF host
control, GPU numerical kernels, provider/metric planning, Graph lifetime,
warm-cache admission, gradients and CUDA library resource lifetimes.
An implementation-only edit could require compiling or reviewing unrelated
scientific CUDA kernels. A reduction in counted CUDA lines could also be
mistaken for retirement of native scientific arithmetic.

## Decision and stages

Use distinct owners for scientific equations, method-specific SCF state,
prepared providers, generic GPU runtime, direct CUDA plan/Graph lifetime,
integral-source execution and stationary derivatives.

The implemented stages were delivered through:

- #262: reference linear algebra, mean-field reference operations and
  initial state preparation;
- #264: host iteration/DIIS/proposal and gradient assembler;
- #265: host topology, resource sizing, queue planning and symmetric eigensolve;
- #268/#269: DF source/export and plan/J/K/replay/response ownership;
- #271/#272: shared SCF device kernels, native libraries/resources;
- #273/#274: direct queue/compaction and direct J/K/one-electron host APIs;
- #276/#277: retained numerical CUDA families and ordinary C++ execution host;
- #456: direct-HF bucket/cache and CUDA Graph lifetime extraction.

This is a **completed structural decomposition**, not a claim that each
native scientific CUDA formula was removed, or that all follow-up numerical
issues closed on 2026-09-22. The source-by-source current map is in
[HF and SCF module boundaries](../../../../docs/developer/scf_module_boundaries.md).

## Rejected alternatives and invariants

- Splitting the direct launch transaction solely to reach a size limit
  would introduce wide callback/context interfaces without an independent
  lifetime. The remaining large host execution owner is an explicit review
  exception, not undecomposed Graph/cache ownership.
- Moving Graph handles to generic resources would conflate direct-HF
  capture identity with common stream/library lifetimes.
- Moving the same numerical equations unchanged between `.cu` and
  `.cpp` does not satisfy generated-science retirement; the
  [semantic CUDA ownership ledger](../../../../docs/cuda_ownership/README.md)
  remains authoritative for that claim.

Preserve source, determinant and plan identities; exact RHF/UHF spin and
force conventions; provider/final-Fock validation; DIIS/proposal behavior;
Graph-before-stream teardown; launch/error ordering; and bounded
resource/failure semantics.

## Historical evidence and its limits

Versioned source snapshots, objects, Slurm device outcomes, reference
comparison results and sample timings live under
[SCF direct-native results](../../../../benchmarks/results/scf-direct-native-rtx5090/README.md)
(including `validation.json`, `incremental.json` and `integration.json`).
The original extraction also recorded development-only compiler samples:
those showed narrower rebuild ownership but did not by themselves prove
whole-build or molecular endpoint speedups.

The final host-control slice's `91dfd97` source measured a 4,892-line
`cuda_rhf.cpp` before separation; the extracted candidate reported
4,452 lines in the residual driver with separate bucket/Graph owners.
These are **pinned historical file counts**, not an inventory of current
`master`. Detailed object sizes, source audit and rejected design options
also appear in the earlier
[direct-HF lifetime decision](2026-09-18-direct-hf-host-control.md).

The subsequent
[production-acceptance audit](../../../../benchmarks/results/acceptance-closeout-20260922/README.md)
pinned a CUDA 12.9/sm_120 Release source and a clean build. Its native
CTest and Python HF/DF/MP2 runtime gates did not completely pass;
[#240](https://github.com/jinzhezenggroup/generativeqc/issues/240)
separates the delivered structural work from runtime reconciliation.
[Issue #1041](https://github.com/jinzhezenggroup/generativeqc/issues/1041)
continues to track charged DF-UHF convergence. There is no basis for
relabeling a historical failed test as passed after unrelated changes.

## Consequences and revisit criteria

Keep the active documentation short and organized by ownership rather than
PR chronology. For new refactors, measure **which translation units rebuild**
and **which source-matched molecular endpoints pass**, not only aggregate
CUDA file count. Revisit the large coupled execution owner only if a
new independent state/lifetime boundary becomes evident or if a measured
bottleneck justifies a different composition.

This note is a retrospective index of prior decisions and retained evidence,
not a replacement for contemporaneous measurements or a new CI result.
