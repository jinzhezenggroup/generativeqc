# Decision: scope Libxc invalidation away from post-HF generators

Status: implemented
Date: 2026-11-15

## Problem

The native source census includes code generators as well as compiled owners.
The direct/all-electron semilocal Libxc energy/force dependency policy already
excludes post-HF source owners, but seven generators feeding those owners were
still treated as unknown shared dependencies. CUDA kernel sharing in PR #3096
therefore altered both CPU and CUDA Libxc revalidation plans without changing
either measured consumer.

## Decision and boundaries

List the seven post-HF-only generators explicitly. The fitted AO source
generator emits `generated_df_ao_source` for `src/posthf/df_interaction_source.cpp`;
the other six generate DF-CCSD/RCCSD/UCCSD code. Their shared helpers also feed
the RHF frame response consumed by `src/methods/df_ccsdt_force.cu`, not the
measured direct semilocal E/F context. This is not a wildcard exemption
for future DF generators or shared compiler lowering. Nuclear repulsion, solver
lowering, unknown build inputs, public ABI and build configuration remain
conservative dependencies.

The same PR also changes `integral/weighted_eri_cuda.py`. That shared integral
generator remains a genuine CUDA dependency. Refresh only the derived snapshot
to remove the seven disjoint inputs; its CUDA revalidation rows continue to
include the weighted integral change. No current
endpoint is promoted, no historical measured receipt or library hash is changed,
and no GPU qualification or numerical parity is inferred from host tests.

## Verification

Replay tests exercise all seven disjoint generator changes against retained
authenticated receipts; measured contexts survive. A shared weighted CUDA edit
still invalidates CUDA energy/force contexts while preserving CPU contexts.
Unknown DF generator additions/deletions and real nuclear/solver changes remain
invalidating. The offline catalog checker passes in ordinary and optimized
Python after the derived snapshot refresh. Its scientific summary, receipts,
blockers and endpoint counts are unchanged from the pre-repair snapshot.

Revisit an exemption if one of these generators gains a semilocal consumer.

The publication patch is replayed on master `01068725898ec46bc2b0ddfd35772e4cf44874cb`,
which includes actual PR #3096 squash `28d142e6` through `cc0e4dd9`. It does not
include the later author-head DF Lambda pooling changes. Official regeneration updates scoped revalidation metadata
only; scientific summaries, historical receipts and qualification stay unchanged.
