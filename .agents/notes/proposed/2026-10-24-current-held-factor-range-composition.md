# Candidate: restore held fitted factors on the current cold endpoint

Status: proposed; current-source GPU and endpoint qualification pending
Date: 2026-10-24

## Existing rationale and current diagnosis

This reuses the held-density restoration and fitted-primary rebase changes from
`c7eec7ed780e2a72b449edf995c8041ec30806f7`, described in that revision's
`.agents/notes/implemented/performance/2026-10-23-held-density-factor-and-fitted-range-cache.md`.
Those changes are absent from the current isolated XC/GEMM source. Its separate
omegaB97M-V DF trace records substantial full-rank force panel products, even
though most SCF iterations already use occupied exchange. Historical timings
and tests qualify their original source; they are not current-source results.

## Composition and invariants

Restore only the restricted AO frame previously consumed by occupied K when
physical confirmation intentionally holds D unchanged. Never adopt the newer,
unaccepted eigensolver proposal. Missing, orthonormal, dense and unrestricted
frames retain invalidation, and failed copies leave readiness false.

During exact-range energy rebasing, the optional source changes only LR K.
Keep the unchanged fitted primary J/K only when its final projection and warm
factor remain authenticated. This preserves the final U lease and reports only
operators actually executed. Original source, density, occupation, scratch and
force lifetime checks remain authoritative.

The predecessor's new exact-LR cache remains explicit, strict and SPD. This
follow-up does not expand automatic admission, resource ceilings, precision,
convergence, screening, metric cutoff or scientific acceptance gates. It also
does not import the historical stack-limit or DF budget changes.

## Qualification

Reuse the executable held-factor fault/identity cases and focused native
restricted/unrestricted fitted WB comparison with independent CPU composition.
Run the current complete range replay, policy invalidation, fitted E/F, moved
geometry and sanitizer suites. Compare complete water96 endpoints with retained
source/binary identities; density and force gates apply to every observation.
