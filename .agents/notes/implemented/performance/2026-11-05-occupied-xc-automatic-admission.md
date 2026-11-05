# Decision: automatically admit bounded occupied-XC initializers

Status: implemented
Date: 2026-11-05

## Problem

The recovered shortcut in the [audited-proposal decision](2026-11-04-occupied-xc-audited-proposals.md)
remained an opt-in even after complete-endpoint qualification found a profitable
domain. Requiring an environment override would leave ordinary eligible users on
the slower matrix initializer. Default promotion must change selection, not
silently weaken physical acceptance or relabel historical opt-in measurements.

## Decision

Unset `GENERATIVEQC_CUDA_KS_OCCUPIED_XC` now permits automatic admission, as
does exact `1`. Explicit `0`, empty and malformed values retain matrix-only
work. The existing native owner still decides whether a canonical frame,
strict-FP64 meta-GGA layout, occupied rank, scratch lease and residual regime
permit proposals. Unsupported and near-stationary cases keep original-D work.

The one-panel non-certifying projection and bounded initializer lifecycle are
unchanged. A permanent original-D target handoff starts a fresh energy epoch;
only the strict physical/canonical gates certify convergence and force
publication. StrictAudit and response consumers never use provisional features.
Neither default selection nor explicit `1` bypasses these invariants.

## Rejected alternatives

- Retaining an opt-in after qualification hides the recovered benefit.
- Restricting the default to a hardcoded molecule size or benchmark method would
  substitute an example for the actual admission conditions.
- Calling historical explicit-1 timings default-path evidence obscures which
  native selection branch was exercised. New receipts record actual absent
  environment values and retain backward-compatible v1 replay.
- Requiring per-point full matrix recomputation restores the cost removed by
  the optimization, without replacing strict endpoint acceptance.

## Evidence and revisit conditions

The earlier source-bound opt-in qualification remains historical evidence, not
a measurement of this new selector. Default-path qualification must compare
actual unset controls with explicit `0`, include complete native cold/warm/
moved E+F and semantic work, and preserve independent and paired scientific
gates. Host execution tests cover unset, exact 1, 0, empty and malformed values;
native tests exercise automatic and explicit admission and a strict matrix tail.
The retained cost floor is `max(2%, 2 * sum(relative MADs))` over three
alternating fresh-process pairs on the original 96-atom Direct Standard
r2SCAN-3c workload: cold must be profitable and frozen warm must not regress
beyond that floor. Small endpoints preserve the existing independent and paired
`2e-9 Eh` / `1e-7 Eh/Bohr` gates without promising a broad speedup.

Revisit automatic admission on a credible endpoint regression, not an isolated
kernel timing. Retain the explicit rollback and original-D publication contract.
This supersedes only the opt-in selection policy in the audited-proposal note,
not its scientific boundaries or the lower-level exact-identity contract.

## Default-path observation

At immutable `ddadce1e29453e0f5cabc4c12948c7c798e8d66c`, the actual unset
control gives complete cold E+F medians `30.439342 -> 28.893960 s` on the
original 96-atom workload (5.077% less time, above the retained 2% floor), with
10 iterations/Fock builds on both routes. Frozen warm
`11.216157 -> 11.179427 s` is below the floor and is not a speedup claim.
Eighty small independent/paired cold/warm/moved E+F endpoints and all twelve
large reference/paired endpoints retain their original scientific gates.
The [default qualification capsule](../../../../benchmarks/results/occupied-xc-default-proposals-20261105/README.md)
keeps actual absent controls, full vectors, work and source/binary provenance
separate from the earlier explicit-1 measurements.
The subsequent delivery merge retains the PR branch's existing upstream/CI
integration and raw-output guards. It does not turn the pinned default image
measurements into merged-target qualification or refresh scientific claims.

## Bounded admission correction

Source review found that first admission on the final permitted submit could
consume the remaining budget without a strict target evaluation. The owner now
requires two later strict submits inside the unchanged budget, because the
first strict Fock starts a fresh energy epoch. The handoff reserves that same
two-evaluation tail while preserving the existing eleven-proposal cap. It also requires a nonzero
occupied rank below the maximum prepared active-AO count: if every tile retains
matrix products, no provisional epoch or unnecessary target handoff is started.
The executable lifecycle regression fails on the prior owner and passes with
budgets 1, 2, 3, 4, 5, 12 and 200 and RKS/UKS capacity boundaries. No scientific gate or
kernel arithmetic is relaxed. The original source-bound GPU receipts are reused,
not relabeled as qualification of this bounded admission correction.

The benchmark reader now rejects duplicate/missing oracle keys and jointly
truncated oracle/observation force vectors. Historical v1/v2 receipts bind to the
runner's fixed three-atom water case without rewriting retained bytes; new
receipts also record the atom count. Streaming library hashing supports Python
3.10 without changing the digest. Existing numerical and cost gates remain
unchanged.

The subsequent [cross-functional survey](2026-11-05-occupied-xc-functional-survey.md)
finds a replicated M06-2X gain and MN15/UKS counterexamples. It supplements the
profitability scope without changing this original observation or selector.
