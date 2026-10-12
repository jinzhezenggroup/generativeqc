# Decision: default to admitted retained-shell RHF cross derivatives

Status: implemented
Date: 2026-10-12

Supersedes the default-policy portion of
[the initial opt-in decision](2026-10-12-rhf-shell-bilinear-derivative.md).

## Problem

Symmetric polarization preserves shell reuse but repeats the same nuclear ERI
derivatives twice. The compiler-owned density-product directional derivative
now permits one traversal without replacing the shell scheduler, primitive
cache, generated recurrences or permutation scatter. The earlier generic
canonical single-pass experiment did not have this property and regressed.

## Decision

`RHFFrameResponseOptions` requests the retained-shell bilinear schedule by
default. Admission requires the generated force lease, numerical audit storage,
one batch item and zero screening. Canonical ERI residency alone is not admission.
The shell API also verifies derivative order, matrix dimensions, device ownership
and finite input/output values before publishing a result.

Missing shell admission or `NOT_IMPLEMENTED` preserves the bounded symmetric
two-pass schedule. Numerical, allocation and submission failures propagate;
they do not silently retry a different scientific path. The response owner
drains pending work before a refused optional shell call releases host storage.
Explicit bilinear opt-out and legacy three-pass validation remain available.
The low-level canonical bilinear probe remains an independent validation route,
not an automatically selected production fallback.

Only completed shell passes are reported as shell passes. Canonical jet census
fields are not invented for a generated shell traversal. The endpoint accepts
`auto` to exercise the actual options defaults without overriding them, and the
native RHF probe has a separate default-policy schedule.

## Qualification

Qualification freezes master `a50eb54f362a1b27edcb0f06353e7a04aa9979e2`
and retains the reconstruction patch, source/binary hashes, verified compiler
cache receipts and finite Slurm allocations. An independently built untouched
stock source, rather than only the candidate's opt-out, is the endpoint baseline.
Endpoint comparisons use fresh processes in ABBA order on the same physical GPU,
strict FP64 and unchanged Q batches and other optimization controls.

Acceptance requires independent energies, retained qualified complete forces,
the explicitly scoped large finite-difference observations, Lambda/Z residuals,
frame stationarity, translation and numerical repeatability. Whole-force
independent small-molecule tests supplement the large retained force comparison;
two large finite-difference components are not described as full large-force
finite-difference qualification. Real-device signed shell/canonical oracles and
default-policy RHF cases cover spherical/Cartesian and through-f sources.

[The source-matched publication](../../../../benchmarks/results/rhf-shell-bilinear-default-20261012/README.md)
retains four fresh complete endpoints from Slurm job 778 on node4, physical GPU
`GPU-5430ba2e-194a-b6d1-f12c-6b99f6d34efd`. Mean native E+F is
`250.408748 -> 227.374415 s` (9.199% less time); the nuclear two-electron phase
is `46.119834 -> 22.998365 s`. Two shell passes become one and generic passes
remain zero. CCSD 19 iterations, Lambda 21 iterations/22 solver actions,
416 auxiliary batches and the accepted physical root are unchanged. These
two-per-side ABBA observations are scoped evidence, not a fleet-wide or
statistically generalized performance claim.

The final rebuilt library has SHA256
`7a935449caffdc1c7a39fe3ac0c9bc0db5124c1ba14b5e3601b8bab9025928ba`.
Slurm job 1486 on node5 passes all 116 real-device RHF/complete-DF tests with
no skips; job 1487 passes 32 selected default/signed derivative cases under
memcheck with zero errors and zero leaked allocations. The selected host
suite passes 728 tests; its 39 skipped native cases are not counted as device
qualification. The default constructor and admission tests execute real C++
contracts, and the promotion inventory retains controls after their default
changes from false to true.

Initial node2 jobs 2929/2930 stop before Python tests because NVML reports a
driver/library mismatch. A node5 runner-transfer retry also precedes numerical
execution. Failed receipts are retained separately; none counts as a passing
test. Node5 uses an isolated Python/oracle environment, never a production
PySCF dependency. All real-device work remains inside finite Slurm allocations
with assigned visibility preserved.

The source ownership audit passes with net handwritten scientific CUDA +2 LOC
and runtime CUDA -4 LOC, with no ownership reclassification. Generated weights
remain compiler-owned. The optional build-output ownership scan fails on both
untouched stock and candidate because upstream families claim
`generated_rccsdt_cuda.cu` twice; retain those identical failure receipts rather
than weakening the ledger or inventing a generated-byte delta. Compiler-cache
receipts are shared-cache aggregate observations, not exclusive hit attribution.

## Rejected alternatives

- Selecting the generic canonical traversal merely because it uses one pass:
  historical negative endpoint evidence remains valid for that different plan.
- Reimplementing Coulomb/exchange mathematics in the runtime: the compiler's
  existing quadratic weight and product rule remain the single scientific owner.
- Promoting mixed precision, triples fusion or Lambda scratch options with this
  default: those are separate experiments with separate evidence and selectors.

## Separate option ablations

[Eight fresh complete endpoints](../../../../benchmarks/results/rhf-other-options-abba-20261012/README.md)
on the earlier frozen `531364e133ae145097b402d66fec5036ed051365` plus its
qualified shell patch use strict FP64, Q32 and one shell derivative pass. They
are not measurements of this default PR's newer frozen source.

Existing triples multi-output scalar fusion reduces mean E+F
`227.393698 -> 219.492646 s` (3.475%) and triples phase
`39.459958 -> 31.642801 s`; actual scalar kernels fall `1320 -> 165` with no
resource fallback. Its planned scalar workspace grows by 431754440 bytes;
whole-endpoint sampled VRAM does not grow because a different phase dominates.
This is not the proposed Lambda Q-batch producer-consumer fusion.

Existing Lambda serial scratch reuse leaves mean E+F effectively unchanged
(`227.397166 -> 227.398488 s`), while reported Lambda capacity falls by
860495616 bytes and sampled endpoint VRAM falls `26500 -> 25678 MiB`.
This is a demonstrated capacity saving, not a demonstrated speedup. Every
cohort retains CCSD 19 iterations, Lambda 21 iterations/22 solver actions,
416 auxiliary batches and 14307 Lambda GEMMs. Energy span is zero and force
span is `5.151435e-14` across all eight endpoints.

Both options remain separately controlled. Requalify source-matched current
defaults and the full intended admission domain before promoting either.

## Revisit when

Requalify changes to admission, signed-weight screening, recurrence scheduling,
projection lifetimes or numerical reduction policy against complete endpoints.
Broaden admission only with the same independent scientific and refusal gates.
