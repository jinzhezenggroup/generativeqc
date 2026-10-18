# Decision: diagnostic MD-J stages and opt-in residual tile prefilter

Status: implemented (experimental schedule; GPU qualification pending)
Date: 2026-10-18
Base: master e4aff7810

The later [default-promotion decision](2026-10-18-md-j-residual-default.md)
supersedes only the opt-in/default policy below. The original experiment,
measurements, limitations and rounding rationale remain historical evidence.

## Problem

The source residual enumerates angular-class shell-pair products before deciding
whether their AO masks or density contributions require contraction. Historical
candidate/admission ratios do not establish the current 96-atom r2SCAN-3c cost,
and uniform reciprocal primitive tiles do not eliminate this distinct source
enumeration. Neither a fixed-density J improvement nor accumulated GPU kernel
time establishes a complete cold endpoint improvement.

## Decision

Keep the default residual traversal. Add a plan-frozen
`GENERATIVEQC_MD_J_RESIDUAL_TILES=1` experiment which traverses 8x8 shell-pair
tiles with the existing persistent cursor and at most 4096 workers. Only the
first warp loads and reduces the two eight-pair envelopes; surviving tasks
reuse block-local pair indices. Diagonal tiles retain exactly one unordered
orientation and ragged tiles do not introduce padded candidates. No resident
queue, CPU oracle or density-dependent host download is introduced. K retains
its original schedule.

A tile can reject candidates for three reasons: all maximum-Schwarz products
are below the threshold, all minimum-Schwarz products are already owned by
uniform MD, or both density directions are negligible by the incumbent error
allowance. Maximum and density products round upward. The minimum product must
round **downward** before testing `>= screening`: upward rounding there could
incorrectly transfer a just-below-threshold residual task to uniform MD, which
would not actually evaluate it. Surviving candidates still use the original
per-candidate and AO screening. The mathematical recurrence is unchanged, but
the atomic accumulation order can change and requires independent GPU gates.

Extend the opt-in work census to separate individual tests from tile-screened
candidates and count visited/rejected tiles. The complete candidate domain is
their sum; admitted shell tasks and contracted primitive products must remain
unchanged. The diagnostic allocation grows from 1200 to 1800 bytes within the
existing optional cap. The uninstrumented path allocates no counters.

Add independent, plan-frozen `GENERATIVEQC_MD_J_PROFILE=1` event tracing for
density bounds, residual, density transform, reciprocal contraction, oriented
contraction and projection. These consecutive on-stream intervals include launch
gaps, output/cursor resets and any counted-kernel overhead; they are not pure
kernel sums or complete cold wall time. Event creation and the final fence are
diagnostic only. No event work is requested by default.

Remove the adjacent duplicate J output clear (the residual launcher owns it)
and density-bound launch (the caller refreshes it before both consumers). No
consumer changes density between residual and uniform MD on the owning stream.
This cleanup does not imply a substantial performance gain.

## Rejected alternatives

- Default promotion without matching-density GPU and complete-endpoint evidence.
- Removing residual tasks based on candidate/admission ratios from another method.
- Replacing the per-orientation AO mask or relaxing the density error allowance.
- Growing a retained quartet queue or using a CPU/reference producer in production.
- Treating zero times infinity as a finite density bound; the exact-zero-density
  rejection remains explicit.

## Evidence and outstanding gates

Host tests execute the production tile mapping and rejection helper against
scalar directed-rounding admission, including ragged/triangular domains, 32-bit
pair-count limits, subnormal/overflow cases and threshold equality. Runtime
doubles cover diagnostic event lifetime/error propagation and both source launch
policies. The resident CUDA tests compare both policies with independent libcint
integrals for restricted/unrestricted and Cartesian/spherical nonsymmetric
densities, and require matching work domains/tasks/roots on replay. These GPU
tests are added but cannot be claimed as run without scheduled hardware.

At implementation time this host's Slurm inventory exposes only `node3` in
`IDLE+DRAIN`; no allowed GPU worker is available. Do not bypass scheduling or
change CUDA visibility to obtain measurements. CUDA compilation is host-only
validation, not a GPU numerical or performance result.

Before promotion, run the independent raw-J and sanitizer gates through finite
Slurm allocations on permitted nodes, measure the actual 96-atom/1088-AO
r2SCAN-3c per-stage/work census, and compare clean same-binary complete cold,
warm and moved-geometry endpoints with all diagnostic fences disabled. Keep
SCF iteration/Fock counts and numerical acceptance separate from timing.

## Scheduled follow-up: n1 / RTX 5090

The initial local-node restriction was resolved by using `ssh n1` and finite
`srun --partition=main --nodelist=node1 --gres=gpu:5090:1` allocations. No work
uses the maintenance node or changes Slurm device visibility. The source snapshot
is based on master `e4aff7810b0462204831eb5168f19b6e217ef520`; the main Release
library, compiled with CUDA 12.9.1 for `120-real` using the existing ccache, has
SHA-256 `5c1108f2d134ecb5ee287c8f8b51a36370b394a8480648d4ba69f323b38315cd`.

Slurm **10076** passes all 29 resident CUDA tests. The domain/task/replay-policy
test additionally passes memcheck, racecheck and synccheck, with zero errors,
hazards or warnings. Source hashes for all five affected numerical files match
this worktree. These sanitizer observations cover that targeted test, not every
large-system execution.

Slurm **10083** evaluates both schedules on the same native-converged
96-atom/1088-AO density (SHA-256
`a0b22059e12adcd39771e5dcfaac0951afb298d8252ef6674889ad0349ed6090`).
Four uncounted prepared replays per schedule give:

| Prepared J-only measurement | Incumbent | Tile prefilter | GPU4PySCF MD J |
| --- | ---: | ---: | ---: |
| Median device-event seconds | 4.210590 | 4.132818 | 0.717517 |
| Median Python-call wall seconds | 4.225147 | 4.141712 | 0.717539 |

The measured packages are GPU4PySCF **1.8.1**, PySCF **2.14.0** and CuPy
**13.6.0**. Native screening is `1e-12`, reference direct-SCF screening `1e-14`;
these are not identical cross-program screening/work contracts. Preparation is
separate, and neither timer includes a complete cold SCF/force endpoint. The
reference receives the same density with explicit AO normalization conversion.
Every native replay and the separately counted replays pass the predeclared
native `atol=rtol=3e-11` comparison and independent MD-J
`atol=3e-9, rtol=3e-11` comparison. Maximum schedule difference is
`2.6290081e-13`; maximum independent absolute J error is `2.4448427e-11`.

The separate diagnostic replay, excluded from those timing samples, records:

| Work per final-density Fock | Incumbent | Tile prefilter |
| --- | ---: | ---: |
| Individual residual candidate checks | 526,842,030 | 495,840,182 |
| Candidates rejected by whole-tile bounds | 0 | 31,001,848 |
| Residual shell-quartet tasks admitted | 43,920,982 | 43,920,982 |
| Residual primitive products contracted | 242,399,046 | 242,399,046 |
| Uniform radial evaluations | 2,225,962,468 | 2,225,962,468 |
| Uniform density directions | 4,013,012,720 | 4,013,012,720 |
| Uniform Hermite summands | 95,323,936,432 | 95,323,936,432 |

All 25 angular-class rows satisfy domain conservation and exact equality of
admitted tasks, primitive products, uniform roots/directions/summands. The tile
schedule visits 8,235,711 tiles and rejects 484,721. Only **8.34%** of incumbent
candidates reach shell-task contraction; these are shell-pair candidates, not
primitive-pair candidates or actual nonzero AO integrals. The unsorted 8x8 tile
bound bypasses just **5.88%** of individual checks, with an observed **1.85%**
prepared device-time reduction. This is not evidence for a large endpoint win
or for default promotion. In particular, aggregate residual time includes
integral contraction as well as enumeration; it does not isolate the cost of
the rejected checks. GPU4PySCF work was not instrumented, so its roughly 5.9x
prepared timing advantage cannot be attributed to a matched work-count ratio.

Initial stage-only runs **10080/10081** retain 24 cold SCF-J intervals each, but
their force endpoints fail closed because the Python-consumer stationary AOT
artifacts were not packaged beside the new library. They are not complete
endpoint observations. Existing campaign artifacts fail the current compiler
contract check and are not reused; the companion force artifacts must be built
from this snapshot before renewing endpoint gates. Keep the failed receipts
rather than relabeling them as successful timing samples.

The companion force artifacts were subsequently rebuilt from the same snapshot,
without changing the main library. The strict-FP64 s/p/d force DSO SHA-256 is
`9fc738752cb6e1270482bff33fefba29e36a810ed3c23db3b109f07203300e01`.
Renewed Slurm **10086/10087** complete cold plus first-warm diagnostic capture,
both with 24 cold Focks and one warm Fock. All four independent endpoint gates
pass: maximum energy error `2.3192115e-11` hartree, maximum force-component
error `5.6950001e-11` hartree/bohr and maximum physical residual RMS
`1.4895401e-12`. The stage-only runs use uncounted kernels; work instrumentation
is enabled only in the separate fixed-density census above.

| Sum of 24 cold MD-J on-stream intervals (seconds) | Incumbent | Tile prefilter |
| --- | ---: | ---: |
| Density bounds | 0.001764 | 0.001751 |
| Residual enumeration **and contraction** | 61.744465 | 59.937093 |
| Density-to-Hermite transform | 0.007081 | 0.007057 |
| Reciprocal uniform contraction | 24.613589 | 24.632459 |
| Oriented uniform contraction | 15.450945 | 15.554312 |
| AO projection | 0.006016 | 0.006023 |
| Complete MD-J interval | 101.823860 | 100.138695 |

Residual work is **60.64%** of the incumbent J interval, establishing its
priority in this actual r2SCAN-3c workload, but not that candidate rejection
alone consumes those 61.74 seconds. Transform/projection and the removed
duplicate small launches are not substantial contributors. The first-warm J
intervals are 4.248939/4.181997 seconds, each for one Fock. These diagnostic
captures run concurrently on separately allocated 5090s, contain event fences
and explicit density export, and deliberately stop before the full warm/moved
cohort. Neither their wall time nor their cross-device delta is a clean endpoint
speedup result. No complete paired cold/warm/moved timing cohort or default
promotion is claimed; the small prepared-J improvement does not justify one.

Source-matched receipts and arrays live on n1 under
`/data/jzzeng/gqc-mdj-residual-20261018/{receipts,results}`; compact local copies
live in the ignored `.artifacts/mdj-remote/` directory. The experiment remains
off by default. No XC density representation or numerical policy is changed.

## Revisit when

Promote only after those gates pass. If unsorted tiles reject little work, use
the new census and intervals to evaluate geometry-bounded indexing or shared
metadata reuse; do not infer a benefit from tile size or avoided candidate count
alone. XC occupied-orbital cancellation fixes remain a separate numerical change.
