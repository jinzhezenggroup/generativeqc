# Experiment: dense canonical-AD pair coefficients

Status: proposed; two-coefficient schedule rejected, canonical graph/native follow-up pending
Date: 2026-10-06

## Problem and full objective

#1894 still requires whole normalized-product graph recognition, independent
fixed-grid gates, RKS/UKS changed-geometry native forces, phase/work/traffic
attribution and complete paired 48/96-atom PBE0 E+F endpoints. The rejected
indexed-domain candidate in PR #1996 does not satisfy those requirements.
Its sampled grid products mostly stay live; serial reverse enumeration loses.

## Candidate dataflow

Keep the ordinary parallel pair domain, ordered normalization and neighbor
gather. Extract the canonical scalar pullback into one shared owner used by
both the ordinary four-word reverse and the two-word candidate. Both consumers
use the same component-expansion owner, with the same multiplication order.
No second scientific formula, logarithm move, broad zero pruning, full xyz
cache or fused reverse/gather is introduced.

The candidate writes only the distance-difference and center-separation
coefficients. Its ordered gather expands the latter using already prepared,
finite center unit directions. Pair primal/log state still requires four
words, so the peak pair reservation and total scratch do not shrink. The
point-normalization boundary protects storage aliasing. The two obsolete
reverse words are never consumed; a host gate poisons them with NaNs.

Require cached center geometry and complete concurrent resource admission;
the ordinary phased/generic path remains the unsupported/resource fallback.
Geometry metadata belongs to the native owner, must be charged as occupied
storage, and must be validated/rebound for every geometry change. No cached
point state or floating-point atomic reduction is introduced.

## Work and traffic model

Every primal/reverse pair and both ordered incident consumers remain. Reverse
pair writes change from four doubles to two; gather pair reads from four
doubles per incident visit to two. Gather additionally reads three center unit
components per visit. Those center reads and extra multiplies can outweigh the
pair-panel saving. These are logical dataflow counts, not hardware-cache
transaction measurements or an endpoint speed claim.

At 96 atoms by 256 points, both plans reserve 39,716,864 scratch bytes. Geometry
metadata and all concurrent native owners are additional admission inputs.

## Qualification ledger

- Initial host cohort: 532 gates pass across coefficient, indexed primitive,
  phased, cooperative, tiled and center-geometry suites. Switch iterations
  1/3/5, cached/direct fallback, geometry rebinding, Decimal/finite differences,
  translation/permutation, rounded zeros and failure publication are covered.
- A final finite-zero overwrite invariant is added before freezing the source;
  the final mathematical source repeats all 532 host gates successfully.
- n1 Slurm job 6128 terminates with exit 2 at CUDA compilation: the new probe
  kernel initially assumes int64 pair indices, while the existing owner uses
  uint2. No GPU numerical or timing evidence is produced. Correct the consumer
  to the existing uint2 contract and freeze a separate retry cohort; do not
  overwrite the failed snapshot or restart the unchanged source.
- Device retry job 6129: exit zero; 21 gates pass, 28 explicitly skipped
  iteration-1/5 parameterizations. The actual CUDA library is iteration 3.
  Memcheck, initcheck, racecheck and synccheck have zero errors; racecheck has
  zero hazards/warnings. Candidate-first initialization is preserved.
- Whole-composition recognition, native force ownership, phase profiling and
  complete endpoints: pending. No production promotion is authorized by these
  isolated tests, even if the candidate wins.

## Completed isolated result: no promotion

Clean unprofiled ABBAABBA medians over 64 sampled grid tiles:

| Atoms | Tile points | Phased ms | Two-coefficient ms | Maximum gradient difference |
| --- | ---: | ---: | ---: | ---: |
| 48 | 256 | 10.314399719 | 10.553583622 | 0 |
| 48 | 1024 | 20.655695915 | 21.143087387 | 0 |
| 96 | 256 | 25.285776138 | 25.917823792 | 0 |
| 96 | 1024 | 68.089103699 | 68.810241699 | 0 |

All four point estimates lose (roughly 1–2.5%); this cohort demonstrates no
material speed advantage. It does not establish a statistically resolved
endpoint regression. Additional center reads and multiplies are not free,
and pair-panel byte savings alone do not predict runtime. Reject this storage
schedule for promotion; do not rerun its unchanged dataflow as a new candidate.
These are not SCF/native-force endpoints, and sanitizer timings are intrusive.

The failed snapshot remains at `/data/jzzeng/qc-becke-coefficients-1894-v2`.
The successful retry remains at
`/data/jzzeng/qc-becke-coefficients-1894-v2-r1`; original local receipts are
`.artifacts/issue1894/coefficients-v2/retry-1/job-6129/`.
The publication is
`benchmarks/results/becke-pair-coefficients-20261006/publication.json`.

Subsequent local Python changes introduce a complete normalized-product
primal/JVP and its recognizer. All 20 scalar and 20 mixed AD identities remain
equal to this measured base. Before final harness-only clang formatting,
re-emitting the default coefficient probe produces the exact same CUDA bytes
as the frozen retry. The newly whole-graph-matched
operation identity itself is not the identity measured by job 6129; its final
native integration/device/endpoint gates remain pending. See the separate
whole-partition IR note rather than treating this isolated probe as promotion.
The frozen sample header omits the coefficient-mode IR identity (null). The
publication reads the actual identity from the immutable, SHA-256-verified
emitted CUDA annotation, explicitly records that provenance, and preserves
the raw samples unchanged. The reporter is fixed for subsequent cohorts.
Final clang-format changes only harness whitespace; the published patch keeps
the exact measured bytes. No post-format GPU rebuild or new timing is claimed.
The final broader host cohort passes 619 gates after the complete-graph API and
reporter changes; the compiler structure inventory contains 467 modules with
zero dependency errors.

## References

#1894; #1830; #1950; PR #1996;
`2026-10-05-becke-partition-primitive.md` preserves the rejected index route.
