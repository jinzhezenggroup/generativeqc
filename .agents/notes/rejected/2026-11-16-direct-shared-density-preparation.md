# Decision: reject a shared high-order density-preparation device call

Status: rejected
Date: 2026-11-16

## Hypothesis

The high-order Direct force worker repeats the same AO traversal and density
weights across strict, relative-response, mixed-precision and resident variants.
Moving that traversal into one noinline helper, specialized only by spin,
radial owner, output mode and component capacity, could reduce compilation work.
The candidate deliberately leaves low orders and pair-separable preparation
unchanged and retains every method, architecture and fallback.

## Experiment and decision

Do **not** retain this candidate. Its modest compilation reduction does not
justify the observed native-task timing risk. The production header is restored
byte-for-byte to the published version; the isolated before/after source closures
and evidence stay in ignored local scratch storage. This rejection is specific
to outlining density preparation, not to the earlier shared high-order
arithmetic callables, whose much larger memory result remains separate evidence.

Both complete `sm_80` fallback objects compile with NVCC 12.9.86, verified
sccache 0.16.0, `-O3 -DNDEBUG -std=c++20 -rdc=true`, PIC and the production
`--disable-optimizer-constants` option. Both have zero cache hits and four
CUDA-stage misses. The comparison is against the already shared-arithmetic
parent, not the original unshared Direct implementation.

| Complete fallback owner | Published parent | Candidate |
| --- | ---: | ---: |
| Embedded PTX bytes, same extraction method | 141,121,029 | 139,834,642 |
| Object bytes | 66,620,976 | 65,228,784 |
| Reaped compiler-worker maximum RSS, KiB | 5,579,972 | 5,471,104 |

The embedded PTX shrinks about 0.91%, the object 2.09%, and the single observed
worker maximum 1.95%. These are small one-sample effects, not an established
compiler-memory or timing win. The retained pre-fatbin NVVM PTX input is larger
(149,129,566 / 147,753,625 bytes); do not compare one representation with the
other. The candidate's owned-server lifetime is 21:04.05, but host tests and
other compiler probes overlap the observations. No cold-build speedup is claimed.

## Scientific and scheduled-device observations

The actual native host consumers and independent precision/range references
pass **922 tests**, with **164 existing skips** for invalid repeated-shell
parameter combinations. No scientific test or invalid-input guard is removed.

Authorized `n1` runs Slurm job **17488**, `main` / `node1`, with one
`gpu:5090:1`, an explicit ten-minute limit and unchanged assigned
`CUDA_VISIBLE_DEVICES=1`. The original and candidate precompiled libraries pass
**120 complete native-task comparisons** against independently normalized
Libcint derivatives and an independent eightfold density-orbit assembly. Cases
cover PPPP/DPPP, both spins, combined/separate sources, full/LR operators, signed
exchange, ordinary/sparse/zero weights, repeated ket/pair domains and strict,
relative and full-range mixed arithmetic requests. LR mixed requests retain
the existing strict route. Diagnostic counters also match between variants.

On 48 dense cases, five alternating before/after samples of 128 tasks show a
median candidate/baseline task-timing ratio of **1.0920**, with ratios ranging
from **1.0047 to 1.1210**. Median ratios by requested mode are 1.0765 strict,
1.0850 relative and 1.0976 mixed. All sampled cases are slower in this probe.
This is a rejection signal, not a claim of a measured full-application slowdown.

The probe ports only existing test transport to CUDA and invokes the actual
production task, including density preparation, primitive geometry, weighted
arithmetic and additive atom scatter. Its one-lane transport also constructs
fixture metadata and diagnostics; it is **not** a production Direct/HF endpoint,
an occupancy qualification, or a full force timing. A small compilation win
plus this warning cannot establish complete-endpoint runtime nonregression.

The serialized report verifies both library SHA-256 identities and records the
allocation, visibility, oracle comparisons, input-domain sizes, actual diagnostic
counters and raw timing samples. Available primitive products and admitted orbit
candidates are domain counts, not measurements of executed work after zero-source
short circuits.
Its SHA-256 is
`b6ee0fbf4fb322961bf3f52d2365412dd30e961070ed4b36f4bf201f4fc581bf`.

## More consequential remaining duplication

A one-off ownership partition of the parent's 149,129,566-byte NVVM PTX input
finds the following function-body sizes; declarations and other metadata are
not included in these subtotals:

| Owning function family | Functions | Body bytes |
| --- | ---: | ---: |
| Bounded scheduler kernels | 108 | 30,187,870 |
| Other native consumers | 198 | 86,611,460 |
| High/low-order density and primitive task bodies | 278 | 24,232,489 |
| Retained weighted arithmetic callables | 49 | 6,084,153 |

These categories describe symbol ownership, not arithmetic semantics: native
consumer bodies still contain substantial inlined scientific arithmetic. Two
Fock dispatchers alone are about 4.8 MB each. Runtime and AOT range dispatchers
for maximum angular orders 8 and 12 repeatedly inline the same admitted
per-order bodies and reach roughly 3--3.8 MB per function.

The next isolated experiment shares high-order per-angular range bodies across
those **distinct retained 8/12 admission domains**. AOT range/omega constants
must remain template-visible, lower-order inlining must stay unchanged, and
no method, bounded capacity or radial fallback may disappear. This remains a
proposal until complete compilation, device-link, independent numerical and
appropriate endpoint/resource gates pass. A smaller PTX file alone is not
permission to adopt it.

## Revisit when

Reconsider density preparation only if a materially different interface avoids
the extra escaped metadata/call cost and complete production endpoint evidence
demonstrates nonregression with a substantial compiler-work/resource reduction.
Do not repeat this same extraction merely because its standalone code is cleaner.
