# Experiment: indexed normalized-product Becke derivative dependencies

Status: proposed; indexed-domain candidate rejected, dense coefficient follow-up pending
Date: 2026-10-05

## Scope and original objective

Keep #1894's complete scope: a compiler/AD-owned generated Becke partition
derivative primitive, independent fixed-grid and finite-difference gates,
RKS/UKS changed-geometry force semantics, phase/work/traffic attribution and
paired complete 48/96-atom PBE0 E+F comparisons against phased execution.
No production promotion or issue completion is inferred from host tests or an
isolated GPU probe. The measured source base for this candidate is
`c5ab37ec9e7b3d38d2e06729319f9eef66510e5c`.

## Candidate dataflow

Recognize actual reachable canonical ratio, logarithm and Becke-switch AD roots,
not a supplied stale identity. Keep the existing scalar AD and phased helpers as
the mathematical owner; the generated candidate supplies an explicit point x
atom dependency domain. Every primal pair and ordered atom log sum still runs.

An atom product with two exact zero factors has zero first derivatives. One
zero does not necessarily annihilate the derivative: a rounded-zero Becke
factor can still have nonzero slope. Build a sorted per-point index panel of all
atoms with zero count <= 1. Do not use magnitude cutoffs, omit single zeros,
stop log traversal, or change saturation and denominator/log-scale semantics.

Normalize the indexed products in their original order through the SAME
`maximum_log_product` and `normalized_product_adjoint` owner. If the selected
owner is not live, append its exact-zero numerator solely for this AD objective,
not to the reverse dependency list. Broadcast the existing denominator adjoint
instead of materializing redundant bars for annihilated products. The shared
pair adjoint receives that bar view; no second force formula is introduced.

Reverse work enumerates each pair with at least one live endpoint once. A
two-live pair belongs to its higher live endpoint; a sole-live pair belongs to
that endpoint even if it is the lower atom. An inactive atom still receives
center-distance response from live neighbors. Its gather visits the sorted
live list; a live atom retains the full ordered incident gather. The compiler's
same gather body, point-motion owner and deterministic publication are reused.
There are no floating-point atomics or topology-independent point-state caches.

Both index/count panels and the common-bar panel are charged against concurrent
owner memory. A 96-atom, 256-point candidate uses 39,917,568 scratch bytes versus
39,716,864 for the phased plan. Index membership is regenerated after each
point normalization and geometry bind. Ordinary phases and generic AD remain
the unsupported-domain/insufficient-budget fallbacks.

## Why this is a different experiment

This does not repeat #1950's pair-domain logarithm move, the losing per-atom
log-zero truncation, a full xyz derivative cache, or a fused reverse/gather
launch. It changes the normalized-product representation, broadcasts a shared
AD root and compacts reverse consumers; it does not merely branch away a few
scalar logarithms. This distinction is not evidence of speed. Random points
without exact product zeros retain the complete reverse domain and can lose
from additional index work; the real grid distribution and endpoint must decide.

## Evidence and acceptance ledger

- Initial dependency-only candidate: 157 host gates pass. A premature assumption
  that random 96-atom points always reduce reverse work fails for iteration 1;
  the assumption is removed and an explicit saturated-domain work gate added.
- Indexed-normalization/common-bar candidate: 450 host gates pass across
  primitive/phased/cooperative/tiled/center-geometry suites. Both cached and
  uncached geometry, iterations 1/3/5, independent Decimal finite differences,
  translation/permutation, signed seeds, rounded zeros, saturation and failure
  publication are covered. This is CPU evidence, not a GPU performance claim.
- Compiler structure check reports no dependency errors. No public runtime,
  oracle or real GPU is loaded during primitive source generation.
- The optional isolated probe compares phased versus primitive in one binary
  and allocation using ABBAABBA. It primes the candidate before the dense arm,
  so initcheck cannot borrow baseline initialization. Live membership is read
  back in a separate post-timing pass; exact reverse/gather visit counts follow
  the emitted domain. Logical traffic models are not hardware-cache counters.
- Device iteration-3 numerical/failure gates: 21 pass, 28 explicitly skipped
  host iteration-1/5 parameterizations. All four sanitizers pass; details below.
- RKS/UKS native-owner integration and changed-geometry qualification: pending.
- Pair primal/switch-log, normalization, reverse, gather and point-motion
  launch/device/traffic attribution: pending.
- Full 48/96-atom cold/five-warm/moved/five-moved-warm PBE0 endpoints with
  independent references and actual solver work: pending.
- Whole-composition graph matching and unsupported-domain audit must be checked
  before claiming the issue's full canonical-graph acceptance. Matching local
  scalar AD roots alone is not a substitute for that audit.

## Completed indexed-domain device result: reject this schedule

N1 Slurm job 6116 completes with exit zero. Its actual primitive CUDA ABI passes
21 device gates with 28 explicitly skipped iteration-1/5 host parameterizations;
the compiled device specialization is iteration 3. Memcheck, initcheck,
racecheck and synccheck report zero errors/hazards. Candidate-first priming
exercises its own initialization. All clean isolated comparisons have zero
maximum gradient difference from the same-binary phased route.

| Atoms | Tile points | Phased median (ms) | Indexed primitive median (ms) | Indexed reverse visits | Dense reverse visits |
| --- | ---: | ---: | ---: | ---: | ---: |
| 48 | 256 | 10.219712257 | 17.529743195 | 18,369,651 | 18,481,152 |
| 48 | 1024 | 20.415664673 | 26.832768440 | 73,451,048 | 73,924,608 |
| 96 | 256 | 25.073840141 | 38.881439209 | 73,191,272 | 74,711,040 |
| 96 | 1024 | 67.590160370 | 87.496929169 | 292,801,589 | 298,844,160 |

These are unprofiled isolated ABBAABBA CUDA-event samples over 64 grid tiles,
not complete endpoints. Sanitizer timings are not substituted for these rows.
GPU membership counters show that most atom products remain live: exact
multi-zero liveness does not eliminate enough incident work to pay for the
serial point/atom reverse domain. Retain this negative result, not a production
route or a claimed geometry-response win. The source snapshot remains under
`/data/jzzeng/qc-becke-primitive-1894`; local original receipts are in
`.artifacts/issue1894/primitive/v1/job-6116/`.

The compact, checksum-bound publication is
`benchmarks/results/becke-partition-indexed-20261006/publication.json`.
It retains unchanged clean samples, the measured dirty-source reconstruction
patch, Slurm/test/sanitizer receipts and a finite Slurm reproduction recipe.
It records a performance rejection, not endpoint or production acceptance.
The publication audit rechecks all seven executable source files against the
frozen build inventory and repeats the 450 host gates without changing that
measured source. Ccache is invoked for compilation; the retained GPU build
statistics show one additional uncacheable invocation, not a cache hit.
The PR subsequently clang-formats the probe harness only; the CUDA measurement
still belongs to the frozen pre-format source. Restore the published patch and
verify its inventory before reproducing that exact cohort. No post-format GPU
rebuild or performance measurement is claimed.

The next candidate keeps the dense parallel pair domain and the existing AD
scalar pullback. Instead of storing all three xyz pair pullbacks, retain only
the distance-difference and center-separation coefficients. Ordered atom gather
expands the latter using already validated center geometry. Primal/log state
and the peak four-word pair reservation are unchanged; reverse stores and
incident reads can shrink without another xyz cache, a new formula, early
log-zero truncation or fused reverse/gather. This needs independent gates and
actual measurement; it is not declared profitable from a byte model.

## References

#1894; #1830; #1950; the pair-log and rejected low-payoff-probe Agent Notes.
