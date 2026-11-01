# Native stationary exact VV10 force reuse

This source-matched WB97M-V DF96 campaign compares the corrected native force
owner against its frozen predecessor on RTX 5090 and RTX PRO 6000 Blackwell.
It removes repeated VV10 pair sweeps for bitwise-identical ingredients within a
fixed geometry/grid/model lifetime. It borrows existing dead scratch and the unused force scalar slot, adding no
device allocation and preserving both original capacity tiers. Minimum-capacity
owners can also reuse their exact seeds.

The recorded source is
`6527e5625b096f10a8c431ce2829b67d44a98abb656889f486f2b13786dc419c`, and the native
library SHA256 is
`327a417cb66c115b7f5d35ca9fde6a89ec217da632e2d0cb0c589bead84b614f`.
The predecessor is source
`6be420fb4ee14175922194d61b41e3a20f13665968bc012b474b21cada96f4cc`, library
`bb2c3e70046ebc753dbb9722b53df1eb9094bb6d3bb8882ff3ae14887f471e10`.
Original dirty-source receipts and base commits are preserved. Neither image is
relabeled with a later publication commit.

## Complete endpoints

Each device runs one conditioning parent (cold plus one warm), then parent,
candidate, candidate, parent fresh owners, each with cold plus three frozen
engine-local warm calls. Cold includes construction, preparation, SCF, analytic
forces and host output. Strict FP64, stock semilocal/VV10 grids, spherical
Def2-SVP, pinned auxiliary basis, SCF/force tiles 256/1024, eight scheduled CPU
slots and single-threaded math are unchanged. The complete cold and warm
populations, variability, SCF iterations and Fock-build counts are retained.
Positive percentages below mean slower candidate time.

| GPU | Complete endpoint | Parent (s) | Candidate (s) | Change |
| --- | --- | ---: | ---: | ---: |
| RTX 5090 | Cold | 235.336 | 235.276 | -0.03% |
| RTX 5090 | First warm | 58.383 | 30.074 | -48.49% |
| RTX 5090 | All warm calls | 58.383 | 30.079 | -48.48% |
| RTX 5090 | Cold + three warm | 410.487 | 325.519 | -20.70% |
| RTX PRO 6000 | Cold | 208.400 | 208.316 | -0.04% |
| RTX PRO 6000 | First warm | 42.946 | 14.327 | -66.64% |
| RTX PRO 6000 | All warm calls | 42.947 | 14.328 | -66.64% |
| RTX PRO 6000 | Cold + three warm | 337.243 | 251.299 | -25.48% |

Cold differences are within the observed spread; this change establishes a
retained-owner warm improvement on these two devices.

This is a predecessor comparison, not a fresh seven-method GPU4PySCF ranking.
The independent README references still qualify every conditioning and measured
energy/force output. Diagnostic trace timings are excluded from clean ABBA
medians. The 5090 trace records one cold pair sweep and zero sweeps in each of
two warm calls, with an actual exact-key comparison and hit on both warm calls.

## Correctness, capacity and lifetime

Both devices pass seven independent GPU4PySCF-backed controls: admitted replay,
explicit disable, default-small fallback, minimum capacity, schedule/selector
changes, failed-call recovery and displaced geometry. Energy and every force
component must satisfy 2e-8 Eh and 2e-7 Eh/bohr gates. A two-item PRO 6000 batch
adds fourteen independently checked E/F calls covering distinct geometry,
retained ownership, a transactional live-capacity-shrink refusal and a newly
admitted minimum-capacity item. Seven native-only 5090 processes each pass
Compute Sanitizer with zero errors and zero bytes leaked; their independent
oracles are generated in a separate uninstrumented process.

The 108-atom 32-GiB control retains six complete phases for each source: cold,
retained, deliberate source eviction, second replay, moved geometry and moved
replay. Paired all-output and semantic-work checks apply to all twelve calls.
This larger control is a bounded lifecycle comparison, not independent
large-geometry production ranking qualification. The single larger cold call
is 0.78% slower; all six phases together improve from
907.138 s to 767.589 s.

| 108-atom phase | Parent (s) | Candidate (s) |
| --- | ---: | ---: |
| cold | 318.910 | 321.412 |
| retained | 71.329 | 35.832 |
| evicted-fallback | 73.122 | 37.665 |
| second | 71.327 | 35.860 |
| moved | 300.989 | 300.856 |
| moved-warm | 71.461 | 35.964 |

The independently prepared predecessor and candidate return identical public
nine-field capacity receipts, including minimum and cached bounds. This protects
fleet admission from shifting tiers at a tight byte ceiling. The diagnostic
predicate uses the scalar slot after six force-seed rows, while scalar pair
energy reduces into dead energy-term scratch before key storage.

Earlier allocating prototypes and their failures stay under their original
identities. A genuine optional-metadata `cudaMalloc` OOM initially left a stale
CUDA error and failed the endpoint. Clearing that error fixed the immediate
fallback, but adding even 32 bytes could still demote a tightly bounded fleet to
its minimum tier. The allocation-free design removes both problems. Failed shim
setup, missing deployed auxiliary-fixture failures and the original displaced-
test warm-seed error are also retained; none is relabeled as a final-source test.

## Replay and limits

```bash
PYTHONPATH=python:. python benchmarks/results/stationary-vv10-reuse-20261101/report.py
PYTHONPATH=python:. python -O benchmarks/results/stationary-vv10-reuse-20261101/report.py
```

The reader restores exact FP64 words, verifies each reconstructed text hash,
binds source/binary/allocation identities, and rechecks independent references,
paired all-repeat errors, complete endpoint/work counts, batch/fallback results
and sanitizer receipts. Large transient trace logs stay in ignored local
artifacts with hashes and sizes in the bundle; the complete relevant force-work
records are retained. No external archive or Release is required. The tracked
benchmark evidence total stays within its existing 80-MiB ceiling.

A100 TCP connection attempts timed out. This source has no new A100 execution
measurements; old jobs 224223/224225/224229 remain unretrieved and were not
resubmitted. No universal GPU speed guarantee or all-functional completion is
claimed. Prior source-only and strict-packet experiments were rejected and are
recorded separately in the Agent Notes.
