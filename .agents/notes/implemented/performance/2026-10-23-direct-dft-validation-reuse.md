# Decision: reuse authenticated stationary array algebra

Status: implemented
Date: 2026-10-23

## Problem and attribution

Direct DFT does not share HF's general warm advantage over GPU4PySCF. On
master `4a783a27e104e9392a7227b7efca79abcfba60ae`, water-96/def2-SVP spherical
Standard E+F spends substantial time in preparation and the stationary force
consumer, even when cold SCF iteration counts match the reference.

Separate, instrumented PBE0 host-wall ledgers locate these costs (seconds):

| Exclusive native ownership | Cold | First warm |
| --- | ---: | ---: |
| Native batch preparation | 4.081 | 0 |
| Native SCF execute | 16.373 | 1.702 |
| Snapshot construction | 0.296 | 0.247 |
| Snapshot decode | 0.732 | 0.245 |
| Stationary array algebra | 0.544 | 0.516 |
| Native integral derivatives | 2.998 | 2.914 |
| Remaining public force work | 5.463 | 2.795 |

These are exclusive ownership scopes, not pure CPU times or CUDA kernel
durations. GPU4PySCF's inclusive gradient is 6.158/4.345 s in its diagnostic
population. Native warm SCF is cheaper, but its complete force path costs about
6.7 s. Cold also pays preparation, more SCF work and first-force setup. PBE0
cold cycles match at 9; native warm builds one Fock, GPU4PySCF two effective
potentials. Thus iteration counts alone do not explain the gap.

Native and reference XC grids contain 1,078,336 and 1,078,528 points. The
native force selects 2,107 tiles of 512 points, scheduled in 101 bounded chunks.
Active-AO selection at the unchanged 1e-16 cutoff reduces point-AO-square work
from 636,028,452,864 to 57,196,409,408. Cold sampled discovery visits
8,281,620,480 AO jet values; warm has 2,107 map hits and no discoveries. The
3.004 s discovery scope overlaps scheduling effects and is not an additive
endpoint saving. Existing counters report 4,666 D2H calls / 4,664 synchronizations,
a 75,481,384-byte host snapshot and a 470,256,008-byte additional device peak
bound within the 512 MiB force budget. These identify remaining data-movement
and scheduling work; they do not alone establish GPU idle time.

## Decision and invariants

The same immutable stationary state undergoes complete matrix algebra in
`from_native`, at force admission and before publication. Keep one bounded
successful-algebra certificate on that detached state. Reuse requires the same
native source, validation policy, current allowance, array objects and their
shape/stride/dtype interpretations. Only exact NumPy arrays backed ultimately
by immutable bytes qualify; writable storage, read-only aliases of writable
storage, subclasses and other executors keep the full-validation fallback.

Every consumption still checks identity, success/convergence/physical flags,
the physical residual and native allowance bounds, then authenticates the live
native epoch and exact snapshot contents. A loose force certificate cannot
authorize strict response validation. No certificate is published until both
algebra and native authentication succeed. Strong array references prevent
object-identity recycling; the state owns the certificate, so there is no
snapshot-to-state reference cycle or extended native lease. Array replacement
and reinterpretation cannot reuse the certificate. Direct diagnostic calls to
`_validate_arrays` still execute the full checks.

Within one algebra check, reuse S*C and F*C for orthonormality and the eigen
residual/scale. Scientific grids, screening, convergence, initial guesses,
precision and force/grid-response semantics are unchanged. Native source and
compiler/AOT identities are unchanged.

Candidate diagnostic PBE0 algebra takes 0.156 s cold and 0.147/0.148 s warm,
versus 0.544 and 0.516/0.514 s. Native snapshot proof still runs every time.
This measured reduction, rather than a changed SCF trajectory, is the causal
optimization. Diagnostic endpoint times are excluded from the clean comparison.

## Clean endpoint evidence

Implementation commit: `4c095715ff5007047259b263c5229f6d16262212`.
One RTX 5090 allocation (Slurm 12979, node1, main, 45-minute limit) interleaves
master, candidate and stock GPU4PySCF in three rotated orders. Each process
measures constructor + preparation + first SCF + analytic force + host export,
then three same-owner E+F calls from its own frozen cold density. Process starts
are fresh; compiled/artifact caches are reused. OMP/OpenBLAS/MKL each use one
thread. GPU4PySCF 1.8.1, PySCF 2.14.0, CuPy 13.6.0, NumPy 2.4.6, CUDA 12.9.
These are qualified default-grid samples, not a search for each engine's
fastest accuracy-qualified configuration or a whole-DFT ranking.

| Method | Master cold / warm (s) | Candidate cold / warm (s) | GPU4PySCF cold / warm (s) |
| --- | ---: | ---: | ---: |
| PBE | 23.773 / 7.067 | 23.275 / 6.506 | 16.262 / 5.628 |
| PBE0 | 30.100 / 8.472 | 29.691 / 8.070 | 20.072 / 6.910 |
| r2SCAN | 24.196 / 7.630 | 23.852 / 7.397 | 19.110 / 7.562 |

Cold values are medians of three complete calls; warm values are medians of
all nine calls. Native master/candidate warm ranges are 6.926–7.239/6.438–6.822
(PBE), 8.278–8.616/7.968–8.128 (PBE0), and 7.433–7.773/7.080–7.461 (r2SCAN).
The warm reductions are 7.9%, 4.7% and 3.0%; cold reductions are 2.1%, 1.4%
and 1.4%. PBE/PBE0 remain slower than GPU4PySCF. r2SCAN warm is close, with a
small candidate advantage in this population.

Additional water-3 and water-12 baseline/candidate holdouts bring qualification
to 39 clean populations / 156 complete calls. Every sample passes the original
common gates: 1e-5 Eh/atom, 1e-4 Eh/Bohr maximum force error and 3e-5 Eh/Bohr
RMS force error. The observed maxima are 2.075e-6, 3.452e-5 and 1.563e-5,
respectively. References retain their original receipts, dense-grid stability
and CPU/GPU checks from `readme-final-20261022`, using its explicit reuse
manifest. Paired settings, iterations and Fock counts match exactly; maximum
paired energy/force differences are 1.819e-12 Eh / 1.113e-9 Eh/Bohr.

Compact raw timing samples, counts, source/receipt hashes, qualification errors,
diagnostic work and reproduction commands are in
`benchmarks/results/direct-dft-validation-20261023/summary.json.xz`. Detailed
receipts, profiles, scripts, cache stats and logs remain at
`n1:/data/jzzeng/direct-dft-cost-20261023/` and the worktree's ignored `.artifacts/`.
Original diagnostic receipts use the abbreviated base SHA; qualification
expands only a copy to the verified full SHA, preserving original raw files.

The reused core has SHA-256
`a7fe1d09bdd07fce2265d6b03a213db2867bd6e370d00b5ccbf4cfa1781e62fb`.
All 2,461 tracked build/native/compiler/runtime inputs of its source checkout
matched master before reuse. This is a verified equivalent existing build,
not a claim of a fresh master compilation. The candidate changes only the
stationary Python validation module among those inputs.

## Validation and rejected alternatives

- 222 host tests pass; four GPU cases are explicitly skipped there. Coverage
  includes strict/budget policies, mutated gates/allowances/arrays, writable
  aliases, reinterpretation, lifetime and real native relabel/replay rejection.
- 15 tests pass in the GPU allocation (10 real-GPU cases and five CPU snapshot
  cases), including PBE0 RKS/UKS independent moving-grid
  oracles, r2SCAN RKS/UKS analytic checks, reconverged directional finite
  differences and late-owner replay/geometry invalidation. The first run had
  14 passes and a missing PBE0-UKS packaged artifact. Building just that target
  and its authenticated manifest with verified sccache 0.16.0 resolved the
  packaging prerequisite; the core hash stayed identical. The complete device
  selection was rerun successfully. Cache statistics and actual launcher
  commands are retained; cache counters may also include other users of the
  shared server. All pre-commit checks pass.
- An exact native AO-bitmask ablation preserves the cutoff and passes numerical
  gates, but its diagnostic PBE0/PBE/r2SCAN warm values are 8.551/6.962/7.477 s
  versus 8.429/6.942/7.538 s on the baseline. There is no stable broad win, so
  the sampled-jets default remains. This agrees with the limitations in the
  earlier exact-AO-bitmask notes.

## Remaining work

This removes repeated validation, not the larger direct DFT gap. Further work
should measure native preparation ownership, repeated snapshot/grid transport,
and the force grid's many bounded submissions and synchronizations. Broader
native compiled force admission for d shells requires its own scientific and
resource qualification; the existing s/p admission is not widened here.
