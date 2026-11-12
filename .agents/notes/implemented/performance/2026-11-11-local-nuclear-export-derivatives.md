# Decision: visit only contributing nuclear pairs during CUDA derivative export

Status: implemented
Date: 2026-11-11

## Problem

An exclusive RTX 5090/node4 investigation of the default DF RHF 96-atom README
endpoint found approximately 1.20 seconds in nuclear derivative preparation.
The scalar CUDA export launched one calculation per coordinate. Each calculation
evaluated the generated Coulomb gradient for every triangular atom pair before
checking whether either atom was the differentiated atom. Across all coordinates,
this performed cubic rather than quadratic pair work. The subsequent generated
electronic force response was not responsible for this preparation cost.

The corresponding fitted DFT endpoint already avoids this export: its stationary
owner adds nuclear gradients through the native pairwise assembly in
`molecule/nuclear_gradient.hpp`. Its resident three-center preparation and occupied
exchange otherwise use the same expensive components as RHF; copying those DFT
paths would not remove the nuclear-export cost.

## Decision

For a nuclear derivative coordinate, visit only its atom's partners. Keep the
existing generated scalar derivative, triangular pair orientation, ascending
contribution order, batch atom offsets, zero result for out-of-range coordinates,
and CUDA execution. No host electronic work, oracle dependency, additional memory,
automatic policy, or benchmark-specific override is introduced.

This optimizes the scalar derivative exporter, not the separate direct-force
kernel. Energy-only exports and their derivative-storage omission remain unchanged.

## Invariants and rejected alternatives

- Do not relax SCF convergence, final physical-state checks, fitting thresholds,
  precision, screening, initial guesses, or reference defaults to hide a cold gap.
- Do not remove nuclear derivatives or add them twice in the fused force owner.
- Retain the original summation orientation/order rather than changing floating
  point reduction semantics or introducing atomics.
- DFT's small native host nuclear assembly is a valid comparison, but this change
  does not need a new host fallback or movement of an existing CUDA responsibility.

## Evidence

The permanent native regression compares singleton and batched nuclear-only exports
against an independently written Coulomb derivative for three non-collinear,
unequal charges and a displaced geometry. It covers all axes, middle-atom signs,
nonzero batch atom offsets, and continued omission of electronic derivative arrays.
The existing native DF oracle/finite-difference tests and independent tight-model
default-SCF tests additionally protect complete forces and displaced endpoints.

Intrusive component tracing and clean endpoint timing are separate populations.
Raw acquisition scripts, exact source/ELF identities, phase traces, all acquisition
attempts, telemetry, and the fixed six-permutation qualification plan are retained
under `.artifacts/df-hf-cold-profile-20261111/` and the matching node4
`/data/jzzeng/df-hf-cold-profile-20261111/` directory. Timed calls use the unchanged
`benchmarks.readme_defaults` public-default constructor and complete host-returned
energy/analytic-force boundary. The two diagnostic sizes do not replace the README
matrix, and same-fitted-model agreement does not waive its disclosed dense-reference
fitting failures.

Exclusive node4 Slurm job 1567 completes all six permutations of baseline,
candidate, and unchanged GPU4PySCF order for HF12 and HF96: 36 fresh processes,
144 complete endpoints. Four GPU-local physical CPUs and NUMA node 1 are bound;
the 5090 retains its stock 575 W limit. Compiler commands are byte-identical to
the frozen baseline. HF96 medians in seconds are:

| Endpoint | Baseline | Candidate | GPU4PySCF |
| --- | ---: | ---: | ---: |
| Cold, six fresh processes | 17.726576 | 16.564723 | 16.544570 |
| Warm, 18 frozen-seed calls | 3.779896 | 3.777622 | 6.909785 |

Cold saves 1.161853 seconds (6.55%); it remains 0.020153 seconds slower than the
reference, not an all-win result. Native cold target work remains 11 iterations /
12 Focks, plus seven preliminary atomic builds. HF12 has no material regression.
The separately traced nuclear export drops from 1.202759 to 0.050192 seconds;
these diagnostic component intervals do not replace the clean endpoint timings.

Nine independent tight-model/default-SCF tests and the native suite's 48 weighted
DF finite-difference gates pass. Across all same-phase repeat pairs, old/new energy
is identical and the maximum force-component difference is `4.27e-13 Eh/Bohr`.
Every measured endpoint remains in the original common and independent fitted
oracle gates. An extra diagnostic requiring `1e-10` force identity even between
cold and warm calls fails for the baseline against itself: allowed warm refinement
changes forces by up to `6.67e-8` for HF12 and `4.24e-9` for HF96. Retain that
failed diagnostic; it is neither a new production gate nor evidence of a candidate
regression. Same-phase identity and the unchanged all-phase scientific gates are
separate checks.

## Revisit when

If nuclear-export launch/transfer overhead becomes material after removing cubic
work, qualify a full-vector bounded device export separately. Preserve the same
independent gates and complete-endpoint timing rather than extrapolating from
an isolated scalar-kernel speedup.
