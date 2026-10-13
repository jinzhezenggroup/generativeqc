# Decision: canonicalize successful ordinary CUDA eigenpair order

Status: implemented
Date: 2026-10-12

## Problem

The SAP/Modified Hückel screening found intermittent native admission refusals
despite independent electron-count and metric-occupation validation. The original
combined eigenframe error could not distinguish solver status, malformed or
nonfinite storage, and eigenvalue ordering. It did not establish an algorithmic
failure of either initial guess.

Four regenerated seeds were saved before admission: SAP and Modified Hückel at
384/768 AOs. Each exact saved input was refused on all three fresh owners by the
original library. A diagnostic-only rebuild isolated strict-order inversions in
the successful ordinary Xsyevd results, inside the large occupation cluster near
two. For example, the 48-atom Modified Hückel frame returned adjacent values
`1.9999999999999936` then `1.9999999999999933`; the 96-atom frame returned
`1.9999999999999896` then `1.9999999999999891`. These are presentation inversions
of one or two double-precision ULPs, not independently observed nonphysical seeds.

A separate analytic closed-shell DCT projector reproduced refusal in the ordinary
DF adapter's device validation. Pairwise-degenerate eigenframe fixtures had not
covered the much larger zero/two eigenspaces of a determinant density.

## Decision

Add a backend-neutral `order_eigenpairs` presentation operation. Stable-sort the
finite eigenvalues' indices, then apply permutation cycles to the values and
entire row-major coefficient columns together. Preserve every numerical value,
including near-degenerate differences, exactly; do not clamp or recompute them.
The already-ordered fast path does not allocate. Reordering needs O(n) index
storage and at most n-1 full-column swaps, not another n-by-n frame.

Use it only on detached successful results in the idle CUDA KS seed operation
and the ordinary CUDA DF eigensystem adapter, before their existing physical
validation. Nonzero provider info still rejects without reordering. Captured SCF
buffers, solver family selection, generalized transforms, warm/final ownership,
generation tokens and device-state publication remain unchanged.

Split the common host eigenframe's refusal diagnostics. An ordering refusal
records the first inverted pair with `max_digits10` precision. This is useful
when diagnosing a new backend failure without weakening any acceptance gate.

## Rejected alternatives

- Ignoring order or adding an order tolerance would weaken the ascending
  eigenpair contract used by occupied-state consumers.
- Sorting eigenvalues alone would break their association with coefficient
  columns; physical residual tests explicitly reject that error.
- Rounding occupations to zero/two or rescaling the imported density would
  alter the candidate and conceal independent admission errors.
- Adding a CPU/PySCF eigen fallback would change production ownership and work.
  The fix makes no new eigen solve and uses no reference oracle.
- Changing captured/resident solver outputs is unnecessary for the demonstrated
  ordinary seed failure and would broaden resource/lifetime obligations.

## Invariants

- Reject nonzero solver info, malformed/aliased storage and nonfinite values.
- Preserve strict ascending validation after presentation is canonicalized.
- Preserve absolute eigen/metric gates of `1e-8` and the scaled residual gate
  of `1e-12`, plus the unchanged source-metric electron/occupation checks.
- Never mutate the imported density, overlap or retained warm/final state.
- Keep SAP/Modified Hückel as explicit experimental providers. Their CPU/PySCF
  preparation is not a production `InitialGuessSpec` or automatic strategy.
- Require independent final-state validation and a strictly greater than 5%
  complete-endpoint reduction against the current default before any future
  automatic-provider qualification.

## Evidence

- All four saved seeds pass independent CPU metric validation. Their unchanged
  bytes fail 12/12 old-library admissions and pass 12/12 fixed admissions.
- The complete native ordinary-eigensystem suite passes through finite Slurm
  GPU execution, including 384/768-AO analytic zero/two projectors, three
  workspace-reusing repeats, no reference solves, and unchanged physical gates.
- Host tests cover stable ties, exact one-ULP inversions, permutation cycles,
  all four-column permutations, nonfinite/dimension/alias refusals, nonzero
  solver info and incorrect eigenvalue/vector associations. ASan/UBSan pass.
- Real-device evidence uses frozen source `4ac8d4517f3323c5b7ba567270bdb2bd3323cde7`
  plus the four-file production patch, not the unrelated dirty current checkout.
  Full seeds and final densities remain in ignored local artifacts and on n1.
  Library/source/seed hashes and Slurm/cache receipts bind the observations.

The full canonical-consumer token pin is updated only for the reviewed ordinary
DF post-download presentation step. Generalized phase execution, provider call
order and the RHF/UHF captured-consumer pins remain unchanged.

## Complete-endpoint qualification

All 24 fresh-process energy/analytic-force endpoints pass independent energy,
force and physical-residual checks and paired final native density gates. Both
external providers pass all six fresh seed admissions. Maximum errors are
`7.28e-12 Eh` in energy, `3.77e-11 Eh/Bohr` in forces, `1.77e-12` physical residual
RMS, and `5.48e-11` paired AO density. The controlled fixed library SHA-256 is
`925fc6b7875e1aa1afbc9dcd2908520a78c19378796a7ceecf2b164baa970e4e`.

Relative to newly measured native MINAO controls, Hückel reduces complete median
wall time by **8.19% / 1.55%** at 48/96 atoms. SAP reduces it by **3.00% / -0.79%**.
The repaired Hückel path therefore fails the existing >5% gate at 96 atoms;
SAP fails at both sizes. Original fallback-inclusive timings cannot qualify a
new library. Numerical success does not authorize automatic promotion, a size
whitelist, or production dependence on an experimental CPU/PySCF provider.

See the [separate retained fixed-library campaign](../../../../benchmarks/results/pbe0-seed-admission-fix-20261012/README.md)
for exact medians, source audit, semantic work counts and reproduction commands.

## Revisit when

A new exact provider stack changes eigenpair presentation, or a resident/captured
consumer demonstrates a similar failure. Qualify those scopes independently;
do not infer safety or a speedup from this ordinary-seed repair alone.

## References

- [Original screening rationale](../performance/2026-10-12-pbe0-initial-guess-screening.md)
- `src/scf/solver/eigen_frame.cpp`
- `src/dft/cuda_ks.cpp`
- `src/scf/cuda/df_eigensystem.cpp`
- `tests/native/test_eigen_frame.cpp`
- `tests/native/test_df_eigensystem.cpp`
- `.artifacts/pbe0-seed-admission-fix-20261012/`
- n1: `/data/jzzeng/pbe0-seed-admission-fix-20261012/`
