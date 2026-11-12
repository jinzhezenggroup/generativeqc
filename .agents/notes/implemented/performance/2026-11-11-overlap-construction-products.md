# Decision: use shared FP64 products for overlap construction

Status: implemented
Date: 2026-11-11

## Problem

After removing cubic pair enumeration from CUDA nuclear derivatives, exclusive
node4 HF96 DF cold qualification still lost to default GPU4PySCF by 20 ms. The
remaining overlap cache miss spent about 386 ms constructing and checking the
symmetric orthogonalizer with three unblocked scalar matrix products. Accelerating
only the eigendecomposition left this host preparation untouched.

## Decision

When an owner supplies an eigen operation, construct `X = U diag(s^-1/2) U^T`
and evaluate the complete `X^T S X` proof through the existing
`solver::ValidationProducts` FP64 host adapter, preferring its available OpenBLAS
provider. This is the same native host provider already used for qualified cold
validation, not a CPU electronic-structure oracle or a new runtime dependency.
Provider transpose flags avoid two explicit transpose temporaries. The adapter
retains scalar products if OpenBLAS is unavailable or explicitly disabled by its
existing private diagnostic selector. Benchmark processes leave that selector
unset, so the measured path is the ordinary public default.

The shared-SCF dependency audit admits that exact product-adapter header only
for production `initial_guess/overlap.hpp`, as it already does for atomic
preparation's specific solver leaves. It does not admit solver drivers, DIIS or
mutable provider ownership, and reference/other initial-guess modules retain
their original restrictions. Boundary tests exercise both the admitted leaf and
rejected dependencies without changing the measured production sources.

## Invariants

- Keep the full symmetric inverse square root, not a Cholesky factor or truncated
  eigenbasis. Preserve the strict eigenvalue rejection below `1e-10`.
- Preserve finite-output and complete metric-identity checks at `1e-8` and the
  independent eigenframe validation. Do not alter any convergence tolerance,
  initial guess, grid, precision policy, or GPU4PySCF setting.
- Leave `scf::reference::multiply` and the independent reference decomposition
  unchanged. Provider accumulation may differ at FP64 roundoff scale and needs
  independent endpoint qualification.
- Retain coordinate-and-value cache identity, bounded storage, and clearing stale
  state before any failing replacement. The callback and its owner are borrowed.
- Keep the existing provider's host thread ownership; do not add benchmark-only
  thread parallelism or hardware-specific crossover settings.

## Evidence

`test_initial_density.cpp` adds an analytic rotated two-dimensional spectrum and
its independently known dense inverse square root. An intentionally incorrect
frame must still fail the metric proof. The suite also covers cache replacement,
failure clearing, nonfinite input and the strict singularity boundary. Both
provider-enabled and scalar-fallback test binaries run in the finite Slurm
qualification, along with the full native DF suite and independent tight-model
HF/DFT endpoint oracles.

Source-bound build, six-permutation complete-endpoint qualification and any
subsequent full README cohort are retained in
`.artifacts/df-hf-overlap-20261111/` and the corresponding node4 deployment.
No favorable individual repeat may replace a published matrix point. The
previously disclosed dense-reference DF fitting failures remain failures;
same-fitted-model qualification does not waive them.

Exclusive node4 job 1568 acquired 36 fresh processes / 144 complete E+force
endpoints, cycling all six baseline/candidate/default-reference orderings at 12
and 96 atoms. HF96 cold medians were 17.723960 s before either optimization,
16.254247 s with both, and 16.543668 s for unchanged default GPU4PySCF. All six
candidate cold endpoints (16.249662–16.257009 s) preceded every reference cold
endpoint (16.533740–16.549327 s). Native target work remained 11 iterations /
12 Focks plus seven preliminary atomic builds; reference remained 10 / 11.
Warm native/reference medians were 3.776703 / 6.907771 s. These observations
qualify this endpoint, not a universal performance claim or a replacement for
the full README matrix.

All same-phase force differences versus the original native image were below
`8e-13 Eh/Bohr`; independent oracle tests passed (9 tests), as did 48 weighted
DF finite-difference gates. The extra all-phase `1e-10` identity diagnostic
still fails for the baseline against itself because ordinary cold/warm
refinement differs, and remains explicitly failed in the audit. No production
gate is changed to accommodate that diagnostic. The candidate passes the
original all-phase scientific gates and the retained independent tight
same-fitted-model HF96 quarter-budget gates.

The measured production source identity is
`9d305b13134f8f32cdee07a5364207784ae4e3101dd95ec66d618d1e63c01d60`,
ELF SHA256 `2200e9fdc8ccb81cb4fda6bf2d42e142e53bf02d57906bbd69e442c00f7bc1f6`.
Baseline/candidate native compiler command lists are byte-identical. CPU/NUMA
binding, GPU processes and host telemetry are retained, including a maximum
IO pressure avg10 of 0.18; do not claim zero pressure throughout the campaign.

The subsequent complete README cohort, exclusive node4 job 1569
(2026-11-11 21:25:45–22:39:25 +08), completed all 128 processes / 512 endpoints
with no failure, retry or point substitution. Native wins all 64 cold and all
64 warm observations. The independently acquired HF96 DF cold point is
16.258817 s versus 16.558978 s reference; warm medians are 3.774124 / 6.910326 s.
All 1,024 cross-engine full-vector comparisons pass, while the same six
independent dense-reference failures remain disclosed (58/64 pairs qualify).
Full-vector and compact replays under normal and optimized Python reproduce
both homepage SVGs byte-for-byte. The tracked evidence aggregate remains below
its unchanged 90,505,216-byte cap.

## Rejected alternatives

Removing the metric proof, weakening SCF convergence, and changing reference
defaults would not be valid optimizations. Replacing the independent reference
multiply would also weaken the oracle boundary. Moving these three products to
CUDA would require additional workspace and lifetime plumbing; reuse the existing
qualified host-provider boundary first rather than introducing a second owner.

## Revisit when

An owner can provide bounded, synchronously drained device products and a complete
endpoint qualification demonstrates a further gain without changing these gates.
