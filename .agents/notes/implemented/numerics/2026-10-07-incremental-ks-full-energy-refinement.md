# Decision: qualify incremental KS energy with consecutive full-density builds

Status: implemented
Date: 2026-10-07

## Problem

The exact-direct CUDA KS controller can reach a stationary density and physical
residual while its screened full-density and reconstructed anchor-plus-ΔD
energies still alternate. Screening the full density and screening its
increment are not identical operations: they can omit different contributions,
even though the underlying unscreened J/K operator is linear.

In the 96-atom PBE0 cold regression, density and residual gates passed around
iteration 22, but energy changes remained around 1e-9 Ha, above the requested
1e-12 Ha gate. The old controller required this alternating energy gate to pass
before it could enter full-density finalization. This circular ordering delayed
accepted energy-and-force endpoints to 76–78 iterations. Early trajectories
agreed with full-density execution; a large early ΔD was not evidence of an
anchor reconstruction error.

## Decision

Once all existing density-change, RMS physical-residual and AO maximum-residual
gates pass, enter a monotonic full-density energy-refinement phase. Do not wait
for the energy gate to initiate this phase. Keep the original DIIS history,
energy tolerance, density tolerance, strict-FP64 operator and screening policy.
The refinement phase never readmits delta builds within that solve, including
if a subsequent full build temporarily increases a residual.

Require at least two full-density refinement builds before declaring energy
convergence. The energy difference then compares two evaluations of the same
full target operator rather than a full evaluation against a screened delta
reconstruction. Every original physical convergence gate must still pass on the
qualifying build.

For RKS without ECP, this qualifying full build is the physical final audit:
it evaluates the current F[D], energy, density proposal and unshifted residual,
just as ordinary full-density RKS does. The shared final-state export validator
remains mandatory and unchanged. Reclassify the actual qualifying provider
build from anchor/periodic work to post-SCF full audit work; do not invent an
additional provider call. The full/delta/final counts must still sum to the
actual Fock-build count.

UKS and ECP keep their separate bounded corrective closure, including existing
occupation stabilization and the unshifted final-state validator. A fresh solve
resets both refinement admission and its full-build count. Incremental resource
planning remains experimental/unsupported where it was already unsupported;
no new public option, allocation, ABI change or default promotion is introduced.

## Rejected alternatives

- Relaxing the 1e-12 Ha energy gate hides the ordering bug and changes science.
- A late-SCF density-RMS tuning value is an experimental admission policy, not a
  repair for an energy gate that blocks its own full-density qualification.
- Merely initiating full refinement while retaining the redundant RKS corrective
  closure improved one sample to 35 iterations, but a second sample exhausted
  the four-correction limit at iteration 37 and returned no forces. That rejected
  trial is retained as negative evidence, not counted as a fast endpoint. Its
  extra full-build energy qualification reintroduced sensitivity to numerical
  summation noise after the normal RKS gates had already passed.
- Repeatedly clearing DIIS, changing screening thresholds or using a CPU oracle
  in production is unnecessary and changes more than this termination defect.

## Invariants

- Final energies and analytic forces belong to the same accepted density and
  full physical operator; failed solves cannot publish a final state or warm
  density cache.
- The existing energy, density, RMS residual, AO maximum residual, electron-count
  and final-state identity gates remain unchanged.
- Refinement admission does not imply convergence: two full energies and all
  physical gates are still required, within the existing iteration budget.
- No delta build occurs after refinement starts. Existing screened-provider
  rebuild cadence remains applicable before refinement.
- Every reported anchor, delta and final build corresponds to actual execution;
  reclassification preserves the total. Unavailable quartet counters cannot be
  used to claim an admitted-quartet speedup.

## Evidence

The executable host controller probe tests the real admission, reconstruction
and accounting methods against an independent linear J/K oracle. It verifies
that each physical gate is necessary, that a blocked energy gate does not block
refinement, and that two full builds and final reclassification conserve work.
Source-contract tests protect reset and the ordering of refinement before the
unchanged energy convergence gate.

Focused native GPU tests compare exact-exchange RKS/UKS against independently
rebuilt CPU integrals/XC, exercise permissive and unreachable delta gates and
provider/SCF screening-policy mismatch, validate final-state export, and repeat
a cold solve on the same owner to catch stale refinement state. The RKS fixture
uses a valid localized two-electron seed: symmetric H2's core seed is already
stationary and no longer legitimately guarantees pre-refinement delta work.

The retained cold experiment uses water-96, 768 spherical def2-SVP AOs,
48 × 16 × 32 grid points per atom, exact direct J/K and strict FP64. Energy,
density and screening tolerances are 1e-12 Ha, 1e-10 and 1e-12, respectively,
with a 100-iteration budget and external delta density-RMS threshold zero.
Each sample creates a fresh process/calculator/prepared owner, with no warm
density seed. Complete synchronized timing includes preparation and the first
energy **and analytic-force** execution, including force-owner runtime setup;
it excludes native-library compilation. Program-cache first fills are retained
but identified separately from cache-populated paired cold solves.

The independent reference is the existing GPU4PySCF cold result in
`benchmarks/results/pbe0-composed-baseline-20261005/reference-96.json.gz`, with
identical scientific protocol and gates of 1e-8 Ha energy and 1e-7 Ha/Bohr maximum
force error. Numerical agreement does not alone establish a performance win.
Failed and interrupted trials are retained separately from accepted complete
endpoints. See the compact experiment records for paired timings and complete
iteration histories.

The repaired same-binary off/on/on/off pair has medians of 214.07210 s with ΔD
off (27/25 iterations) and 214.25545 s with ΔD on (32/26 iterations). The original
broken ΔD cohort has a 421.15756 s median (78/76 iterations). The observed repair
removes 49.127% of that regression, but ΔD itself has no established cold win:
the primary medians differ by only 0.08565% and sample ranges overlap widely.
The repaired first-fill sample is separately retained at 219.54439 s and 30
iterations. All five repaired endpoints return forces and pass the independent
oracle, with maximum errors below 7.8e-11 Ha and 8.3e-11 Ha/Bohr. Seventy-nine
focused host tests and both incremental native GPU test targets pass.

The reconstruction patch preserves the exact measured pre-format source bytes.
A subsequent clang-format join of the full-energy-history boolean declaration
changes the checkout's source hash but not its behavior; both focused native GPU
targets were rerun successfully after that formatting-only change.

## Consequences

Full-density late iterations may cost more per build than delta reconstruction,
but they make energy qualification use a consistent target operator. This is a
correctness/termination repair, not evidence that incremental mode should become
the default. Cold wall time must be measured directly rather than normalized by
iteration count or inferred from isolated J/K timing.

## Revisit when

Reconsider the full-density refinement boundary only if a screened incremental
operator has a quantitative energy-error contract below the requested SCF gate
and independently accepted complete endpoints. Reconsider the retained UKS/ECP
closure only with independent final-state, occupation and force qualification;
RKS timings do not justify removing those policies.

## References

- `src/dft/cuda_ks.cpp`: incremental admission, full-energy refinement and final
  physical convergence accounting.
- `tests/python/test_ks_incremental_controller.py` and
  `tests/native/test_ks_cuda.cpp`: executable controller and physical-state gates.
- `docs/user/ks_options.md`: current incremental KS behavior.
- `benchmarks/results/pbe0-incremental-refinement-20261007/`: frozen cold evidence.
- PR #2044 plus the screened generated J provider on baseline master
  `e6f1a6b0c16b2763ee65e124188fdfbea37a68a5`; simulated merge tree
  `34b6c9cc478cf2dd194f0217cb9229b44ed72a0f` and repaired behavior on master
  `f05015e7c71809155b9be0aa69c3ff62d5d89e30`.
