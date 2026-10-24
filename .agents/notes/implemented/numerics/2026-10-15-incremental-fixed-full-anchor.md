# Decision: retain a fixed full-density anchor for screened incremental J/K

Status: implemented; experimental opt-in only, not default promotion
Date: 2026-10-15

## Problem

Revisit delta-D on master `5c9ce272b4a5a367a9b962e31a76248b4d5bc59c`.
There are distinct historical failures, not evidence that J/K is nonlinear:

- Screened full and delta energies can alternate above the energy gate. The
  October 7 full-energy-refinement repair already addresses this for CUDA KS;
  do not remove it or relax the energy threshold.
- Rolling a screened reconstruction into the next anchor adds a fresh screening
  omission at every delta step. The shared controller bounded that chain by
  overriding every screened rebuild interval, including 0 and 8, to 1. Thus
  even stationary trajectories had to alternate expensive full and delta builds.
- The retained October 11 RHF comparison regressed at the complete endpoint.
  Its 384-AO cold trajectory took 38 iterations (19 full, 19 delta, plus two
  final full builds), versus 19 baseline iterations. Its warm trajectory took
  one full SCF build, no delta builds, and still two final full builds. Thus
  extra convergence/finalization work is a separate cause of regression, not
  something a faster delta kernel alone could repair. Fixed anchors do not
  remove the strict incremental HF finalization fallback or prove warm wins.
  Its failed-point receipts do not diagnose nonconvergence and its invalid
  quartet census cannot establish a semantic-work speedup. Do not relabel these
  records as successful qualification of the new implementation.

GPU4PySCF 1.8.1 in the independent reference environment inherits
`direct_scf=True` from PySCF 2.14.0. Its HF `get_veff` subtracts `dm_last` when
that flag and history are present; RKS retains J/K state while evaluating XC at
the full density. This confirms that incremental J/K is an ordinary direct-SCF
technique, not that every provider/default/DF route has identical admission,
screening error, finalization or performance contracts.

## Decision

For a density-weighted screened provider, retain `D_anchor` and `J/K_anchor`
only from the most recent full-density build. Each accepted delta build uses
`D_current - D_anchor` and reconstructs against that same full anchor. It does
not overwrite the anchor with its screened result. Honor the requested maximum
delta count between full refreshes, including zero for no periodic refresh.

If a screened operator is `G_screen(D) = G(D) + e(D)`, each reconstruction is

```text
G_screen(D_anchor) + G_screen(D - D_anchor)
  = G(D) + e(D_anchor) + e(D - D_anchor).
```

The error depth is independent of the number of preceding delta builds. A
rolling screened anchor instead sums every intervening delta omission. This is
not a bound on the molecular energy error: existing provider screening and all
independent physical endpoint gates remain necessary. Fixed-anchor deltas may
be larger than rolling deltas, so fewer full calls alone do not prove a win.

Unscreened/exact-linear providers retain their rolling-anchor implementation.
The shared policy owns the distinction for both CUDA HF and CUDA KS. CPU
exact-linear execution is unchanged. No density-dependent screening threshold,
precision, public ABI, memory buffer, default selector or resource-planning
admission changes. `anchor_updates` continues to count actual delta-to-anchor
promotions and is therefore zero for fixed screened anchors.

## Invariants and rejected alternatives

- Nonlinear XC/nonlocal density consumers still see the full density.
- Full-energy refinement, strict full physical finalization, density/residual
  gates, failed-state invalidation and geometry-scoped anchor reset remain.
- Late-SCF RMS and nonfinite-RMS guards may force an earlier full refresh.
- Do not just lift the cadence cap while continuing to roll screened anchors;
  that restores the original error-accumulation defect.
- Do not make delta-D a default or claim a GPU4PySCF-equivalent speedup without
  fresh source/device-matched complete endpoint evidence.

## Evidence and qualification boundary

Host tests execute the production KS controller and HF kernel bodies with
sequential host thread dispatch. An independent thresholded linear operator
tests the full-anchor-plus-current-delta identity and includes the old rolling
screened chain as a negative control. Cases include both spin counts, intervals
0/1/3/8, nonfinite-RMS refresh, inactive batch items, anchor reset and build
counts. These tests do not establish GPU concurrency or molecular performance.

An additional ignored qualification probe links the actual compiled CUDA matrix
kernel object and executes both spin blocks across multiple CTAs with an
inactive batch item. On n1 in finite Slurm job 8670, 20 synthetic provider
applications at intervals 0/1/3/8 give full/delta counts 1/19, 10/10, 5/15,
and 3/17, respectively. Slurm job 8671 repeats the probe under Compute
Sanitizer memcheck and synccheck, each with zero errors. These are controller
identity/census checks with a synthetic linear provider, not molecular timings
or admitted-quartet work evidence. Local receipts are
`.artifacts/delta-anchor/parallel-gpu-{probe,sanitizer}.log`.

The CUDA molecular regression has independent PySCF 2.14.0 STO-3G RHF HeH+
and UHF HeH fixtures at both original and moved Bohr geometries. The reference
was evaluated in a finite Slurm job on n1; retained local raw values and package
metadata are under `.artifacts/delta-anchor/pyscf-oracle.log`. Acceptance is
1e-8 Eh in energy and 1e-7 Eh/Bohr in every force component. Test failures now
print the diagnostic rather than returning an unexplained failure status.

The first full-library CUDA run passed those HF fixtures but stopped at the KS
baseline's final-state export, before its incremental solve. That fixture paired
25% exact exchange with 100% PBE exchange, whereas the current model validator
requires the admitted PBE0 composition. Repair only this qualification fixture
to use 75% PBE exchange, including its independent CPU fixed-density energy and
residual reconstruction. Explicitly select and restore the ordinary iteration
owner for this controller test. Do not bypass or loosen the physical validator.

Final source-matched reduced-build qualification passed:

- 395 focused host tests, including unsupported KS resource-plan admission,
  default-promotion inventory, work-counter publication and CPU closure;
  the native CPU Fock-provider executable also passed.
- Native CUDA RHF/UHF at requested intervals 0/1/8, including moved geometry
  and independent pinned energy/force gates. The screened 12-iteration RHF
  control executes 6 full + 6 delta builds at interval 1, versus 2 full + 10
  delta builds at interval 8. Both retain two final full physical builds.
- Native CUDA RKS/UKS PBE0 controller, fresh-solve, CPU fixed-density physical
  reconstruction and final-state export gates passed in Slurm job 8698.
- Twelve prepared-batch water RHF and OH UHF/def2-SVP spherical endpoints,
  baseline/incremental cold, verified density-warm and moved geometry, passed
  independent PySCF 2.14.0 energy/force gates. Maximum absolute energy/force
  errors were 1.990e-13 Eh and 1.351e-10 Eh/Bohr. The warm points actually use
  the retained density without fallback; each converges in one iteration.
  Incremental warm execution still has one full SCF build, no delta and two
  final full builds. This repair does not remove that separate overhead.
  This complete small-system correctness run is Slurm job 8702, not a repeated
  performance campaign or evidence for production-profile default promotion.

Broader sanitizer qualification is not fully clean. In Slurm job 8706, both
native HF and focused KS executables pass memcheck with zero errors. The
following HF synccheck run reports 96 divergent-warp barrier errors in the
unchanged `reduce_shell_pair_density_bounds_kernel<false>` during its first
ordinary full-density baseline, before constructing/executing the incremental
candidate. The test then fails closed, so the subsequent KS synccheck command
is not executed. Keep the failing receipt in
`.artifacts/delta-anchor/native-gpu-sanitizers.log`; do not describe this as
complete sanitizer acceptance or infer that it caused missing historical
failure records. The focused changed-controller probe remains clean under both
tools, but that does not discharge this separate full-provider blocker.

The library SHA256 is
`488277dff663ace122a6aa65e094a4b4c89cd81b6072df4a0b3a25426df02ff9`;
its native source-content identity is
`da8100d5a00d1e7193dc2708aa9f244598fc3c2f8eaabe0ea3bf5e56dfb6b88f`.
These identify the dirty master-plus-patch build, not a clean publishing SHA.
Logs/configuration and the final patch stay in ignored `.artifacts/delta-anchor/`.
An initial standalone Python harness compared its default Cartesian basis with
PySCF's spherical basis and correctly rejected the baseline. The corrected
harness explicitly binds both representations; it does not relax the gates.
Repeated single-point calls were not counted as verified density-warm evidence;
the final qualification uses `prepare_batch(warm_start=True)` and checks the
reported warm/fallback flags. Standalone and rejected receipts remain local.

Production-profile cold/warm/moved 192/384/768-AO performance qualification is
still required. Neither host tests nor a reduced non-AOT correctness build can
replace that campaign. Keep the negative default-promotion decision unchanged.

## Repeated diagnostic endpoint measurements

The follow-up performance experiment uses the same reduced, generic/non-AOT
Release library identified above, not a production-profile build. Scientific
controls are RHF/def2-SVP spherical, FP64, energy/density/screening tolerances
1e-12/1e-9/1e-12, and the current default native Modified Huckel initialization.
Geometries are nested prefixes of the existing synthetic water-32mer fixture;
the moved endpoint displaces the final atom by +0.03 Bohr in z.

Each trial constructs an independent calculator and prepared owner. Owner-cold
timing includes construction, preparation, native initialization, SCF, physical
finalization and returned host energy/forces; common process/library/CUDA
bootstrap is excluded. Both arms receive an unmeasured complete priming trial.
Warm and moved calls reuse only their own owner's density, with warm-use and
no-fallback checks. Teardown is recorded separately. The six projected arm
orders have three OFF-first and three fixed8-first trials. No reference density
or Fock is imported into production execution.

Slurm allocations 8730 (96 AOs), 8735 (192 AOs) and 8731 (384 AOs) each retain one
assigned RTX 5090 across both arms, four fixed CPU threads, device UUID, library/source/harness
identities, raw samples, stage counts and independent CPU PySCF energy/force
references. Every measured repeat must meet 1e-8 Eh and 1e-7 Eh/Bohr maximum
absolute error gates; a matching iteration branch is not an accuracy filter.

Median complete owner-cold / warm / moved energy-plus-force times, seconds:

| AOs | Repeats per arm | Phase | OFF | fixed8 | Wall-time change |
| --- | --- | --- | ---: | ---: | ---: |
| 96 | 6 | cold | 1.412887 | 1.242965 | -12.03% |
| 96 | 6 | warm | 0.183119 | 0.308310 | +68.37% |
| 96 | 6 | moved | 1.123288 | 0.938781 | -16.43% |
| 192 | 6 | cold | 13.082864 | 11.313082 | -13.53% |
| 192 | 6 | warm | 1.883388 | 3.257607 | +72.97% |
| 192 | 6 | moved | 11.051389 | 8.292338 | -24.97% |
| 384 | 3 | cold | 47.643624 | 39.463198 | -17.17% |
| 384 | 3 | warm | 5.952936 | 10.713360 | +79.97% |
| 384 | 3 | moved | 42.220389 | 28.780954 | -31.83% |

All 90 formal RHF endpoints and 18 unmeasured priming endpoints pass the
independent numerical gates. Their maximum energy/force errors are
2.774e-11 Eh / 9.330e-11 Eh/Bohr. The 384-AO diagnostic has only three repeats
per arm, in projected orders OFF/fixed8, fixed8/OFF, fixed8/OFF, not a completed
six-order balanced cohort. Its paired cold/moved candidate-to-baseline ratios
are 0.8233--0.8341 / 0.6792--0.6886; warm ratios are 1.5721--1.9024.

Every paired cold/moved sample favors fixed8; every paired warm sample favors
OFF. These are real but restricted diagnostic wins, not a uniform endpoint win
or evidence for default promotion. The 192-AO cold/moved target trajectories
actually require 17/14 iterations instead of OFF's 16/13, so their timing gains
are not caused by a relaxed stopping rule or fewer SCF iterations.

Stage counts explain the warm loss: both arms perform one full SCF build and
zero delta builds, but fixed8 still performs two final full physical builds;
OFF reuses the qualified converged state without another final build. For cold
96 AOs, OFF has 18 full SCF + one final build; fixed8 has two full + 16 delta
SCF builds + two final builds. Both cold arms additionally charge seven native
preliminary Huckel Fock builds. Accepted populations report one target attempt,
zero discarded target builds, and no warm fallback. Public HF `fock_builds`
remains unavailable; the separately recorded stage/initialization counters are
not a complete operator inventory. Baseline quartet counts are unavailable;
fixed8 quartet census is valid at 96 AOs but unavailable for bounded streaming
at 192/384 AOs. Missing work counts remain unknown, not zero or estimated savings.

Failure receipts are retained rather than folded into speedup medians:

- The interval-1 arm is a same-binary old-cadence control, not an exact historical
  executable. Pilot allocation 8725 fails its 96/192-AO cold endpoints with
  `SCF did not converge`. A detailed 96-AO follow-up retains 100 returned target
  iterations (50 full + 50 delta), plus 100 discarded iterations from the
  automatic Huckel retry. Density RMS is about 1.16e-11, but the solve still does
  not pass convergence and publishes no forces. Do not call that a fast or
  accepted endpoint, or claim fixed anchors fix every interval/provider.
- The initial GPU4PySCF reference attempt fails its tight convergence request
  before candidate timing. The accepted RHF measurements instead use converged
  independent CPU PySCF references with unchanged endpoint acceptance gates.
- The original multi-method allocation 8730 stops at its 96-AO UHF CPU reference,
  which also fails the requested convergence check. Its completed RHF population
  is retained, and allocation 8735 measures RHF 192 AOs separately. This is not
  a completed UHF/KS campaign or a broad guard-domain qualification.

Harness snapshots, receipts, references and raw samples stay in ignored
`.artifacts/delta-anchor/performance-*`; the reducer is
`.artifacts/delta-anchor/analyze_performance.py`. The native source/library and
default selectors are unchanged during this performance experiment. Revisit
warm admission/final-state reuse under the existing physical contract; do not
remove required final validation merely to erase its measured cost.

## References

- The subsequent [full-only resident warm finalization repair](../performance/2026-10-15-incremental-full-only-warm-finalization.md)
  addresses the measured two-build warm overhead without licensing screened
  delta Focks for retained physical publication. The earlier timings above are
  historical pre-repair measurements, not measurements of that subsequent patch.
- `src/scf/types.hpp`, `src/scf/cuda/scf_matrix_kernels.cu`,
  `src/scf/cuda_rhf.cpp`, `src/dft/cuda_ks.cpp`.
- `tests/python/test_incremental_direct_jk_fixed_anchor.py`,
  `tests/python/test_ks_incremental_controller.py`,
  `tests/native/test_incremental_direct_jk_cuda.cpp`.
- [Earlier late-SCF policy](../../proposed/2026-10-07-pbe0-adaptive-incremental-direct-jk.md).
- [Full-energy refinement](2026-10-07-incremental-ks-full-energy-refinement.md).
- [Negative default-promotion closeout](../performance/2026-10-11-incremental-direct-jk-negative-closeout.md).
