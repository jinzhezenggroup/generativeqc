# Diagnosis: GPU4PySCF PBE0 iteration inflation from absolute DIIS truncation

Status: proposed (benchmark-only normalized-DIIS control; no production change)
Date: 2026-10-09

## Problem and source boundary

The retained 96-atom PBE0/def2-SVP water32mer comparison reports native cold
iterations `17,18,17`, versus GPU4PySCF `37`; moved geometry reports native
`12,14,12`, versus GPU4PySCF `47`. These are not equal-work timings.

Those native results belong to source
`4385f72751b829883407c01106186917c344317b`, using its supported CUDA/MINAO
implementation, not the older dirty VibeQC worktree. The raw baseline is
`.cache/pbe0-96-cold-test/2715/native-round{0,1,2}.json`; its independent
reference is `.cache/pbe0-96-cold-test/2709/reference.json`.

This investigation changes neither production source nor benchmark defaults.
It uses the exact installed GPU4PySCF 1.8.1/PySCF 2.14.0 and the existing
benchmark's independently reconstructed grid/basis and full-density Fock
wrapper. The native baseline is reused, not rerun or relabeled as a current
worktree measurement.

## Main cause: scale-dependent loss of DIIS directions

In the installed `gpu4pyscf/lib/diis.py`, `DIIS.extrapolate` diagonalizes the
augmented Pulay matrix and discards eigenvalues of absolute magnitude below
`1e-14`. The constraint row/column remains order one, while the residual Gram
block shrinks quadratically with the SCF error. Small residual magnitude is
therefore confused with dependence of useful history directions.

The observed stock trajectory first discards one direction at cycle 14. By
cycle 20 it discards seven directions out of the eight-history augmented
system: the Gram maximum diagonal is only `3.985e-15`. Near this limit the
remaining constraint modes produce nearly uniform Fock mixing rather than
effective extrapolation. The gradient tail takes many additional full J/K
evaluations despite negligible remaining energy error.

The native CUDA KS loop already passes `normalize_metric=true` to
`launch_update_diis_kernel`. That solver divides the entire Gram block by
its maximum diagonal before its pivot test, and retires the oldest history
entry after a failed solve while more than two entries remain. Relevant pinned
source: `src/dft/cuda_ks.cpp`, `src/scf/cuda/scf_diis_kernels.cu`.

A single common scale preserves the exact constrained Pulay coefficients:
in `B c = lambda 1`, `1^T c = 1`, replacing `B` with `B/s` only rescales
the Lagrange multiplier. Normalizing each history error separately would
change the problem and is **not** the control tested here.

## Controlled evidence

Finite Slurm jobs on n2/node2, partition `main`, `gpu:pro6000:1`, preserve
Slurm device visibility. Hardware is RTX PRO 6000 Blackwell, not RTX 5090.
The 768 spherical AOs and 2,359,296 explicit grid points are unchanged.
All arms retain energy/gradient/screening thresholds
`1e-12 / 1e-10 / 1e-14`, history length eight, exact full-density J/K,
native GPU XC and complete grid-responsive analytic forces.

Initial four-arm control, job 2739:

| GPU4PySCF arm | Cycles | Actual SCF full J/K builds |
| --- | ---: | ---: |
| Stock solver, observed only | 42 | 43 |
| Common Gram normalization only | 18 | 19 |
| Normalization plus native-like AO error metric | 20 | 21 |
| Normalization plus AO metric plus DIIS starting at cycle zero | 18 | 19 |

Thus normalization alone removes 24 cycles without relaxing the GPU4PySCF
convergence contract. Changing the error metric/start cycle is not needed
to explain the large discrepancy. The recorded initial-density arrays are
retained for direct seed identity checks; no native or reference converged
density is supplied to any cold arm.

Two additional fresh-process repetitions, job 2740, confirm the contrast:

| Arm | Cycles, all three samples | Full SCF J/K builds, all three samples |
| --- | --- | --- |
| Stock GPU4PySCF | 42, 43, 37 | 43, 44, 38 |
| Common normalization only | 18, 20, 19 | 19, 21, 20 |
| Existing native MINAO baseline | 17, 18, 17 | 17, 18, 17 |

The stock/normalized medians are 42/19 cycles. All eight diagnostic endpoints
have **bitwise identical GPU4PySCF initial densities**: maximum array
difference zero, including the AO-metric/start-cycle controls. Their final
density arrays differ by at most `6.803e-12` in the common GPU4PySCF AO order.

Every initial control passes the existing independent complete endpoint gates
of energy error <= `1e-8 Eh` and force error <= `1e-7 Eh/Bohr`. Maximum
errors in the four-arm control are `2.729e-12 Eh` and `3.257e-12 Eh/Bohr`.
Across all eight endpoints, comparison with the original GPU4PySCF reference
gives maxima `3.638e-12 Eh` and `3.614e-12 Eh/Bohr`; an additional independent
comparison against **each of the three retained native cold endpoints** gives
maxima `9.095e-12 Eh` and `3.775e-11 Eh/Bohr`. Every comparison passes. The
strict aggregate also rechecks unchanged thresholds, final orbital-gradient
and AO-maximum gates, finite 96-by-3 forces, counts and seed identity.
Numerical reduction noise can move the exact last cycle, so a single cycle
count is not a deterministic method constant.

## Secondary differences and interpretation traps

- **Different residual definitions.** GPU4PySCF stops on the unnormalized
  global norm of `2 C_virtual^T F C_occupied`. Native CUDA KS checks density
  RMS, AO commutator RMS and the actual maximum AO commutator element. Equal
  numerical tolerances on these quantities do not imply equal stopping rules.
  In the stock control, the energy/density-RMS/AO-residual gates pass at
  callback cycle 25 while the orbital gradient is still `6.779e-10`.
  This is a diagnostic milestone, not a replay of the native solver's
  next-density proposal gate at the same cycle.
- **Energy roundoff at the last gate.** At total energy approximately
  `-2441.6 Eh`, one FP64 ULP is `4.547e-13 Eh`. An energy-change tolerance of
  `1e-12` is only about 2.2 ULPs. The original reference stops at 37, whereas
  the observed control already passes the gradient gate at 37 but has
  `delta_E=1.819e-12`, and stops at 42. Native histories likewise retain
  late energy-gate-only tails. Neither samples nor thresholds may be changed
  merely to obtain a preferred iteration count.
- **Counting convention.** GPU4PySCF evaluates the initial potential before
  its counted cycles, hence `cycles + 1` full SCF J/K builds in these arms.
  Native ordinary CUDA RKS reports its actual target Fock builds as the
  iterations. This is a one-build offset, not an explanation for twenty
  additional cycles; analytic-force contractions are separate.
- **MINAO is not a bitwise common seed.** Native MINAO adds trace normalization
  and bounded metric-occupation projection, whereas the stock GPU4PySCF arm
  uses its own MINAO density. Their initial physical energies differ
  (`-2367.203181 Eh` native versus `-2439.927913 Eh` GPU4PySCF). The causal
  comparison nevertheless keeps the *GPU4PySCF* seed unchanged between arms;
  it does not depend on equating the two implementations' MINAO preparation.
- **Telemetry names are not mathematical definitions.** The benchmark's
  GPU `density_rms` field stores `norm_ddm`, a Frobenius norm; actual matrix
  RMS here divides it by 768. The pinned native public
  `physical_residual_max`/`density_change_max` fields describe the maximum
  of per-spin RMS values, not the largest AO element. The native CUDA solver
  independently enforces the actual maximum residual internally. This note
  does not change the diagnostics schema or infer an AO maximum from RMS.

## Proposed policy and invariants

Retain the stock GPU4PySCF arm as the stock comparison. If a normalized-DIIS
arm is promoted into benchmark tooling, label it explicitly, expose the
modified policy and continue reporting both actual work and complete endpoint
timing. Do not silently optimize the reference or weaken its stopping rule.

The present endpoint timings are **diagnostic**, not clean performance
measurements: callbacks add matrix products and synchronization, and each
process also retains its initial/final density. No speedup is inferred from
the iteration reduction or these instrumented timings. Default production
paths must never consume an oracle density as a cold seed.

This diagnosis is established for the retained 96-atom PBE0 case; it is not
proof that every molecule, functional, GPU4PySCF version or DIIS configuration
has the same tail. The moved-geometry discrepancy has source-level motivation
but no separate causal replay in this investigation.

## Evidence and reproduction

Local evidence root: `.cache/pbe0-scf-investigation-20261009/`.
Remote evidence root: `/data/jzzeng/pbe0-scf-investigation-20261009` on n2.
`diagnose.py`, `run-n2.sh`, `repeat-n2.sh`, `summarize.py`, JSON histories,
initial/final NumPy densities, exact installed/pinned source snapshots and
SHA256 identities remain outside production ownership. `summary.json`
rechecks all endpoint gates, final residuals, build counts and seed identity.

The saved source snapshots identify the exact code used, rather than an
unpinned upstream latest version. Relevant installed source locations are
`gpu4pyscf/lib/diis.py:219`, `gpu4pyscf/scf/diis.py:50`, and
`gpu4pyscf/scf/hf.py:137`/`:305`. Relevant pinned native locations are
`src/scf/cuda/scf_diis_kernels.cu:215`, `src/dft/cuda_ks.cpp:1341`/`:2015`,
and `src/scf/initial_guess/minao.cpp:110`.

Repeat using an eligible finite allocation on n2:

```bash
srun --partition=main --nodelist=node2 --gres=gpu:pro6000:1 \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=48G --time=00:15:00 \
  bash /data/jzzeng/pbe0-scf-investigation-20261009/run-n2.sh
```

## Revisit when

GPU4PySCF changes its DIIS conditioning policy; new molecular/functional
domains are qualified; or a transparently labeled benchmark control needs
clean, complete-endpoint performance qualification. Fixing telemetry naming
requires its own compatibility-aware change, not reinterpretation of old logs.

## Subsequent benchmark correction

After this investigation, the user requested fixing our benchmark without an
upstream report. The implemented metric/versioning, stopping-policy labeling
and public-output timing decisions are recorded in
[the compatibility note](../implemented/compatibility/2026-10-09-benchmark-scf-metric-contract.md).
The original diagnostic observations and archived artifacts remain unchanged.
