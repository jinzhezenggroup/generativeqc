# Decision: Bind exact range-PBE curvature to the accepted Fock

Status: implemented
Date: 2026-11-21

## Problem

The original #2840 range-only PBE model has a native CUDA stationary SCF and
gradient. Its `(SR, LR, omega) = (0.20, 0.65, 0.33)` is not the WB97M-V model;
the existing exact nuclear response used the latter's fixed weights and omega.

## Decision

Authenticate unscaled PBE, the strict physical SCF generation, the full-range
primary and the exact LR correction as one model. Decompose exchange as
`SR * K(full) + (LR - SR) * K(erf)`. The correction can be negative. The
induced Fock uses the accepted signed, spin-scaled coefficient; the nuclear
quartet graph takes the corresponding physical fraction and keeps RKS/UKS
density conventions in the compiled contraction. A DF primary retains its
own full-range value, metric and derivative owners; only its LR term uses
the exact nuclear response.

Factorized four-center callbacks accept a bound runtime omega and carry a
separate artifact identity. Their named probes still evaluate the declared
baseline. Expanded fixed leaves and fitted two-/three-center radial leaves
do not acquire arbitrary-omega admission. No new JIT or reference provider
is invoked. WB97M-V's model and fitted LR admission stay separate.

The energy-only resource ledger queries the native rebuild layout for the
second exchange panel and the radial inventory for the actual provider
lifetimes. An environment-dependent DF LR policy is not silently overridden.

## Evidence

The original H-cluster cases at omega 0.33 cover Direct and DF-primary/Direct-LR,
RKS and UKS, including a negative LR-minus-SR correction. The public endpoints
run with `ResourceBudget`. Tests compare dense symmetry and dense/HVP parity
at `2e-8` and the HVP against two-step Richardson-extrapolated, independently
reconverged Libcint/Libxc energies on moving grids at `2e-6` Hartree/Bohr².
On the RTX 5090 (SM120), Slurm job 2286 passed all four molecular comparisons
and six existing WB97M-V endpoint cases. Job 2288 passed two runtime-omega
component cases against displaced Libcint integrals with the batched and scalar
schedules, plus twelve fixed-radial regressions. The host suite also checks
signed spin weights, fixed versus parameterized dispatch, and independent
state capacity for the two exchange panels. Build, cache and test logs are
under `build/hessian-gpu-sm120/pbe-rsh-*`.

## Revisit when

The fitted LR parameterization and molecular qualification are recorded in
[the fitted range-PBE decision](2026-11-21-cuda-pbe-range-fitted-hessian.md).
The CLI still requires an explicit MethodIR transport before it can express
this composition.
