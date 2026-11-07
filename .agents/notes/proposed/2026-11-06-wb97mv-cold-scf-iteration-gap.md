# Diagnosis: explain WB97M-V cold SCF steps before tuning the solver

Status: proposed
Date: 2026-11-06

Follow-up: [measured provisional RMS handoff and rejected unrebased history transfer](../implemented/performance/2026-11-06-vv10-provisional-rms-handoff.md)
supersedes the proposed handoff experiment below. It attributes two avoidable
initializer Focks, not the complete native/reference cycle difference.

## Problem

The retained 96-atom full-Direct GPU4PySCF endpoint converges in 10 reported
cycles, while native WB97M-V reports 18. Optimizing pair kernels alone does not
explain the additional J/K applications. Do not assume this is another public
stopping-default mismatch or that all 18 native evaluations use the full target.

## Evidence and confirmed mechanism

No new GPU job is needed to establish the following. The existing actual A100
and RTX5090 records have the same native operator sequence:

| Native steps | Operator used | Count |
| --- | --- | ---: |
| 1--9 | Semilocal/RSH initializer, omitting VV10 | 9 |
| 10 | First current-density full target | 1 |
| 11--14 | Inner solve with the VV10 potential frozen at step 10 | 4 |
| 15--18 | Current-density full target and final confirmation | 4 |

Thus 18 is 13 provisional evaluations plus 5 full-target evaluations, with
4 executed VV10 feature pair sweeps. The stationary-density confirmation reuses
an exact-key pair result. GPU4PySCF reports 10 cycles and 11 effective-potential
builds. Those are different work decompositions, not identical loops taking
different numbers of steps. Reducing native steps by disabling the outer solve
can increase expensive pair work; judge complete endpoints and both work counts.

The stage transition also changes the fixed-point problem substantially:
the recorded residual at step 9 is about 1.46e-7, but introducing full VV10 at
step 10 raises it to about 5.06e-5. These history fields are **maxima of spin
RMS norms**, despite their `physical_residual_max` name, not AO element maxima
or the global MO-gradient norm. They cannot replace either of those gates.

At each provisional-to-full transition, the native owner clears the DIIS history
and restarts the target energy epoch. This prevents combining incompatible
operators or using a provisional energy as a full-target energy baseline, but
also forfeits previous extrapolation information. It explains why completing
the initializer is not equivalent to already converging WB97M-V.

## Additional independently verified trajectory differences

- Native default is **Huckel**, not MINAO. The executed receipt says `used`,
  with 7 preliminary atomic Focks and about 0.124 s preparation in Slurm16019.
  These are not 7 extra molecular iterations. The retained actual A100
  GPU4PySCF record uses **MINAO**. Their separate contribution to iteration count
  has not been isolated; do not claim that replacing the guess saves 8 steps.
- Native CUDA KS forms the CDIIS Gram matrix from the raw AO commutator
  `FDS-SDF`. GPU4PySCF 1.9.0 supplies an orthonormal `Corth` to CDIIS and uses
  `Corth.T @ (SDF-FDS) @ Corth`. The overall sign is immaterial, but the basis
  metric is not. Equal history size 8 does not mean equal extrapolation weights.
  This is a verified implementation difference, not a measured attribution of
  a particular number of extra iterations.

## Why the final native steps are not an arbitrary stopping delay

The qualified WB97M-V candidate already uses energy change below 1e-9 Eh and
the physical MO-gradient norm below sqrt(1e-9), as GPU4PySCF's ordinary loop
does. Density change is not an extra candidate stopping condition here.

The existing Slurm16019 trace resolves the tail without guessing its norms:

- Step 16: energy change about 1.68545e-10 passes, but
  `nonlocal_confirmation_orbital_blocks=1` proves the MO-gradient gate fails.
- Step 17: the MO gate passes and the stationary-density hold is selected, but
  energy change 1.21574256e-9 still exceeds the unchanged 1e-9 threshold.
- Step 18: unchanged-density energy confirmation passes; the first final-state
  probe is accepted. There is no earlier rejected canonical probe causing the
  18-step count in this trace.

The exact per-step MO-gradient values were not retained. The predicate receipts
prove pass/fail only; do not invent scalar values or replace them with AO RMS.

## Proposed next experiment and invariants

Prioritize solver work rather than another packet-width search. A bounded actual
A100 control can isolate the cold guess first, then the orthonormal DIIS metric
or provisional handoff policy separately. Preserve the retained default as the
control, ordinary full-target acceptance, force gates, finite budgets and the
RTX5090 no-regression requirement. No default change is justified by this
diagnosis alone. Do not simply combine old DIIS entries across different
operators, relax convergence, erase the final confirmation, or turn Direct into
DF to obtain a smaller printed step count.

## Sources and reproduction

- `benchmarks/results/direct-range-warm-anchor-20261105/publication.json.xz`:
  actual A100 native/reference settings, 18/10 iterations and 11 reference Veff
  builds; SHA256 `6659c9962ca035cc6c0b5411cf54c6f1f64409a922eda2d00aff29726b65880f`.
- `benchmarks/results/vv10-packet-fp64-fma-20261106/diagnostic.json.xz`:
  complete native history, actual Slurm16019 trace and seed receipt. Its timing
  is intrusive; DF_TRACE bypasses optional MD-J and is not clean default timing.
- Native `src/dft/cuda_ks.cpp`, `src/dft/ks_stopping_policy.hpp`,
  `src/dft/cuda_ks_kernels.cu`, and `src/scf/cuda/scf_diis_kernels.cu`.
  These sources match verified master `318ae3b99f7af1e7fdc83d33dc7acd9da2917efb`.
- Official GPU4PySCF v1.9.0 `gpu4pyscf/scf/hf.py` and `scf/diis.py` match the
  measured reference module hashes exactly: respectively
  `c4caa754089e22927377408693602bd3d65574597f9817079ebb1e9e7a19f8c6` and
  `371f68ec9ea45cc671a7360217ea41a111cb20d9eae934c613768fe317e00d35`.
- Ignored task-local parser and receipts:
  `.artifacts/scf-iterations-20261106/analyze.py` and `analysis.json`.
- External memory: #2976. Evidence/rationale PR: #3018. No scientific production
  code, default, precision, functional or grid changes accompany this diagnosis.
