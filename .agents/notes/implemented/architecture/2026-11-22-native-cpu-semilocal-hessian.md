# Decision: bind bare CPU LDA/PBE to the linked scalar Hessian owner

Status: implemented
Date: 2026-11-22

## Decision

The default CPU Hessian catalog binds exact bare LDA/PBE RKS for Direct FP64
all-electron Cartesian AOs. The existing prepared-batch Hessian bridge owns
HVP and scalar dense execution for the installed SDK, CLI and Python. The
bindings use the physical SCF point evaluator and compiler-generated J-only
nuclear contractions. They do not run a zero-weight K program or introduce
another semilocal derivative implementation.

Explicit block or recycled Python dense schedules retain the existing bounded
owner, as do Python callers with a library that lacks the exact linked graph.
The [recycle follow-up](2026-11-22-native-cpu-recycled-hessian.md) supersedes
this part of the decision for linked recycled groups of two to four RHS.
The CLI never JIT-compiles a missing binding. Its explicit range-PBE MethodIR
is rejected before CPU method selection because the intermediate PBE selector
is a carrier, not the requested range-separated model. An admitted native
failure does not retry through Python.

The scalar native response uses the existing physical residual test with
`rtol=1e-10`, `atol=1e-12` by default. An explicit absolute tolerance is not
changed. The absolute floor matters for symmetry-zero columns: on aligned
STO-3G H₂, explicit `atol=0` can refuse a column with a roundoff-scale
absolute residual below `8e-17` but a large relative residual. The prior
recycled Python path can pass the same dense endpoint. Systems without any
occupied–virtual rotations remain outside both response implementations.

## Evidence

- On distorted STO-3G water and the same discrete Becke grid, independent
  PySCF reconverged-gradient differences give maximum raw Hessian errors of
  `1.315e-7` (LDA) and `1.420e-7` (PBE) Hartree/Bohr² at `3e-4` Bohr.
  Finer-step convergence, raw symmetry, translation nulls and dense/HVP
  consistency pass without post-hoc symmetrization or displaced SCFs in the
  linked endpoint.
- Installed C++ SDK and CLI run with no Python or compiler on `PATH` and
  reproduce the Python scalar outputs and accepted energy. The SDK checks
  stale geometry, transactional one-byte-below HVP refusal and moved replay.
  Python tests cover exact and one-byte-below composed host capacities for
  HVP and dense, including the live primal reservation and explicit reserve.
- An explicit B3LYP-only build passes the CLI contract checks for absent
  semilocal bindings. Explicit recycled Python compatibility remains distinct
  from native scalar admission.

On one AMD EPYC 7K62 with one BLAS/OpenMP thread, three fresh-process samples
per semilocal method/product/geometry compare the previous packaged fallback
with the fixed candidate library. The fallback's generated artifacts were
warmed before timing; both endpoints reconverged SCF and included preparation,
response, curvature and publication. Initial and moved cases all ran 10 SCF
iterations, 12 Fock builds and either 1 or 9 derivative directions. The largest
raw old/new output difference was `1.211e-12` Hartree/Bohr². Within this
cohort, HVP medians went from `32.36–32.86 s` to `0.260–0.346 s` and dense
medians from `94.72–99.55 s` to `2.758–3.675 s`. Absolute legacy times varied
with acquisition conditions; these are fixture timings, not a general speedup.

The candidate used 9 Krylov iterations and 20 induced Fock actions per HVP,
and 79 iterations and 176 actions per dense endpoint. The dense schedule
reuses integral results: both products counted `23,004/32,886` first/curvature
ERI primitive visits. Five paired samples per PBE0/B3LYP-D3(BJ) control kept
the raw results and work counters unchanged and met the 5% median
non-regression gate. The local binaries, input script, raw records and checker
are in the `semilocal-matched` receipt directory under
`/data/jzzeng/gqc-2227-sdk-JsefEq0x`.

## Limits

This is an unmerged CPU slice of #2227 based on `cae2959e1`, not new CUDA
evidence, an admission for other functional graphs, or completion of R1–R6.
Resource ceilings and exact graph identity still apply; large/heavy systems,
UKS, spherical LDA/PBE, ECPs, corrections and fitted integrals are not
qualified by this change.
