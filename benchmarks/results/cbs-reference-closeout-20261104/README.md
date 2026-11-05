# Restricted CBS reference matrix and representative H100 endpoints

This source-bound numerical publication completes the original two-fixture,
three-model CUDA error matrix of #2681 and supplies the representative physical
water endpoint requested in #2998. It does not qualify arbitrary molecules,
open-shell CBS, later source revisions, default vendor settings, cross-GPU
speedup or accuracy against the true CBS limit.

Measured CUDA source: `6707989cce434b0ca4ef43f9920d5f91525f30c1` (clean Git LF
snapshot). Native identity:
`233483379c13c97f992bba0399025e9d0cfcaaac8aacba4989a8d398f4b8e411`.
Core SHA-256:
`3bce06074b2cdff13d12a6231fe9b9b4948740527d406b925c931ddb982d3691`.
The publishing checkout is a different revision. Replay checks all 2,497 frozen
runtime inputs against their measured Git objects, including native inventory
membership and the C ABI source identity.

## Population, references and fixed gates

The six CUDA combinations are physical spherical H2 cc-pVDZ/cc-pVTZ and the
synthetic water-algebra fixture, each with conventional, correlation-only DF
(exact RHF), and full RI-JK models. Each combination returns one complete
energy-plus-analytic-force call, one energy-only call and four reconverged
displaced energy calls at directional steps 3e-4 and 1e-4 Bohr. The five
constituents and composite are retained in full. Energy and force/directional
gates remain 1e-9 Eh and 2e-6 Eh/Bohr; energy-only parity remains 1e-10 Eh.
Convergence, analytic-response provenance, translation, triples, composition,
Hamiltonian, fitting, basis, spin and returned CUDA backend are checked.

The conventional model uses the unchanged committed PySCF reference fixtures.
Four independent fitted-model oracle runs reconverge fresh PySCF 2.14.0
calculations at every Cartesian displacement, at 1e-3 and 3e-4 Bohr: 124 center
and displaced observations in total. The original fixture files are unchanged.
These matching-model reference observations are separate from native processes;
the native runner rejects imported PySCF/GPU4PySCF modules.

Physical water separately uses full DF with matching cc-pVDZ/cc-pVTZ JKFIT
tables, a nonzero (T) contribution, and two complete energy/analytic-force calls
in one process. Its reference is the exact retained water member of
`cbs-routine-standard-20261025/records.json.xz`, bound by parent and extracted
hashes. Both complete Cartesian finite-difference steps (1e-4/3e-5 Bohr), the
original 1e-7 Eh / 3e-6 Eh/Bohr maximum / 1e-6 Eh/Bohr composite RMS gates,
and the 0.01-Eh constituent locality condition are preserved. The synthetic
water-algebra fixture is not substituted for this physical-water calculation.

`errors.csv` gives the recomputed energy, maximum-force and RMS-force errors;
`summary.json` includes all gates and constituent/composite comparisons. Timing
fields describe complete singlepoint calls with prebuilt code and exclude
process startup, imports, calculator construction and compilation. The physical
water repeat reuses its calculator; these two calls are not independent timing
replicates. No comparator or performance acceptance is registered.

## Backend and environment boundary

| Model | CPU | CUDA on the measured H100 |
| --- | --- | --- |
| Conventional | Historical frozen source `bbd41a4` accepted; reused | Six-row matrix includes both fixtures |
| Correlation-only DF | Explicitly unsupported by frozen public contract | Both fixtures under explicit workspace profile |
| Full RI-JK | Explicitly unsupported by frozen public contract | Both fixtures, plus physical water |

The historical conventional CPU publication is replayed and compared exactly;
its 13 native tests, 65 unit tests and six capacity-validator tests are not
represented as new-source execution. Four fresh CPU DF admissions return the
exact documented `NotImplementedError` before any scientific call. They are
UNSUPPORTED, not scientific PASS. The custom CUDA observations reproduce the
original scientific gates; the original CUDA pytest suite was not run here.

The accepted CUDA attempt explicitly sets `CUBLAS_WORKSPACE_CONFIG=:16:8` in
each fresh child and records its actual presence/value before the scientific
imports. All runs use one math thread, tight all-FP64 controls and a 32-GiB
correlation budget. NVIDIA documents this workspace profile as a debug
reproducibility setting that may limit performance:
<https://docs.nvidia.com/cuda/archive/12.9.2/cublas/index.html#results-reproducibility>.
It changes neither the 192-MiB provider guard nor production defaults.

The earlier V2 attempt is retained and independently replays as
FAIL_OR_INCOMPLETE: four CUDA combinations and physical water passed, while H2
conventional and correlation-only DF returned no energy/force because their
observed provider-retained deltas were 371,195,904 bytes (354 MiB), above the
unchanged 201,326,592-byte allowance (192 MiB). V2 did not record workspace
presence/value; that environment field remains UNKNOWN. Do not relabel those
refusals or treat the accepted explicit profile as default-environment success.

Actual device name, UUID, 81,559-MiB capacity, driver 595.58.03, platform status,
build/cache commands, terminal receipts, hashes, process outcomes and complete
raw outputs are retained. The science jobs have separate authenticated
preflight-release bindings and finite platform/child deadlines. Cleanup controls
cover leader-exited, TERM-ignoring and detached descendants. Prior failed and
superseded predispatch audits remain history; final numerical acceptance is
separate from readiness, platform success, PR/CI and merge.

## Offline replay and recovery

Use Python 3.11 or newer, NumPy and Git, with the measured commit available in
the local object database (a shallow checkout may need history first):
run without `-O`/`-OO` or `PYTHONOPTIMIZE`, as the retained CPU reader requires
non-optimized Python.

```text
python benchmarks/results/cbs-reference-closeout-20261104/replay.py
python benchmarks/results/cbs-reference-closeout-20261104/replay.py --recover /new/empty/directory
```

Replay verifies the publication and retained-member hashes, reconstructs only
the needed measured reference inputs from Git, recomputes both the failed V2
attempt and accepted explicit-profile attempt, and checks the exact published
summary and CSV. It runs no CUDA/native calculation and downloads nothing.
Recovery refuses an existing destination, checks paths and preserves raw bytes
and ordinary POSIX modes. Original absolute execution paths are provenance,
not commands to execute blindly. Reviewer verdicts remain same-family/provisional.
