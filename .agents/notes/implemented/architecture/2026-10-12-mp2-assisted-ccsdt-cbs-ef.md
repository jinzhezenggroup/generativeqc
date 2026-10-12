# Decision: native MP2-assisted CCSD(T)/CBS energy and analytic forces

Status: implemented
Date: 2026-10-12

Execution/ownership policy superseded by
[native same-reference component groups](2026-10-12-cbs-native-reference-groups.md).

## Problem

The GPU MP2 E+F source-reuse optimization (PR #2352) enables efficient finite-
basis MP2 components but does not itself define or execute a CBS model. The
requested endpoint is both energy and force: extrapolating total MP2 energy,
dropping the small-basis correction force, or returning only a large-basis
MP2 force would not differentiate the requested composite energy.

## Decision

Add `generativeqc.cbs.CCSDTCBSCalculator`, an orchestration-only public facade
over the existing native `rhf`, `mp2` and `ccsd(t)` endpoints. The focal-point
model is HF(limit) + MP2-correlation(limit) + [CCSD(T)−MP2](small). Fixed
cardinal/exponent inputs define a geometry-independent linear combination,
used identically for energy and analytic force. Default correlation power is
3; default HF is the fixed-alpha exponential model with alpha 1.63. An explicit
highest-basis HF option does not pretend to extrapolate HF.

The input records carry full basis identities and explicit cardinal numbers.
Same method/basis identities execute once per call, including when the small
correction basis coincides with either extrapolation basis. Production never
queries a basis service or imports a reference package. Local arbitrary records
can be used for qualification, but declared cardinals do not certify a
systematic extrapolation family or CBS error.

The supported policy is FP64, exact integrals, all-electron canonical RHF,
full occupied-space correlation and one closed-shell singlet. Common charge,
geometry, AO representation and solver/resource controls apply to all terms.
Do not mix frozen-core CC with all-electron MP2 or fitted and exact Hamiltonians
without an explicit scientifically qualified extension. Native component
domain/stationarity/response gates retain authority; composition broadens none.

## Ownership and work

Execute native components sequentially and call `clear_cache` in `finally`
before admitting the next native owner. Returned native result storage remains
owned independently. This avoids simultaneously retained multi-basis CUDA
executables and guarantees cleanup after subset failure. Publication occurs
only after every component and the final finite/shape/backend checks complete.
One workflow serializes its calls.

Separate HF E+F endpoints are retained deliberately: MP2 currently publishes
total, not separate HF/correlation, forces. Deriving a correlation force by
subtracting an independently converged HF force preserves the explicit energy
model and its derivative. Avoid presenting hypothetical shared-reference force
storage as an existing production optimization. Component budgets are native
endpoint budgets, not a measured full-workflow device peak or process-RSS cap.
Python basis/result storage remains outside those native accounting contracts.

The implementation gives up retained prepared-plan warm reuse to bound owner
lifetime. Artifact/library caches still replay, but benchmarks call these
samples **complete replay endpoints**, not prepared-warm endpoints. Each result
exposes its actual native components, their combined coefficients, contribution
energies/forces, diagnostics and timing.

## Rejected alternatives

- Direct large-basis CCSD(T) extrapolation: defeats the requested MP2-assisted
  work reduction and inherits much more expensive high-order correlation work.
- Extrapolating MP2 total energy with one correlation power: treats the HF
  convergence model incorrectly.
- Geometry-dependent exponent fitting without response: changes the derivative.
- Production PySCF or finite-difference forces: violates native E+F ownership.
- Keeping every prepared GPU plan resident without aggregate admission: several
  independent per-method budgets are not one complete-workflow memory bound.
- Enabling CUDA RI-MP2 forces by composition: that native derivative boundary
  remains explicitly unsupported.

## Evidence

Independent offline PySCF 2.14.0 records include every constituent E+F and
full CCSD(T) triples Lambda. Their composite gradients are checked against
freshly reconverged displaced energies at steps 1e-3 and 3e-4 Bohr. H2 uses
the physical spherical cc-pVDZ/cc-pVTZ pair with the DZ correction. Water uses
an explicitly **synthetic** pair to stress the composition with nonzero triples
(about -6.7314e-5 Eh); it is not predictive water CBS accuracy evidence.

Public gates are 1e-9 Eh for energies and 2e-6 Eh/Bohr for component/composite
forces and directional derivatives. Production-native finite differences also
reconverge all constituent energies at steps 3e-4 and 1e-4 Bohr. CPU acceptance
checks water algebra E+F and physical H2 CBS energy; physical H2 CPU force replay
is an explicit `GENERATIVEQC_CBS_LARGE_TEST=1` qualification because the existing
conventional CPU replay is expensive. The allocated GPU suite exercises both
complete E+F cases and bounded-refusal recovery. Routine CPU acceptance returned
30 passed / 10 skipped in 390.25 s before six additional mocked failure gates.
Public API/contract/documentation checks returned 94 passed / 1 skipped and
39 subtests before those additional cases. Neither interrupted large CPU replay
nor an unsupported CUDA path is counted as successful qualification.
The final allocated CUDA suite returned 7 passed / 8 deselected in 96.51 s;
the focused composition/failure suite returned 31 passed. Public docstrings and
contracts, Ruff and `git diff --check` pass; Sphinx HTML builds with `-W` and
`--keep-going` without warnings.

Complete source-matched GPU endpoint evidence, Slurm job 7841 on node1/main,
one RTX 5090, CUDA visibility preserved, 15-minute finite allocation:

| Case | Cold E+F (s) | Complete replay E+F (s) | Maximum E error (Eh) | Maximum F error (Eh/Bohr) |
| --- | ---: | ---: | ---: | ---: |
| H2 cc-pVDZ/cc-pVTZ + DZ correction | 75.480466 | 74.006895 | 3.62e-13 | 5.76e-10 |
| Water synthetic-pair algebra + STO-3G correction | 15.093604 | 13.788085 | 2.70e-13 | 1.75e-7 |

These are one cold and one replay sample per case, **not** medians over several
independent repetitions or speedups against a preexisting CBS implementation.
Every measured component and composite passes the independent gates. The
endpoint includes all five native component constructions, SCF, correlation,
analytic forces and owner cleanup; reference comparison and work tracing are
outside the timed samples. This is not realistic-molecule TZ/QZ qualification.

H2's high-basis MP2 component accounts for 72.45 s of the 74.01 s replay.
The untimed completed **MP2** work census reports 1,399 device/source reads,
709,194,400 source values, 9,952,548,636 contraction summands and 60,331,160
D2H bytes across the two MP2 E+F components; no failed scopes occur. Water's two
MP2 E+F components report 1,154 reads and 9,390,430 contraction summands. The
trace counts are not mislabeled as the whole CC/HF integral-work census:
native CC performance/response records are retained separately in the artifact.

Evidence is retained under ignored `.artifacts/ccsdt-cbs-20261012/`, including
endpoint JSON/logs, CPU/GPU tests, independent fixtures and source/library
provenance. The native CUDA library is the previously qualified MP2 candidate
SHA256 `95f5e5531174813261f7830b5c1073c677fe086c45ffb66f812a9f5fd2be6d6b`;
native equations are unchanged by this orchestration. Endpoint JSON records
the exact CBS Python source hashes alongside the frozen native library hash.

## Revisit when

- Native MP2 exposes HF/correlation-resolved forces or a method-neutral shared
  reference owner: eliminate duplicated HF work without weakening lifetime and
  complete-workflow budget admission.
- Native CUDA RI-MP2 derivatives or a common frozen-core active-space contract
  are independently qualified: extend every term together, never just CC.
- A combined multi-basis prepared owner can admit all simultaneous residency:
  add prepared batches without pretending independent budgets compose by default.
- Larger physical TZ/QZ cases are affordable: qualify practical scaling rather
  than generalizing H2 or synthetic-pair timings. The existing response/replay
  work, not this scalar combination, is the remaining dominant cost.

## References

- [Current public workflow contract](../../../../docs/user/ccsdt_cbs.md)
- [MP2 source E+F rationale](../performance/2026-10-12-mp2-cuda-reference-source-ef.md)
- `tools/generate_ccsdt_cbs_reference.py`
- `benchmarks/ccsdt_cbs_endpoint.py`
- `tests/python/test_ccsdt_cbs.py`, `tests/python/test_ccsdt_cbs_native.py`
