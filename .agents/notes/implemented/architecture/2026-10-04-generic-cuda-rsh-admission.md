# Decision: Admit CUDA RSH through prepared provider capabilities

Status: implemented
Date: 2026-10-04

## Problem and decision

The public KS adapter rejected CUDA range-separated exchange unless the complete
composition was WB97M-V, although the common CUDA KS owner already supported
generic exact primary/range providers. PR #1863 removes that stale restriction
and exempts RSH from the separate global-hybrid composition gate. Admission now
checks the prepared CUDA Direct binding, exact full-range Coulomb and optional
primary exchange, exact long-range correction, matching spin and screening, and
finite positive omega. Upstream semantic validation owns finite exchange
coefficients, the RKS/UKS Fock factors, and a shared SR/LR omega.

This changes public admission, not integral mathematics, solver ownership or
scientific precision. The existing composition is full-range primary exchange
plus the long-range correction with coefficient proportional to `a_LR - a_SR`.
Neither a carrier method ID nor an alias supplies a provider capability.

## Retained boundaries

- The public range semilocal lowerer remains PBE or complete WB97M-V. The CPU
  generic reference calls `run_pbe_rsh_rks/uks`; admitting B3/CAM or generated
  point programs without their corresponding lowerers would change the model.
- RSH density fitting remains rejected. WB97M-V still requires its exact
  B97M/SR/LR/VV10 composition and device-fused MolecularV1 nonlocal execution.
  Generic PBE RSH does not gain CUDA nonlocal/VV10 admission.
- Existing ECP restrictions remain at their consumers: WB97M-V ECP, fitted ECP,
  and CUDA stationary integral derivatives with ECP remain unavailable. RSH
  admission is not a claim that these consumers gained ECP support.
- Strict FP64 and AUTO retain the existing component precision schedule. AUTO
  may lower supported Direct J and density contractions; exchange, XC algebra
  and final audits remain strict. Fitted/nonlocal AUTO is still rejected.
- PBE component scaling uses the existing device-fused implementation;
  host-unfused execution rejects nonunity scales. Range correction continues to
  exclude CUDA Graph replay even when semilocal replay is available.
- Zero SR exchange makes the public primary J-only. CUDA energy execution falls
  back from fused RSH values to primary J plus the LR correction. Existing CPU
  RSH and CUDA RSH derivative consumers require a present primary K and reject
  this case. This change does not qualify those zero-SR endpoints. Equal SR/LR
  fractions retain a present, zero-coefficient correction.
- Warm state is density only across geometry changes. Providers and grids are
  rebuilt, and final-state identity includes primary/range strategies, spin,
  grid, scales, owner and solve generation. A new attempt invalidates old tokens.

## Evidence and limits

Review of `e256cd1f81da1a71d67287c5707c0703d1654db0` ran 79 host tests:

```text
tests/python/test_native_semilocal_admission.py
tests/python/test_ks_execution_plan.py
tests/python/test_stationary_rsh_cuda.py::test_native_cuda_rsh_integral_bridge_is_method_neutral
tests/python/test_stationary_gradient_plan.py
tests/python/test_stationary_composite_cuda_routing.py
tests/python/test_ks_component_precision_census.py
```

They ran with CUDA hidden, one CPU core and one BLAS/OpenMP thread. C++20
syntax-only checks of `tests/native/test_dft_api.cpp` passed with
`GENERATIVEQC_HAS_CUDA=0` and `=1`; these were not native test execution.
`git diff --check` also passed.

The PR adds public PBE-RSH RKS/UKS CPU/CUDA comparisons through both LDA and PBE
carrier IDs, warm replay, RSH-DF rejection, and mismatched-omega rejection. It
extends the existing lower-level CUDA KS cases with changed-geometry warm/cold
and CPU comparisons. These native numerical cases were inspected but were not
executed in this bounded review. The head's successful CuMetal workflow run
`37208821626` ran `generativeqc_cuda_runtime_tests`; its full-suite step was
skipped and does not establish those added endpoint results. No new GPU,
full-build, chemistry or timing campaign was run.

Issue #1849 requests force acceptance only where the primitive inventory is
already covered. This admission fix preserves that qualification scope; it does
not certify all generic RSH forces, zero-SR derivatives, or new ECP/precision
combinations. Existing provider evidence remains distinct from complete endpoint
qualification. No new performance claim is made.

## Revisit when

Extend a lowerer or precision/derivative domain only with its own independent
numerical and state-lifecycle evidence. Do not restore method-name admission to
compensate for a missing provider or semilocal capability.

## References

- [Issue #1849](https://github.com/jinzhezenggroup/generativeqc/issues/1849)
- [PR #1863](https://github.com/jinzhezenggroup/generativeqc/pull/1863)
- [CuMetal run 37208821626](https://github.com/jinzhezenggroup/generativeqc/actions/runs/37208821626)
- [CUDA RSH derivative provider rationale](../numerics/2026-09-26-cuda-rsh-stationary-derivative-provider.md)
- `src/methods/dft_method.cpp`, `src/dft/cuda_ks.cpp`,
  `src/scf/cuda_fock_execution.cpp`, `src/scf/fock_build.cpp`
