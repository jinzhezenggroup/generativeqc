# Decision: native CUDA stationary admission follows the generated graph

Status: implemented
Date: 2026-10-12

## Problem

The first native CUDA force checkpoint independently qualified a small fixture
inventory, but encoded that inventory twice as a PBE/PBE0 name whitelist: once
in native source generation and again in the C++ prepared admission rule. This
confused evidence coverage with scientific eligibility and required runtime
branches for additional graphs already represented by the stationary compiler.

The expanded endpoint gates also exposed two genuine cross-backend gaps. A
boolean LDA/GGA resource query cannot size tau-dependent XC panels. The old
stationary r2SCAN point wrapper differentiated an interior expression instead of
using SCF's component-wise Libxc work-domain policy, producing nonfinite force
partials at an exactly empty spin even though the same native SCF converged.

## Decision

Derive native bindings from the existing compiler AOT catalog and complete
stationary source coverage. CMake intersects that generated inventory with its
selected build profiles; it contains no separate native functional list.

Emit an exact semantic key from MethodIR and generated point metadata:
functional selector, point-domain version, spin convention, partition iterations,
semilocal X/C scales and full-range exchange energy weight. Prepared C++ admission
matches this key against actually linked bindings. Names are build/diagnostic
labels only. Convert the authenticated Fock coefficient back to its energy
weight, respecting restricted occupation two versus unrestricted spin density.
Changing coefficients, domain, spin or available providers never borrows a
different graph's binding. Equivalent renamed graphs retain the same admission.

Add point-aware allocation inventory v2 using the actual native XC allocator
layout. Preserve the existing v1 boolean ABI as an adapter. Python planning
uses v2 when available; older libraries retain non-tau planning but cannot
silently admit tau panels under a GGA bound.

Reuse `emit_r2scan_program` in the stationary Cartesian wrapper. This retains
SCF's component thresholds, work-input floors and original-density energy
rescaling, plus its existing total-density tail policy. Do not invent a second
regularization, differentiate clipping, or drop empty-spin fixtures. Physical
empty-spin tangents are checked independently against Libxc; all emitted
partials are checked against the actual native SCF policy. The minority rho/tau
extensions are not asserted to be an unconstrained endpoint Hessian.

CLI bulk identities hash the common/integral/XC source families, including
unimported modules. Track that hashed closure as generated-catalog dependencies;
import-only depfiles otherwise leave stale native identity rows after the point
consumer changes. This does not broaden unrelated kernel-generator dependencies.

## Retained boundaries

This is not universal functional support or a native response/HVP cutover. The
compiled inventory is finite and optional. Complete providers, all-electron s/p
bases, exact full-range Direct J/K, strict FP64, authenticated final state,
generation/lifetime validity and bounded admission remain prerequisites. Missing
bindings or unsupported source inventories retain the existing compatibility
consumer; no production oracle, runtime compiler discovery or partial native
publication is introduced.

## Rejected alternatives

- Extend the handwritten whitelist with more names: recreates the same ownership
  error and mishandles renamed or scaled graphs.
- Admit any graph sharing a point selector: loses X/C and exact-exchange identity.
- Remove the tau rejection without a point-aware layout: undercounts simultaneously
  retained device work/feature panels.
- Replace the failing empty-spin fixture or add a functional-specific exclusion:
  conceals the mismatch between the force and SCF point contracts.

## Evidence and reproduction

- `tests/python/test_native_stationary_cuda_binding.py`: executable generated
  registry subset/empty inventory and semantic-key mutation rejection.
- `tests/python/test_native_cuda_force_resources.py` and
  `tests/python/test_ks_point_batch_resources.py`: legacy ABI parity, tau panel
  charging, old-library rejection and full per-item fleet admission.
- `tests/python/test_functional_independent_xc_optimization.py`: independent
  Libxc physical point outputs and complete SCF-policy parity at empty spin.
- `tests/python/test_public_native_cuda_forces.py`: complete independent
  energy/force/D/W fixtures, renamed graphs, minimum-budget cold/warm geometry
  replay, rollback and injected allocation/driver failure.
- Run real-device tests and memcheck only through finite Slurm `srun` jobs,
  preserving assigned `CUDA_VISIBLE_DEVICES`. Local source-matched receipts are
  retained under `.artifacts/2227/cuda-model-admission/`.

## References

This supersedes only the name-bound admission portion of
[the native source-controller decision](2026-10-12-native-cuda-stationary-source-controller.md)
and [the fleet-admission decision](2026-10-12-native-cuda-force-fleet-admission.md).
Their lifecycle, provider ownership, reduction and fleet-reservation contracts
remain in force.
