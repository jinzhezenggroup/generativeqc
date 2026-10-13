# Proposal: bind the existing generated CUDA stationary owner natively

Status: proposed
Date: 2026-10-12

The static binding and private complete-source controller portion is now
[implemented](../implemented/architecture/2026-10-12-native-cuda-stationary-source-controller.md).
The public cutover and remaining acceptance gates below are still proposed;
this historical proposal is not a current public capability declaration.

## Problem

The resident final D/W lease removes a detached-frame prerequisite from the
uncached Direct integral consumer. It does not make the complete force native.
`python/generativeqc/_stationary_cuda.py` still owns provider selection, grid
task iteration, generated geometry execution and final source composition.
Its native integral branch receives compact host arrays and adds them after
`sources.reduced()`. Moving that loop into C++ while retaining the host addition
would not establish the all-device production contract requested by #2227 R1.

## Existing owners to reuse

- `CudaKsPlan::prepare_resident_final_stationary_weights` authenticates the
  physical frame and stages total D/W without a dense export. Its companion
  resident-density binding supplies the separate spin densities and stream.
- `src/dft/cuda_grid.cu` already owns native grid topology, resident density
  binding, deferred device-point tasks and the `GridTaskView` lifetime.
- `emit_stationary_profile_aot_wrapper_cuda` emits the existing generated
  XC/AO/Becke and nuclear contractions around
  `src/dft/stationary_gradient_cuda.cuh`. It does not require the monolithic
  primitive inventory to be regenerated per profile. The packaged profiles
  already separate reusable primitive objects from method/spin wrappers.
- `stationary_geometry_reset` resets the geometry/source inventory without
  requiring detached host D/W. The resident-weight geometry v2 entry receives
  device weights/raw measures and a generation-bound task, including the
  paired-feature witness. That witness must come from the actual producer.
- The prepared Direct one-electron consumer already borrows normalized
  resident shell metadata and force scratch. Full-range generated J/K
  derivatives use the retained Direct topology and scientific coefficients.

## Proposed binding and source composition

Prefer a build-time typed binding to the same generated profile rather than
an independent derivative implementation. Selection must retain the exact
generated plan identity, spin, source inventory, quadrature contract and
immutable method composition. The initial native admission remains narrow;
unqualified domains retain their existing explicit Python compatibility path.

Static composition needs deliberate symbol isolation. Existing wrappers expose
the same `stationary_*` C names, `generativeqc_stationary_cuda` namespace and
global point/pullback helper names. Merely linking several wrapper objects
into one native library is not a demonstrated safe binding. Namespace/export
adaptation must change ownership only, not the emitted mathematical programs,
coefficients, precision or derivative conventions. Keep the current packaged
shared-library ABI separately compatible.

An alternative authenticated AOT loader must reuse the existing exact-plan,
compile-contract and binary-identity model. An unauthenticated `dlopen` plus
symbol lookup is not equivalent to the current Python artifact admission.
No reusable core loader has been established in this investigation; do not
invent another artifact/physics framework to bypass the missing binding.

The terminal composition must use the generated source reduction itself.
Factor resident enqueue/consumption from the existing one-electron and
full-range Direct derivative execution paths so the native stationary owner
can receive their compact device sources before they are overwritten. The
one-electron and J/K consumers may reuse the same provider force scratch;
their outputs therefore cannot be retained independently without an explicit
copy/lifetime/stream dependency. The current J/K host adapter negates native
force outputs to publish energy derivatives. Preserve that convention in any
device transport rather than silently changing the final force sign.

Source slot numbers, required-source completeness and reduction coefficients
come from the compiler-owned stationary plan. Do not hardcode a second PBE
or PBE0 force equation or add a host NumPy/C++ numerical sum. Authenticate
the final token before work and again before atomic force publication; a
failed source must not leave a partially published per-item force. Preparation,
scratch reservations, retained capacity and exceptional drains stay owned by
the existing prepared calculation and resource lifecycle.

## Acceptance before production admission

Require real-device independent force/final-D/W acceptance, source-specific
comparisons and finite differences where appropriate. Cover cold, warm,
genuinely moved geometry, empty beta, stale/capture rejection, rollback,
ragged failed neighbors and exact/below resource limits. Report complete
endpoint work/transport/live capacity, not just the resident proof packets.
Qualify the installed native SDK with Python and the compiler/toolchain absent,
and check the complete default endpoint regression gate. Neither this proposal
nor the resident lease authorizes a CUDA force capability bit or completion of
#2227 R1–R6.

## References

- [Resident final-weight decision](../implemented/architecture/2026-10-12-cuda-ks-resident-final-weight-lease.md).
- `src/scf/cuda/direct_coulomb.cpp::execute_generated_full_range_energy_derivatives`.
- `src/scf/cuda/one_electron_gradient_bridge.cu::execute_prepared_cuda_stationary_one_electron_pair`.
- `python/generativeqc_compiler/method/stationary_cuda.py::stationary_runtime_sources`.
- `src/dft/stationary_gradient_cuda.cuh::stationary_finish_reduced`.
