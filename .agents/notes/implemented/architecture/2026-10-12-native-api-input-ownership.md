# Decision: Resolve public auxiliary handles at the C API boundary

Status: implemented
Date: 2026-10-12

## Decision

Single and batch C API preparation resolve the optional opaque auxiliary-system
handle to a borrowed native `core::System`. The registry forwards that native
input synchronously. HF, DFT, RI-MP2 and DF-RCCSD(T) copy it into their prepared
owners before returning; replay must never dereference caller-owned handles.
Methods still use the public descriptor and status vocabulary. This is an opaque
handle-layout boundary, not removal of every dependency on the public header.

Core backend and basis tags use signed 32-bit native constants. The API ingress
checks constant parity with the published ABI. Headers that genuinely return
public status codes include the public header explicitly, including the ECP CUDA
interface used by both real CUDA and CPU stub implementations. The GFN2 orbital
bridge uses the native spherical-basis tag.

## Invariants and evidence

- Public ABI layouts and method IDs are unchanged; tags have matching types and
  values, checked at compilation.
- The compiled C API test exercises distinct explicit and implicit auxiliaries,
  real single HF/DFT/RI-MP2 and HF/DFT batch preparation, mutation and destruction
  of both input systems, repeated execution, and batch geometry rebinding.
- Null inputs, short descriptors, and mismatched auxiliary geometry still fail
  closed. The test does not invoke undefined behavior with fabricated handles.
- The former source-sliced auxiliary-reader fixture was replaced by these real
  owner tests. Its unrelated grid-option test remains and uses compiler caching.
- CPU compilation found two transitive-header dependencies (ECP status and GFN2
  spherical tags); explicit dependencies/native tags repair them. CUDA execution
  requires its ordinary hardware-qualified CI; CPU validation is not GPU evidence.

## References

PR #2383 and `tests/native/test_native_api_inputs.cpp`.
