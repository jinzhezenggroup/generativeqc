# Decision: standard CPU eager arrays without broadening TensorIR qualification

Status: implemented
Date: 2026-10-11

The reference-only public execution boundary is superseded by
[native Array API and AD](2026-10-11-array-api-native-autodiff.md).
The eager conformance and no-implicit-device-transfer decisions remain intact.

Supersedes the eager discovery boundary in the
[public-preview decision](2026-10-07-experimental-array-api-public-preview.md),
while retaining its bounded symbolic/compiler invariants.

## Problem

The bounded compiler preview intentionally rejected integer/complex data,
non-finite values and many standard operations. Adding discovery to VibeArray or
forwarding missing operations to NumPy during capture would falsely claim
compiler conformance and weaken scientific/native admission.

## Decision

Provide an explicit NumPy-backed CPU eager Array API 2025.12 surface, using
array-api-compat as an execution adapter. Concrete ndarray storage views own
discovery and rank-zero indexing/scalar conversions. Standard public declarations
remain explicit for source-only API documentation and contract inventories.
Retain the original bounded public lowering helpers in a private symbolic
adapter; mathematical ownership stays in generativeqc_compiler.array_api/TensorIR.
Only supported capture operations dispatch there, including creation without
array operands. Unsupported calls never fall back to NumPy during tracing.

The dtype namespace uses NumPy scalar-type tokens, not dtype instances: NumPy's
float64 dtype instance compares equal to None, which breaks promotion-table
lookups in upstream conformance tests. Array storage dtypes remain native NumPy
dtypes. Adapt finfo fields to Python scalar values and implement standard IEEE
floor-division and complex acosh/expm1/tanh exceptional limits, while preserving NumPy's
finite numerical execution where applicable.

## Rejected alternatives

- Advertising VibeArray as conforming: native dtype/function/domain admission is
  incomplete, and symbolic control flow and scientific index domains differ.
- Delegating production calls to array-api-strict: it is an independent test
  reference, not the existing NumPy eager storage owner.
- Implementing a second scientific IR or generic CUDA executor: not required for
  CPU eager conformance and would bypass native scientific qualification.
- Broad wildcard exports or dynamic public inventories: source-only API
  ownership/docstring tooling must identify each standard public declaration.

## Invariants

- Eager CPU storage is explicit, not a hidden native/GPU production fallback.
- Symbolic inputs and active capture cannot reach eager implementations.
- Compilation still admits only bool/float32/float64 inputs and retains bounded
  host conversion, finite values, exact constants and scientific TensorSpecs.
- Foreign device protocols cannot trigger implicit conversion through functions,
  array operators or nested host containers; DLPack handoff is explicit.
- Copy=False wraps/views existing ownership without hidden allocation/transfer.
- Native SCF generation, scientific graph identity and compiler ownership do not
  depend on the new eager adapter.

## Evidence

The pinned suite/spec revisions and 100-example gate are recorded in
manifests/array_api_conformance.json and tools/check_array_api_conformance.py.
Local test_array_api_standard.py protects protocol, dtype, IEEE, aliasing,
foreign-device and fail-closed capture boundaries. Existing scientific tests
retain compiled finite-domain and autodiff acceptance gates.

The minimum-NumPy CI lane uses Python 3.11. A preliminary Python 3.10 attempt
failed before importing the array namespace because existing runtime batch and
dispersion modules import typing.Self (added in Python 3.11). That repository-wide
Python compatibility defect is outside this change; no test-time typing shim or
unrelated runtime patch is used to manufacture conformance evidence.

## Revisit when

An independently qualified native backend can implement the declared full
standard without implicit reference work, or TensorIR gains additional dtypes
and operations with independent scientific acceptance gates.
