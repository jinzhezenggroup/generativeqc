# Decision: a separate byte ABI for bounded CPU Boolean programs

Status: implemented
Date: 2026-10-13

## Problem

The materializing TensorIR CPU owner packed every live value into `double`.
Admitting Boolean data through that ABI would change its representation into a
floating mask and invalidate both typed resource accounting and publication.
The [reference contract](2026-10-10-boolean-tensorir-reference-contract.md)
established Boolean IR legality, but deliberately left native storage unqualified.

## Decision

Only `tensor.cpu.NativeTensorProgram` opts into Boolean production preparation.
Graphs with live Boolean data use a distinct unsigned-byte input/output ABI:
the size arguments count bytes, FP64 fields have eight-byte alignment, Boolean
fields occupy one byte, and both external buffers have double alignment.
Generated offsets and dtype determine each field; there is no caller-controlled
dtype reinterpretation. `memcpy` loads/stores avoid double object-lifetime and
aliasing assumptions in the byte arena. Native input bytes for Boolean fields
must be exactly zero or one. Six comparisons accept finite FP64 operands and
produce canonical Boolean bytes. Existing Boolean transpose, reshape, slice,
gather and broadcast use this same storage contract.

Byte budgets include aligned arena/output storage and conservative adapter
staging/snapshots. Work and node limits retain their existing semantic accounting.
Evaluation writes detached outputs at the arena tail; it publishes only after
every live value passes its dtype/domain check. Invalid sizes, budgets, alignment,
null pointers or overlapping external buffers reject before evaluation. Errors
do not publish partial results and do not invalidate later valid executions.

Pure FP64 programs retain the existing double-count ABI and arithmetic emitter.
Shared preparation rejects Boolean by default. The small `cpu_emit` owner,
`NativeTensorProgramBundle`, CUDA, portable and scalar owners remain closed to
Boolean data. Boolean AD and `where` are unchanged and unsupported. FP32 native
CPU comparisons and nonfinite comparison semantics are outside this slice.

## Rejected alternatives

- Encoding Boolean data as floating zero/one or existing int64 control maps:
  neither preserves the first-class Boolean data contract.
- Changing the existing FP64 ABI for all consumers: it would impose unnecessary
  scientific migration and bundle changes.
- Casting byte-vector storage to `double*`: alignment alone does not establish
  double object lifetime or valid aliasing.
- Removing the shared Boolean gate: unrelated emitters would inherit admission
  before their storage and native execution were qualified.
- Python/NumPy execution per node: the native runtime must execute generated C++;
  NumPy and array-api-strict serve only as independent test oracles.

## Evidence and remaining scope

`tests/python/test_tensor_cpu_boolean.py` executes public compiled CPU comparisons
against NumPy and array-api-strict for all six operations, signed zero, extreme
finite values, empty/scalar/broadcast shapes. It exercises Boolean input/constants,
typed views, mixed real outputs, serialization/replay, immutable detached results,
malformed Boolean storage, native nonfinite failure, alignment/alias/budget errors,
overflow and recovery. Existing CPU scientific and ABI tests remain required.
An opt-in one-time comparison against base `2c1e249382ca137182c1b5703010aeea2adc3295`
found identical emitted sources/resources for eight representative pure FP64
graphs; no permanent source-body fingerprint test was added.

This primitive qualification is not a QC endpoint performance claim. Parent
issues #2137/#2111 retain `where`, selected-real-branch AD, CUDA Boolean storage,
IEEE nonfinite comparisons and full Array API conformance as separate work.

## Revisit when

Another native owner has an independently executed typed ABI and explicit
resource/lifecycle qualification, or separately reviewed IEEE/AD semantics are
available. Preserve the old FP64 ABI unless all its consumers migrate explicitly.
