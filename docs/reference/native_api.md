# Native C and C++ API

The stable native declarations live in
[`generativeqc.h`](https://github.com/jinzhezenggroup/generativeqc/blob/master/include/generativeqc/generativeqc.h).
The move-only C++ convenience wrappers are in
[`generativeqc.hpp`](https://github.com/jinzhezenggroup/generativeqc/blob/master/include/generativeqc/generativeqc.hpp).
Internal `src/` headers are not public API declarations.

## ABI and capabilities

Call `generativeqc_get_abi_version()` to discover the loaded ABI.
Initialize public versioned descriptors with their exact `struct_size` and
`abi_version = GENERATIVEQC_ABI_VERSION` before calling the relevant entry
point. Do not assume an incompatible descriptor layout will be accepted.

`generativeqc_method_available` and `generativeqc_method_get_capabilities`
expose registry information, not support for every basis/device/precision and
derivative combination. A prepared execution can apply stricter checks.
See [capability sources](capabilities.md).

## Core C handles and ownership

| Entry points | Contract |
| --- | --- |
| `generativeqc_context_create/destroy` | Create the execution context and release it after dependent owners |
| `generativeqc_system_create/destroy` | Create a system with atom, geometry and basis descriptors |
| `generativeqc_calculation_prepare/destroy` | Prepare and dispose of single-system method state |
| `generativeqc_calculation_execute` | Execute synchronously; callers provide the output descriptor and force buffer when requested |
| `generativeqc_batch_prepare/destroy` | Own and release a persistent ragged-fleet plan |
| `generativeqc_batch_execute` | Execute the fleet with per-input result statuses rather than assuming every member succeeded |
| `generativeqc_context_get_last_detail` | Borrow the last context-specific failure detail until the next failure or context destruction |

Keep the execution context valid for its dependent handles, and release each
handle with its matching `*_destroy` function. Never assume a function
allocates an output buffer merely because it accepts a pointer: consult its
buffer-size and null-pointer conventions. Unless a function explicitly
documents transactional behavior, do not rely on output contents after failure.

The high-level `generativeqc::Context`, `System`, `Calculation`, and
`Batch` C++ wrappers own the corresponding C handles using RAII, are movable
but not copyable, and require their dependent owners to remain alive.
`generativeqc::check` and wrapper methods throw `generativeqc::Error` for
failed native status calls. `Error::status()` preserves the status code.
`Calculation::execute` checks supported properties; omitted forces are
represented by `std::optional` rather than an all-zero vector.

## Units and failure semantics

Coordinates use **Bohr**, energies **Hartree**, and forces **Hartree/Bohr**
unless the selected function specifies a different convention; nuclear
gradients are the negative of forces. Matrix and vector layouts remain
specific to their C descriptors. See [units](units.md).

`GENERATIVEQC_STATUS_SUCCESS` means the native call succeeded.
Other status codes distinguish invalid arguments, ABI mismatch,
unsupported operations, non-convergence, numerical failure,
CUDA failure, out-of-memory and internal errors. The special
`GENERATIVEQC_STATUS_PRECISION_UNAVAILABLE` means a completed precision
provenance record is not available; it does not, by itself, mean a solve
failed. Use `generativeqc_status_message` for status explanations and
`generativeqc_context_get_last_detail` for contextual detail.

Batch execution may succeed structurally while individual member results fail.
Always inspect each input-indexed member's status and convergence fields before
using its numerical data; a missing force array is not a zero force.

For advanced Python/compiler extensions instead of C handles, consult
the [extension guide](../developer/extensions.md) and
[Python API](api.md).
