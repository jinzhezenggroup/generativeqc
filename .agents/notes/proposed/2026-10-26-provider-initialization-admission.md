# Decision candidate: explicit native provider-initialization admission

Status: proposed; native HOST candidate, public request transport and device acceptance outstanding
Date: 2026-10-26

## Problem

The physical-reference guard's existing 96 MiB per enabled provider allowance
can reject a real provider initialization. Reporting the measured byte counts
does not establish a replacement allowance or a bound on opaque library peaks.
Growing this default from a GPU product name, free-memory snapshot or remaining
numeric budget would make an unproved capacity promise.

## Candidate

Keep absent/legacy requests on their existing policy, without an additional
runtime identity query. A native-only explicit request supplies a finite positive
allowance and an expected identity. The ordinary reference owner validates the
request before its existing numeric preflight and binds a fresh native observation
after its current context is established and before vendor handle creation.

The observation uses the selected SDK declarations for optional driver lookups,
the current device UUID, driver/runtime versions and `cuCtxGetId`. A context pointer
only transports the query; it is not a generation. The context ID has CUDA's
program-lifetime scope and must not become a portable qualification certificate.
No context is created, retained, selected or reset by the observer itself.
Identity DTOs do not include ELF or CUDA-driver implementation headers; ordinary
CPU/legacy option consumers acquire no loader implementation dependency.

Direct runtime, BLAS, solver and driver build identities come from bounded mapped
GNU build-ID notes in already loaded ELF objects. The requested symbol must belong
to the pinned object; the runtime/BLAS/solver execution addresses must agree with
that binding. Foreign symbols, dependencies, loader namespaces, missing notes and
malformed bounds refuse the explicit request. A build-ID identifies a linked
build, not the object's current file checksum or all transitive dependencies.

The loader query is deliberately preloaded-only. A provider-free wheel trampoline
or an absent library is unavailable instead of being inferred from its SONAME,
path or headers. Unsupported platforms/providers also keep the legacy path and
refuse this explicit capability. A public loader/transport solution is not implied.

Actual CUDA/driver query failures preserve their domain and original status.
They do not become an ordinary unsupported-provider fallback. Absent capability
and unformed identity remain distinct from an executed query's failure.

## Budget and lifetime

The selected allowance feeds the existing reference base, workspace and optional
residency checks. No second numerical sizing formula is introduced. For an explicit
owner, retained admission subtracts the selected allowance exactly once; the
observed free-memory delta remains a separate diagnostic. Absent/legacy retained
accounting is unchanged. The request is copied into native options and participates
in reuse identity; another context, build, device, family or allowance cannot reuse
the previous request. Loaded objects stay pinned through provider/binding teardown.

The selected-inventory planner already excludes runtime-allowance rows from its
numeric ledger cap. Adding another provider lease to that cap would double
withhold capacity. This candidate introduces no voucher or ledger algorithm change:
an explicit request with an active ledger has an unknown accounting basis and is
refused before vendor creation. Existing legacy ledger behavior is retained.

The existing scalar reference sub-budget and downstream retained-source contract
remain the authorities for total admission. No allowance is advertised as a causal
allocation ledger, physical peak, full library workspace, successful provider
admission or scientific result. Existing teardown remains the owner boundary; this
HOST candidate does not establish real CUDA destroy success or physical pool release.

## HOST evidence and missing gates

Executable tests compile the real native policy, error adapter, runtime observer
and retained-capacity owner. Real small ELF fixtures exercise build-ID ownership,
different symbols/objects, missing IDs, namespace boundaries, malformed notes and
bounded input. Context callbacks and small fake CUDA-ABI DSOs test current-ID
changes, absent entry points and original error propagation without a CUDA device.
The selected CUDA 12.9 headers also compile the actual runtime translation unit.
No permanent function-body hashes, source spelling gates or scientific value
copies are added.

The internal request currently has no public method-descriptor transport. A later
size/version-tagged immutable prepared request must be designed and tested before
ordinary SDK clients can select it. Full SDK builds, real-device identity/allowance
qualification, numerical endpoints and whole-process opaque telemetry are outside
this HOST candidate and remain incomplete.

## References

- #2809 point 4; #2811 preserves diagnostics independently of allowance policy.
- `src/runtime/cuda_provider_identity.hpp`, `loaded_provider_identity.hpp`.
- `src/scf/provider_initialization.hpp`, `src/scf/cuda/provider_initialization.cuh`.
- `src/runtime/resource_ledger_plan.cpp` and its selected-inventory accounting.
- [CUDA 12.9 context IDs](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-driver-api/group__CUDA__CTX.html).
- [Loaded ELF program headers](https://man7.org/linux/man-pages/man3/dl_iterate_phdr.3.html).
