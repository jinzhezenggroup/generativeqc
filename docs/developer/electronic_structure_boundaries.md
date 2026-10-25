# Electronic-structure architecture boundaries

GenerativeQC shares native infrastructure across HF, DFT, post-HF and coupled-cluster
methods. This document records the top-level dependency direction protected by
`tools/check_electronic_structure_boundaries.py`.

## Shared native owners

### Compiled named KS programs

`src/methods/named_ks_catalog.*` owns the build-time MethodIR catalog once in the
native library. The native CLI borrows its listing and component/exchange arrays;
it no longer instantiates a separate CLI-only table. Python `native_ks_options`
queries the already-selected library for a matching complete MethodIR identity
and numerical domain, preserving caller grid, tiling, schedule and precision
controls. Components and spin-resolved Fock coefficients come from the existing
compiler, not a second native scientific algebra implementation.

A missing optional query retains explicit old-library metadata lowering. A
successful zero selection declares a custom/noncatalog or unrepresented-domain
compiler route; it does not replace a same-name custom model with a catalog
default. Invalid arguments, native refusal and inconsistent publication never
retry compiler lowering or discover another library. Selected pointers borrow
immutable library storage until ordinary native preparation copies them.

Catalog backend flags describe representability only. Existing prepared owners
still admit basis, provider, backend, precision and requested properties. This
does not qualify every catalog entry, retire custom MethodIR compilation or
establish complete method/basis/CLI parity.

### KS host capacity ownership

`src/scf/ks_host_resources.hpp` owns CPU and CUDA KS host capacity and fleet
lifetimes for Python metadata planning and the native CLI. Binding reservations
come from the same CUDA-header-free constants used by the actual tensor
bindings. Host sizing neither discovers providers nor admits new scientific
domains. Consumer allowances and old-library compatibility are specified in
[resource planning](../maintainer/resource_planning.md); device inventories,
prepared leases and endpoint allocation receipts remain separate contracts.

### Method-neutral infrastructure

The following directories are method-neutral owners:

- `src/core/`
- `src/runtime/`
- `src/tensor/`
- `src/solver/`
- `src/response/`

They may depend on other shared chemistry, integral and runtime primitives, but
must not include concrete method implementation headers from:

- `src/scf/` (legacy mixed SCF/HF execution while the split is in progress)
- `src/hf/`
- `src/dft/`
- `src/posthf/`
- `src/cc/`

The check resolves repository-relative include spellings, including `../`
paths, before applying the rule. It scans `.cpp`, `.hpp`, `.h`, `.cu` and
`.cuh` files. Commented-out examples do not create edges.

The method-neutral electron-interaction interface lives in the otherwise mixed
`src/integrals/` area. The checker therefore protects
`src/integrals/electron_interaction_source.hpp` as an individual provider
contract rather than labeling every integral implementation as shared.

Two existing reverse edges are carried as explicit debt ceilings rather than
silently exempted:

- `runtime/cuda_runtime.cu -> scf/aot_shell_registry.hpp`
- `runtime/host_component_trace.hpp -> scf/reference/observation.hpp`

They may disappear without a coordinated baseline edit; any additional
shared-to-method edge fails the check.

Post-HF still has 16 direct dependencies on SCF-owned driver, derivative, and
provider headers. Those exact edges are also explicit shrink-only debt:
removing one is accepted, while any new `src/posthf/ -> src/scf/` edge fails at
the include location. This prevents the shared-reference migration from moving
backward without pretending the current migration is already complete.

This cross-method gate complements, rather than replaces, the more detailed
`tools/check_scf_structure.py` and `tools/check_compiler_structure.py`
ownership checks.

### CPU RCCSD reference-view handoff

`cc::prepare_canonical_reference_input` consumes the existing immutable,
borrowed `core::ElectronicReferenceView` contract. The RCCSD problem builder
gets its occupied/virtual layout, reference energy, canonical orbital spectrum,
and row-major `C^T F C` from this common view instead of independently reading
`hf::PhysicalReference` fields. It rejects malformed matrices, nonfinite values
and unsupported unrestricted occupations **before** generating transformed Fock
data or requesting AO integral blocks.

The producer owns the orbitals: `RccsdNativeState::reference` retains the HF
reference through CC response and source retirement, while the synchronous
adapter owns only its freshly transformed MO Fock matrix. Validated RKS final
states can expose the same input shape without introducing an SCF-private
dependency, but this contract check does **not** enable CCSD-on-KS or UCCSD.
Conventional AO and CUDA DF providers still retain their existing HF-specific
source binding; provider migration and persistent
source/geometry/solve-epoch invalidation remain separate follow-ups. No GPU
performance or CUDA scientific acceptance is inferred from the CPU
reference-view tests.

### HF reference ownership

The first explicit HF owner is `src/hf/reference.hpp`. The canonical restricted
HF reference consumed by MP2/CC is method state, not SCF solver state. Legacy
`scf::PhysicalReference` remains a compatibility alias while callers migrate,
but new post-HF interfaces should name `hf::PhysicalReference` directly.

The architecture checker treats `src/hf/` as a concrete method area. The
remaining `src/scf/` tree is tracked as transitional ownership rather than
being counted as HF-only; concrete RHF/UHF execution can move out later
without changing the shared solver contracts.

The current shared-facility boundary mapping is:

| Facility | Shared owner | Enforced boundary |
| --- | --- | --- |
| Iteration and DIIS | `src/solver/` | directory cannot depend on a concrete method; CC and SCF remain include-graph consumers |
| Electronic reference | `src/core/electronic_reference.hpp` | `src/core/` cannot depend on a concrete method; SCF and DFT remain consumers |
| Interaction provider | `src/integrals/electron_interaction_source.hpp` | the individual provider contract cannot depend on a concrete method; SCF and post-HF remain consumers |
| MethodIR and ExecutionIR | `python/generativeqc_compiler/method/` | `tools/check_compiler_structure.py` enforces compiler-package directions |
| Linear response | `src/response/` | directory cannot depend on a concrete method; post-HF remains a transitive consumer |
| Runtime and workspace | `src/runtime/` | directory cannot acquire new concrete-method edges; the two existing reverse edges remain explicit debt |

For the native facilities in this table, the JSON report derives direct and
transitive consumers separately from resolved repository include edges. A
missing canonical owner or required consumer fails in a full checkout. This
catches a method silently forking away from a shared facility without depending
on source-text formatting or a line-count threshold.

## Duplicate infrastructure inventory

CC host bounded iteration and host DIIS plus its small dense linear solve moved
to `src/solver/`. Their duplicate-debt ceilings are
now zero, so reintroducing any of those generic host owners under `src/cc/`
fails the architecture gate. One method-specific CUDA DIIS orchestration owner
remains in `src/cc/cuda_solver.cu`; its ceiling is one until a shared device-loop
contract is justified.

Duplicate declarations are located with a comment- and literal-aware C++ token
scan that distinguishes definitions from calls, forward declarations, template
parameters and inheritance. The one CUDA exception is restricted to
`src/cc/cuda_solver.cu`; moving that identified owner fails even if the total
count stays at one. Failures name every matching `path:line`.

This named-declaration inventory is not a proof that arbitrary renamed code is
semantically DIIS. Canonical owner/consumer checks prevent CC from silently
replacing the shared host facilities, while infrastructure PR review remains
responsible for identifying an additional or fundamentally different duplicate
algorithm. The debt baseline is a ceiling, not a target: removing the remaining
CUDA owner does not require a synchronized baseline update to keep CI green.

## CUDA KS declared-capacity composition

When the selected library exposes the additive CUDA KS fleet query,
`scf::CudaKsDeviceFleetResources` composes the existing allocator-owned leaf
inventories. It owns the value-only one-electron setup bound, alternative ECP
scratch, quadrature/Coulomb overlap, retained state/XC/J and matrix-provider
totals, serialized rebuild excess, and minimum/cached stationary candidate peaks.
Successful composition is allocation-free and shape-only, including in a CPU-only build;
obtaining CUDA leaf inventories still requires their independently supported
native queries. Composition is not CUDA scientific execution or admission.
The native CLI uses the same typed fleet owner for its independently supported
ordinary energy dry runs; this does not expand CLI force/model admission.

Python marshals compact topology and native leaf capacities, validates the wire
publication and presents the existing resource labels. Only a missing fleet
symbol permits explicit legacy per-item composition. An actual native refusal
or inconsistent publication never retries Python sizing. The candidate decision
`cuda_fleet_composition` records which route actually supplied the capacities.

The shared resource-plan engine retains global lifetime composition, ranking and
declared-budget admission. Scientific consumer eligibility, host force
allowances, provider allocation receipts and complete simultaneously-live
physical/RSS accounting remain separate contracts. No all-domain CUDA force or
response/HVP capability follows from a capacity query. Native ABI transaction,
overflow, aliasing and exact declared-budget tests protect this boundary; GPU
execution and full endpoint performance require separate qualification.

## Regression inventory

The report carries stable, method-specific test-file pointers for the required
regression stories:

- RHF/UHF energy, DFT energy/XC, and RCCSD energy/amplitude equations;
- DFT, post-HF and CC forces;
- shared GMRES plus UHF, RKS, CC orbital response and molecular HVP behavior;
- runtime workspace and CC numeric-capacity accounting; and
- compiler and native runtime dependency audits.

These entries are an ownership/consumer index, not a replacement for executing
the tests. In a full checkout, removing an indexed gate without updating the
architecture inventory is CI-blocking. Infrastructure PRs must still state the
affected endpoints, numerical tolerances or unchanged-behavior evidence, and
resource-accounting evidence relevant to their actual change.

## Metrics

Run:

```console
python tools/check_electronic_structure_boundaries.py --json
```

to obtain:

- shared-module dependency edges,
- canonical shared-infrastructure owners plus direct/transitive consumers,
- numerical/resource/dependency regression test pointers,
- source files/lines/bytes by top-level native area,
- diagnostic shared/HF-only/DFT-only/CC-only ownership rollups,
- generated-line counts separated from total lines, and
- locations/counts for the tracked duplicate solver infrastructure.

The size metrics are inventory only. They do not impose arbitrary line-count
gates; architectural direction, required owner/consumer relationships,
regression-gate presence and duplicate-owner growth are the enforced invariants.
