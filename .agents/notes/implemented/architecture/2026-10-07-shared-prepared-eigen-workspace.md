# Decision: share prepared symmetric-eigen workspace contracts

Status: implemented
Date: 2026-10-07

## Problem

The shared provider lowerer and prepared handle owner removed duplicate vendor
calls and lifecycle code, but method adapters still implemented provider sizing:
GFN2's nested mode/capacity loops, signed Jacobi units, transactional maxima, and
SCF/DF singleton or ordered-capacity queries and allowance checks. Leaving those
algorithms embedded would not establish shared prepared workspace ownership.

## Decision

`solver/cuda/symmetric_eigen_workspace.{hpp,cpp}` owns an actual preparation
algorithm above the one-query provider ABI, without callbacks or allocators:

- A borrowed immutable domain declares family, dimension and ordered inclusive
  capacity ranges with explicit vector modes
- Preparation visits every declared query, including duplicate capacities, in
  order; validates signed Jacobi elements and byte conversion; and reduces
  componentwise device-byte, host-byte and Jacobi-element maxima
- Previously admitted byte components can provide a floor, so another provider
  sharing a GFN2 subarena is never shrunk. No result is published after a partial
  provider failure or invalid Jacobi size
- Admission applies explicit device/host limits and optional positive-device
  policy. It neither invents host caps nor treats unknown opaque allocations as
  numeric bytes
- Binding validates borrowed pointer/extents before publishing a provider view.
  Queried Jacobi lwork and padded byte-derived capacity are distinct modes

The prepared envelope contains only numeric requirements. Method owners retain
shape/mode dispatch admission and provider selection. It is not an executable
cache, a generalized eigensystem specification, or a new allocation owner.
Envelopes are setup locals; existing retained fields and metadata reservations
are unchanged.

## Actual consumers and deleted logic

| Consumer | Declared domain and policy | Removed embedded work |
| --- | --- | --- |
| GFN2 generic bucket, including spin adapter | NOVECTOR then VECTOR; capacities 1 through actual count; zero allowed; previous requirement floor | Both nested provider-query loops and method-local componentwise maxima |
| GFN2 Jacobi bucket | Same ordered modes/capacities inside existing <=32-orbital gate | Signed-element validation, byte multiplication and maxima loop |
| RHF/UHF setup | Jacobi singleton spin capacity; generic ordered physical/spin capacities, including duplicate RHF queries; Xsyevd singleton; positive device bytes | Repeated family-specific query and aggregation branches |
| Ordinary KS eigensolver | Xsyevd singleton; device and host bounded separately; zero allowed | Direct query and inline allowance comparisons |
| Compact DF SCF library | Selected-family singleton actual batch; positive device bytes; existing device-only allowance | Duplicate Jacobi/generic sizing, byte conversion and allowance checks |
| Ordinary DF AO frame | Xsyevd singleton; both allowances bounded; zero allowed | Direct query and inline allowance comparisons |
| DF metric setup | Xsyevd singleton query followed by unchanged serial per-system submissions; device-only allowance; zero allowed | Direct query and inline allowance comparison |

GFN2 keeps the device subarena and pinned host buffers. RHF retains its async
pool allocation and cuBLAS sharing. Ordinary KS/DF keep synchronous allocations
and vectors; compact DF keeps synchronous allocation and malloc. The shared
binding preserves actual allocation extents rather than replacing padded
capacities with query bytes. GFN2 Jacobi still passes floor(device_bytes/8);
SCF/DF pass their exact queried element count.

## Invariants and rejected alternatives

- Preserve all GPU kernels, generated science, method dispatch, host launch,
  copy and synchronization order, and numeric arena layouts
- No workspace monotonicity assumption, query deduplication, combined sum of
  device/host maxima, or premature publication after partial query failure
- Do not unify GFN2 Cholesky/TRSM transforms with SCF canonical-orthogonalizer
  GEMM transforms; those are different scientific contracts
- Do not route density/P/W work through `PreparedSymmetricProduct`: its XC
  ABt+BAt operation is not weighted Gram, and #1879 parks new BLAS optimization
- Do not claim complete embedded runtime removal: overlap/cache provenance,
  graph/device-tail orchestration, method arena binding and CPU ownership remain
- Keep the inherited prepared-handle namespace repair: SCF contains its own
  `solver` namespace, so shared handle fields require global qualification

## Evidence

The host fixture compiles the actual shared service and lowerer with official
opaque-pointer/enum-status and CuMetal strided signatures. It executes complete
GFN2 query/launch and compact DF setup/solve bodies, complete ordinary DF AO
preparation, and verbatim RHF, ordinary KS and DF metric sizing/admission regions.
Unrelated allocation/device services are explicit host doubles.

Traces cover both modes and every reachable capacity, non-monotonic maxima,
duplicate RHF queries, every partial query failure, unchanged output after
failure, negative/zero/maximum Jacobi sizes, maximum byte allowances, terminal
INT64_MAX capacity, pointer/extents and both Jacobi binding modes. DF AO tests
inject every allocation failure and verify total numeric bytes and warm reuse;
ordinary KS traces verify existing host metadata charges. Metric traces retain
one query, exact per-system offsets and first-failure serial submission stops.
Adversarial include-order tests compile real SCF owner headers.

A fresh cached CPU build and all 76 native CTests pass. CPU endpoint/oracle,
workspace and solver-lowering tests pass (25 passed; 10 CUDA/optional-provider
skips) using the existing LP64 SciPy OpenBLAS provider and its dependency path.
All 35 device functions in the only changed CUDA source are byte-identical to
base 3fda3b6; all ten generated GFN2 science headers are byte-identical. Generated
solver metadata changes only SHA-256 identities and includes the shared service
sources in both generator and CMake dependency closure.

The native host tests do not qualify GPU numerics, capture or performance. This
authoring environment has no CUDA compiler/device; ordinary NVIDIA/CuMetal CI
remains required. No new performance campaign or provider promotion is implied.
The shared implementation retains the original xTBloom attribution and scoped
CUDA/MKL linking permission; adapted source hashes are updated honestly.

## References

- `2026-10-07-shared-symmetric-eigen-provider.md`: one-call vendor lowering
- `2026-10-07-shared-prepared-eigen-handles.md`: shared handle lifetime
- #1240/#1890 and #933/#934: migration/solver ownership
- #1814/#1879 and #1882: density and mixing boundaries
