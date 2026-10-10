# Shared CUDA matrix products

`src/scf/cuda/matrix_library.hpp` is the borrowed execution boundary used by
CUDA RHF/UHF and resident KS matrix products. Provider selection belongs to the
prepared caller. The adapter never changes an explicit library selection into
native execution.

## Device masks and output ownership

A null `active` pointer means every system is active. Otherwise every nonzero
mask byte selects one physical system, including all its spin matrices. Inactive
output bytes must remain unchanged, even when their input matrices contain NaNs.

Native products apply the mask in their arithmetic kernel. Library products
write their complete GEMM result into `MatrixLibraryResources::masked_output_`
and then enqueue the existing selected-matrix copy on the same stream. The
ordinary-versus-strided GEMM and physical/spin operand strides are unchanged.
No host mask read, synchronization, allocation, or provider substitution occurs
at this boundary. Masks are read at execution, including successive Graph
replays; callers must order all accesses on the borrowed stream or establish the
corresponding dependencies themselves.

Masked library execution requires a contiguous scratch span of at least
`batch_size * spin_count * nbf * nbf` doubles, with `spin_count = 1` for plain
products. The span must not overlap either operand, the output, or the mask.
The caller retains it through completion of every queued operation or captured
Graph that refers to it. Missing, short, overlapping, overflowing, or
nonrepresentable storage/launch shapes return `INVALID_ARGUMENT` before any
submission. Native and unmasked library requests do not need this span.

## Resource accounting

The RHF/UHF arena reserves one output-sized span only when its unchanged
prepared policy selects cuBLAS. Its checked arena layout includes those bytes
before allocation, reference-capacity admission, and capture. Native-only
resource-query entry points retain their existing library-route rejection.

KS reserves one spin-output-sized span whenever its existing shape-only matrix
provider policy admits a library. The same typed `KsStateStorage::partition`
serves numeric allocation and the public state-byte query, so these are explicit
numeric bytes, separate from the unchanged opaque provider allowance. A later
provider allocation fallback can leave this bounded span unused.

## Validation boundary

`tests/python/test_matrix_library_mask.py` compiles the complete host adapter
with vendor/queue doubles, executes the selected-copy kernel body with host
launch indices, checks against NumPy, and exercises both owners' real storage
layouts. It covers changed masks across replay, bit-exact inactive sentinels,
spin broadcasting, scaling, alias/capacity refusal, errors, and launch bounds.
These checks do not qualify real CUDA Graph execution, cuBLAS arithmetic, or
complete scientific endpoints. Source-matched real-device qualification remains
necessary for those claims.
