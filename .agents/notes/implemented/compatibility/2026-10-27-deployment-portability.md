# Decision: preserve deployment paths and the selected visible GPU

Status: implemented
Date: 2026-10-27

## Problem

The public autotuner fixed device zero without exposing a selector. Merely
adding a CLI option would leave candidate timing and independent numerical
drivers on zero while probing and timing endpoints on another GPU. Persistent
benchmark grids also queried device zero regardless of the current device.
Two older benchmark helpers removed an empty `CUDA_VISIBLE_DEVICES`, exposing
devices that the parent intentionally hid. Batch screening defaulted to a
site-specific `/group/software` compiler, runtime provider discovery favored
fixed system paths, and optional ERI spill ignored `TMPDIR`.

Two-GPU execution then exposed a separate TensorIR lifecycle bug. Preparation
correctly restored the caller's original current device, but subsequent static
initialization, execution and resident operations merely checked the current
device instead of selecting their owner. GPU 1 therefore failed when the caller
remained on GPU 0. Standalone Array API JIT also bypassed the chemistry loader's
provider preloading and could not resolve a relocated cuBLAS installation.

## Decision

Carry the requested CUDA-visible ordinal through probing, candidate drivers,
independent numerical drivers and final endpoint validation. Select the device
before initialization/allocation in generated host drivers. Embedded candidate
functions inherit that selection and derive occupancy from the current device.
Never edit CUDA visibility, including an empty mask. Default zero remains an
overridable visible ordinal, not a physical GPU index or topology assumption.

Use shared NVCC discovery, supporting explicit compiler paths and configured
toolkit roots before PATH. Invalid explicit configuration must not silently
choose another compiler. Provider discovery honors loader paths, relocated
toolkits, the current Python prefix and host architecture before system fallbacks.
Optional spill follows the platform temporary directory or its explicit override;
an invalid/empty directory disables it. The resolved directory participates in
retained-source policy matching so a changed temporary-directory setting cannot
silently reuse the old spill policy.

Use the shared `CudaDeviceScope` around TensorIR static initialization, ordinary
and resident execution, graph configuration, transfers and same-device copies.
Its lifetime includes exception cleanup: queued host copies must drain on the
owner's device before restoring the caller. Failed device selection admits no
stream operation. Cross-device resident copies remain explicitly unsupported
and fail before touching either stream. Array API execution invokes the shared
provider preloader before loading its JIT library; offline lowering stays free
of runtime discovery.

Developer validation runners use the same compiler discovery instead of a site
prefix. Their explicit target selectors include the low-rank and spatial runners.
Spatial and timing benchmarks accept visible device selectors. Temporary
validator caches honor the platform temporary directory, and the stationary
runner preserves the caller's compiler cache instead of forcing a `/tmp` cache.

## Retained memory and system interfaces

The audited Direct fixed-topology/streaming policy derives capacity from actual
`CudaTargetInfo.total_global_memory`, with conservative unknown-target limits.
The existing native policy tests cover a synthetic 2-GiB device (64-MiB topology
and 128-MiB streaming arenas). Optional resident ERI caches retain their explicit
8/16-GiB ceilings, complete numeric-budget admission, live `cudaMemGetInfo`
checks, allocation refusal and bounded recomputation. These ceilings are not
assumed physical VRAM sizes; removing them would weaken resource admission.

Linux `/proc` and `/sys` interfaces and optional system library search locations
are not developer-machine dependencies. The compiler-cache server's short
private `/tmp` socket directory intentionally avoids Unix socket path-length
limits; its separate leased persistent cache remains configurable and outside
the shared cache tree. Do not move that socket under an arbitrarily deep cache
directory without addressing the socket ABI limit.

The pinned allocation/residency audit runners intentionally require exactly one
visible GPU and verify that invariant through CUDA before capture. Their ordinal
zero denotes that sole Slurm-visible device, whose physical UUID is independently
resolved; it is not a request for physical GPU zero. Changing these versioned
single-device evidence protocols is unnecessary for multi-GPU runtime support.

## Validation and limits

Executable host-driver probes simulate heterogeneous visible ordinals and
reject architecture mismatches or invalid selections before candidate work.
Environment/discovery tests cover hidden GPUs, UUID/remapped masks, relocated
toolkits, invalid explicit compilers, non-x86 library layouts, and temporary
directory overrides/failure. Generated host ABI tests use the real device scope
with fault-injected CUDA controls to cover owner selection, restoration, partial
upload failure, retry and cross-device copy rejection.

- Deployment/TensorIR/tool regressions: 272 passed, 38 skipped, one deselected.
  The deselected native source/probe identity test requires a built chemistry
  library unavailable in this checkout; its required coverage is unchanged.
  The skipped tests include opt-in GPU and deliberate-damage controls.
- Shared device-scope host contracts: 19 passed.
- Node1 Slurm job 14712 (`main`, two RTX 5090s, five-minute limit) preserved
  `CUDA_VISIBLE_DEVICES=2,6`: emitted tuner probes selected visible 0/1, rejected
  2, and Array API FP32/FP64 primal/gradient results matched NumPy on both GPUs.
- The opt-in `test_tensor_cuda_devices.py` passed on the same node with two GPUs
  and a five-minute limit: external static initialization, ordinary execution,
  graph warmup/capture/replay/reset, resident upload/copy/run/download and owner
  destruction restored current GPU 0 after operating on GPU 1. NumPy weighted
  scatter is the independent numerical oracle.
- Node1 Slurm job 14715 (two GPUs, three-minute limit) passed all four timing-probe
  smoke cases and their independent FP64 checks on visible GPU 1.
- CUDA compilation used verified sccache 0.16.0 with finite compiler timeouts;
  host probes used the existing verified ccache fixture. Local raw logs and
  cache statistics are retained under ignored `.artifacts/cuda-portability/`.

The original target-portability PR #2831 merged while these additional repairs
were being validated. The deployment change was then reapplied to master
`ffef9a8a91063f43bd80996b60d9802fcf29fb7d`, preserving the merged offline IR
rejection checks. The combined host suite passed 295 tests (38 skipped, one
native-library-dependent test deselected). Node1 job 14716 repeated the two-GPU
Array API checks and the graph/resident regression on the rebased sources,
again preserving `CUDA_VISIBLE_DEVICES=2,6` within a five-minute limit.

Scientific kernels and numerical acceptance tolerances are unchanged. These
tests establish device selection and deployment behavior on available hardware,
not non-5090 numerical/performance qualification. No full wheel was built; the
separate wheel PR retains that time/cost qualification requirement.
