# Decision: derive CUDA targets from callers or the executing device

Status: implemented
Date: 2026-10-27

## Problem

The compiler and native runtime already had explicit target/resource records,
portable kernels and profile compatibility checks. Three source/runtime entry
points still implicitly selected the RTX 5090 target: Array API execution, the
generic fast development preset, and derivative registry generation.
In particular, overriding a CMake target did not override the derivative
generator's hardcoded `sm_120` profile or production manifest.

## Decision

- Array API defaults to lazy CUDA driver compute-capability discovery for the
  requested visible ordinal. Tracing/lowering remains offline. Explicit targets
  still support device-independent planning. Detection neither creates a context
  nor changes visibility and has no CuPy/native chemistry-library dependency.
- Generic development/release presets accept explicit numeric architectures and
  explicitly select portable kernels. Named `sm120` release presets retain their
  measured profile and intentionally specific binary identity.
- Shell generation and derivative registry generation share CMake target/profile
  argument resolution and the configured production manifest. Each NVCC device
  pass selects its registry. Portable profiles and disabled shell AOT produce
  empty registries and retain the existing bounded radial implementation.

Wheel architecture coverage, compiler snapshot identity and CI time limits are
kept unchanged in this compatibility fix. A separate packaging change must
qualify a wider image matrix and its build cost after these fixes land. Existing
single-target wheels still require source builds for other GPU architectures.

## Retained boundaries and rejected alternatives

Do not replace every `sm_120` string: schema-v1 compatibility, measured manifests,
resource qualification, named distributions and historical evidence have explicit
target semantics. In particular the measured cooperative/phased Becke defaults
retain their existing generic fallback on other devices. Removing those gates
without independent numerical and complete-endpoint performance qualification
would invent cross-device evidence. Hardware/model names are not used for the
new automatic selection; CUDA-visible ordinals are resolved by the driver.

Do not promote the 5090 shell schedule into portable profiles. A smaller/unknown
device must not acquire unqualified register/shared-memory assumptions. A custom
accepted profile can enable derivative packages on another target; tests exercise
this with a synthetic Hopper profile and stub arithmetic, not a claimed Hopper
scientific qualification.

## Evidence

- Executable tests cover visible ordinals, discovery failures, explicit/offline
  selection, CMake presets/profile mappings and host-compiled derivative dispatch
  for SM 80/86/89/90/100/120 plus an unlisted target. Integral arithmetic is
  unchanged; selector probes use stub arithmetic to isolate admission.
- CUDA 12.9.86 compiled the same FP64 TensorIR example independently for all six
  probed targets using the verified sccache 0.16.0 launcher.
- Node1 Slurm job 14675, `main`, `gpu:5090:1`, finite 10-minute limit, preserved
  scheduler visibility: four Array API tests passed, including FP32/FP64 primal
  and gradient results against NumPy. This validates detection/execution on the
  available GPU, not performance on other architectures.
- Multi-target CMake configuration and derivative code generation were exercised
  with portable profiles plus an explicit `sm_120` override. Local raw build/JIT
  records and the node1 receipt are under ignored `.artifacts/cuda-portability/`.
- The generated derivative header compiled with stub arithmetic into one CUDA
  fatbinary; `cuobjdump --list-elf` confirms all six requested cubins. This checks
  the real NVCC preprocessor/type contract, independently of host selector tests.
- Compiler ownership check: 567 modules, zero dependency errors.

Before opening the PR, the change was rebased onto master
`32fb32195ba3b4cd1588dcfe9e7eff91225b996e` without the original worktree's
unmerged pruning commits. The repeated compiler ownership check covered 579
modules with zero errors. Node1 Slurm job 14683 repeated the four Array API tests
successfully with the rebased runtime/compiler sources and the same finite GPU
allocation contract.

No full wheel distribution build or non-5090 device/performance qualification was
performed. The additional six-target compilation of
`direct_bounded_fallback.cu` reached its 600-second limit (exit 124) without a
finished object; it is not a passing native build receipt. The command contains
the verified sccache launcher and all six requested NVCC image targets. Its
processes exited without stopping the shared cache server.

No release, release tag or publication was created. Target-specific source builds
remain the deployment path for GPUs outside the prebuilt wheel's target.

## Revisit when

New devices acquire independent endpoint qualification. Keep scientific profile
admission separate from binary packaging and from device identity discovery.
