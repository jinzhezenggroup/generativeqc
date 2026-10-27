# Decision: AOT shell dispatch is exact-architecture-match, so cross-architecture binaries must be rebuilt, not reused

Status: implemented
Date: 2026-10-27

## Problem

A build produced on one GPU architecture silently produces plausible-but-slow
numbers on another. When benchmarking on a new site (USTC SCC: A100 `sm_80`,
V100 `sm_70`) it is tempting to reuse an existing `sm_120` build, or to only
change `GENERATIVEQC_CUDA_COMPILE_ARCHITECTURES` without updating
`GENERATIVEQC_AOT_PROFILE`. Both produce a library that loads, runs, and
converges to the right energy and forces, but runs on portable fallback kernels.

The failure is invisible from the outside: no warning, no error, no changed exit
status. Only the timings move. Diagnosing it cost a full investigation because
every obvious self-check (does it run? is the energy right? is the GPU the
expected one?) passes.

## Decision

1. `select_profile_for_device(major, minor, ...)` in
   `src/scf/aot_shell_registry.hpp` resolves the AOT profile by **exact**
   `(compute_capability_major, compute_capability_minor)` match against the
   generated `kCompiledProfiles` table.
2. On mismatch the generated `resolve(major, minor)` returns `nullptr` and the
   runtime falls back to `kGenericProfile` (`{"generic_cuda","portable_cuda",0,0,
   false,true,false}` — `tuned=false`, `portable=true`). All tuned AOT shell
   kernels (force, fock, mixed_fock, streaming_fock, resident) are disabled.
3. This fallback is **retained deliberately**: a portable path that always works
   is better than a hard error for users on unanticipated hardware. The cost is
   that callers must not infer "it ran" from "it produced output".
4. Architecture is therefore expressed as a pair, and both halves must agree:
   `GENERATIVEQC_CUDA_COMPILE_ARCHITECTURES=80` with `GENERATIVEQC_AOT_PROFILE=sm_80`
   (analogously `70`/`sm_70`, `120`/`sm_120`). `cmake/GenerativeQCCuda.cmake`
   derives `profile_architecture = "sm_${architecture}"` and passes it as
   `--target-architecture`; the `--profile` argument is only emitted when the
   profile is not `auto`. Leaving the profile at `auto` while changing the
   architecture is the specific mistake this note exists to prevent.
5. Cross-architecture reuse of a build tree is **not** supported. Each
   architecture gets its own `binaryDir` (`build/cuda-release-sm80`,
   `build/cuda-release-sm70`, `build/cuda-release-sm120`) and a full rebuild.

## Rejected alternatives

- **Fuzzy/compatible dispatch** (e.g. run an `sm_80` kernel set on `sm_86`).
  Compute capability is not an ordering; kernel-level resource assumptions
  (register pressure, shared memory, warp-specialization shape) do not transfer
  monotonically, and a wrong-but-plausible kernel set is worse than the portable
  path.
- **PTX / JIT recompilation from a higher-arch build.** Would need a retained PTX
  payload in the shipped library and moves first-call cost into the benchmark,
  which is exactly the endpoint timing being measured.
- **Hard error on architecture mismatch.** Rejected because it would break the
  general-purpose deployment story for unknown hardware; see decision 3.
- **Warning on fallback.** Considered, and still open as a future improvement, but
  not required to obtain correct numbers and would need a channel that is quiet
  during legitimate portable use.

## Invariants

- A tuned-vs-portable dispatch decision must remain observable to whoever is
  qualifying a build. The build scripts under `tools/scc/` verify it out of band
  (`cuobjdump --list-elf` for the cubin architecture set, plus the generated
  `generativeqc_generated_shell_registry.cu` for the `"sm_XX"` profile literal).
- Never mix an architecture change with a profile change that disagree. Treat the
  pair as one setting.
- Numerical results must stay identical across the fallback boundary. The
  fallback changes *which kernels* run, never *what is computed*; observed
  agreement on water-96 is at the `1e-12` level in energy.

## Evidence

- `build/cuda-release-sm120/generated/production_shell_kernels/generativeqc_generated_shell_registry.cu`
  contains `launch_sm120_force`, `launch_sm120_fock`, `launch_sm120_mixed_fock`,
  `launch_sm120_streaming_fock`, `launch_sm120_resident` in the single entry of
  `kKernelSets`, alongside `kGenericProfile` — i.e. exactly one compiled profile.
- `cuobjdump --list-elf` on an `sm_120` `libgenerativeqc.so` reports 172 cubins,
  all `.sm_120.`, zero `.sm_80.`. The same check is the arch gate in
  `tools/scc/build.sh`; it costs ~0.9 s on a 1.1 GB library.
  (`cuobjdump -sass` on that file does not finish in minutes — do not use it as a
  build gate.)
- `src/runtime/cuda_device_facts.cpp::selected_facts()` performs no compute-
  capability gate: it reads 11 attributes, tolerates `cudaErrorInvalidValue` /
  `cudaErrorNotSupported`, and never rejects a device. Confirms that nothing
  upstream of the profile lookup will catch a mismatch.
- Observed consequence in practice: an `sm_120` build executed on an A100 yields
  correct physics and degraded timings, with no diagnostic anywhere in the
  output.

## Consequences

- Every new architecture costs a full compile. Mitigated by the compiler cache
  (`sccache`/`ccache`), which is mandatory per repository policy and is verified
  in `tools/scc/build.sh` via `--show-stats` before and after.
- Site build scripts must carry an explicit arch self-check, because CMake
  configure success says nothing about which cubins were embedded.
- Benchmark results are only comparable between builds with the same
  architecture profile. Timings from a fallback dispatch look like a performance
  regression, not a configuration error.

## Revisit when

- The kernel set is redesigned so that a single compiled profile can serve a
  compute-capability family (then fuzzy dispatch becomes coherent), or
- a diagnostics channel is added that can report the selected profile (at which
  point a one-line warning on fallback becomes cheap).

## References

- `src/scf/aot_shell_registry.hpp` — `ProfileInfo`, `select_profile_for_device`.
- `cmake/GenerativeQCCuda.cmake` — architecture → profile derivation and
  `--target-architecture` / `--profile` / `--profile-map` wiring.
- `CMakeLists.txt` — `GENERATIVEQC_CUDA_COMPILE_ARCHITECTURES`,
  `GENERATIVEQC_AOT_PROFILE`, and the `CUDA target architecture is required`
  fatal path.
- `tools/scc/` — site build/run scripts and the architecture self-check.
