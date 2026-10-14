# Explicit offline CUDA object closure

`generativeqc_compiler.xc.bulk_aot_closure.collect_cuda_closure` is a manual
offline entrance to the existing `CacheClosure` and object store contracts.
It admits x86_64 Linux ELF NVCC 12.9, native system G++, whole-program `.cu` objects
and an explicit `sm_80` target. It never compiles, publishes, loads or activates
an object. It is not called by ordinary imports, runtime or installed packaging.

Pass the exact canonical CUDA `SourceVariant`, its LF source file, an absolute
NVCC path, ordered plain flags and a `CudaToolchainManifest`. Flags are limited
to C++17, O0–O3, `--fmad=false` and absolute safe-spelled include directories.
Other targets/releases, wrappers, response files, plugins, alternate sysroots,
RDC/device linking and LTO are unsupported. Ambiguous path spellings fail closed.
The child environment is exactly `CPU_ENVIRONMENT`; inherited `NVCC_CCBIN`,
`NVCC_PREPEND_FLAGS`, `NVCC_APPEND_FLAGS`, include and loader overrides are absent.

The collector observes actual `--version`, `--help`, system G++ specs and native
target, NVCC `-M` including system headers, preprocessing and a version-limited
`--dryrun --keep` whole-program diagnostic. `-M -Xcompiler=-H` witnesses original
user/device header path spellings and rejects ambiguous paths even with decoys;
NVCC's implicit runtime census is not entirely exposed by `-H`, so that trace
is a subset guard, not proof of complete header/search namespaces. It binds actual selected tools,
`nvcc.profile`, source/path/cwd, header bytes and the caller snapshot, and repeats
observations before returning. Diagnostic subcommands are never executed.
[NVIDIA's driver documentation](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-compiler-driver-nvcc/index.html)
defines phases as the stable interface; internal jobs can change across releases.

## Snapshot and object handoff

Manifest records have role `toolchain-file`, absolute identities and expected
SHA-256. `complete` defaults to `False`. The caller alone asserts coverage of all
transitive downstream executables, loaded libraries, device libraries, host
frontend/assembler, profile/configuration, header/search namespaces (including
absent, shadowing and PCH files), and opaque assembler inputs. Depfiles, dryrun,
ldd and guessed companion lists cannot establish this assertion.

An incomplete or drifting input produces reasons and either no closure or a
non-reusable closure with no key. A complete assertion still requires observed
driver, host frontend/assembler, actual device jobs and profile coverage.
The caller must hold that snapshot immutable throughout the following bracket:

1. Collect a reusable closure.
2. Verify the compiler-cache launcher and compile a fresh object using the returned
   `recipe["compiler"]["executable"]`, ordered `recipe["flags"]`, source,
   `recipe["environment"]` and `recipe["working_directory"]` with finite process
   ownership. The flags carry a non-self-referential `--frandom-seed` closure salt.
3. Independently inspect object/dependencies; collect again and require equal
   reusable keys before publishing through `bulk_aot_store`.

Retain launcher statistics, actual object hashes and negative receipts. Never
relabel an old `compile_probe` object with a new key. Cache-launcher provenance
does not change scientific source identity. A successful compile is not device,
numerical, resource, endpoint, full-catalog or installed-package qualification.

See [CPU closure](libxc_cpu_aot_closure.md) for the preserved GCC/Clang contracts.

## Explicit integration qualification

Run the real tests with a verified sccache 0.16+ launcher on PATH and an absolute
canonical NVCC path. Supply `GENERATIVEQC_NVCC_CLOSURE_COMPILER` and
`GENERATIVEQC_NVCC_CLOSURE_MANIFEST`; the latter is a JSON object containing
`files` (CacheDependency records) and the producer's Boolean `complete` assertion.
The fixture deliberately does not infer completeness from discovery. With both
variables present, real tests assert prerequisites and cannot silently skip:

```sh
PYTHONPATH=python python -m pytest tests/python/test_bulk_aot_cuda_closure.py -q
```

Retain `nvcc-object-receipt.json` and `nvcc-cacheable-receipt.json` under the pytest
base temp directory. The tests independently compare raw `-MD` observations,
verify pre/object/post and object-store hashes, exercise actual NVCC cache hits
and salt misses, and retain per-stage counters. A cacheable NVCC driver hit does
not imply every internal host stage was cached; safe non-cacheable fallbacks
remain visible. Opaque `.incbin` changes must change actual object bytes.
