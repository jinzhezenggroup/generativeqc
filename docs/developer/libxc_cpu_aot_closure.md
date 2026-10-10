# Offline CPU AOT dependency collection

`generativeqc_compiler.xc.bulk_aot_closure.collect_cpu_closure` collects actual
GCC dependency data for a CPU `SourceVariant` and a caller-owned `.c` file whose
bytes match that variant. It runs only on explicit invocation. It neither builds
an object nor invokes a compiler during import or ordinary runtime execution.

The supported driver is the ELF executable resolved by `/usr/bin/gcc` on Linux,
with a native x86_64 or aarch64 Linux target matching the host. The supported
ordered flags are `-std=c99`, `-O0` through `-O3`, `-fPIC`, and joined `-I` options
with existing absolute directories. Collection adds `-Werror=date-time` and a
deterministic `-frandom-seed` derived from the unsalted closure payload to the
returned recipe. All child processes use the exact `CPU_ENVIRONMENT` mapping;
compiler/include/loader overrides from the caller's environment are not inherited.
Other drivers, wrappers, plugins, response files, languages, targets/sysroots and
flags yield no reusable closure. This is a deliberately narrow offline support
domain, not a general build-system interface.

## Completeness boundary

GCC `-M` includes system headers as well as nested user headers. The collector
parses its fixed-target make rule, hashes each resolved file, repeats dependency
discovery, and rechecks source, compiler, header and manifest bytes. Path parsing
preserves ordinary literal backslashes and decodes GNU make's odd backslash runs
before whitespace. GCC can emit terminal filename backslashes indistinguishably
from escaped spaces or line continuations. These ambiguous spellings fail closed,
even when a merged decoy pathname exists: escaped whitespace followed by a slash
or another unescaped whitespace, even backslash runs before whitespace, and
continuations without a preceding separator are rejected. This also excludes
otherwise valid filenames with the same ambiguous spelling. Unambiguous terminal
spaces and tabs are preserved; compiler queries trim only CR/LF record terminators.
It binds the
exact source/ABI emission identity, source bytes and physical path, working
directory, ordered flags,
compiler path/bytes/version, native target, resolved downstream programs,
preprocessor output (including non-including `__has_include` branches), emitted
specs and controlled environment into the existing `CacheClosure` identity.
A finite shared deadline
covers compiler queries, with the existing compiler-process group owner enforcing
each subprocess timeout. Hashing checks the deadline before and after each regular
file; underlying filesystem I/O itself is not forcibly interrupted.

A depfile does **not** enumerate all object-generation inputs. A
`CpuToolchainManifest` therefore defaults to `complete=False`. Its records must
have role `toolchain-file`, absolute file identities and expected SHA-256 values.
All listed files are verified, and coverage must include the resolved driver,
`cc1`, and assembler. These checks are minimum structural requirements, not proof
of a complete manifest.

Only the manifest producer may assert `complete=True`, based on a controlled,
immutable toolchain/build snapshot that covers downstream programs, dynamically
loaded libraries, specs/configuration, and all include search namespaces,
including absent/shadowing files and precompiled headers. Neither a guessed list
of GCC companions nor `ldd` output alone proves this coverage. Keep the snapshot
stable through collection and compilation. The collector cannot detect a change
that is restored between observations or prove the producer's coverage assertion.
Without that evidence, retain the default incomplete manifest and diagnostic
header census; no reusable key is issued.

Missing files, unsupported depfile syntax, compiler failure, timeout, observed
mutations or missing downstream coverage fail closed. Results contain reasons
and, once compiler/source identification succeeds, an incomplete `CacheClosure`.
Earlier failures return no closure. Neither state has a reusable key.

## Object and store handoff

1. Under the immutable snapshot, collect a reusable closure before compilation.
2. Compile the exact file using the returned recipe flags and identified driver,
   in the recipe's `working_directory` under `CPU_ENVIRONMENT`, through a
   verified compiler-cache launcher and the
   finite compiler-process owner.
3. Collect again under the same snapshot. Require both reusable keys to match
before calling `bulk_aot_store.store_artifact` with the new object.
4. Before a later lookup, collect a fresh closure from the current snapshot and
   pass it to `bulk_aot_store.lookup_artifact`.

The physical source path deliberately affects identity because `__FILE__` and
quoted include resolution can affect object semantics. The working directory
also affects assembler directives such as relative `.incbin` inputs, even when
the complete manifest covers every possible input file. It must remain unchanged
through collection and compilation. Registration aliases
emitting the same source/ABI at that same path retain distinct import provenance
and share executable identity. Changed headers, toolchain/configuration bytes,
source or ordered flags produce a different key or an explicit rejection.

Collecting today's headers cannot qualify an old `compile_probe` object: the old
probe recipe did not observe the dependencies used to generate it. The collector
does not alter that probe's diagnostic contract or the existing cache/store
schema. Production packaging, catalog-wide economics and CUDA numerical/resource
qualification remain separate work under #1123.

The compiler cache is a separate identity layer: a store miss does not prove it
returned fresh object bytes. The recipe's deterministic seed transports the
observed dependency identity into the GCC command, including manifest-covered
opaque assembler inputs, configuration and downstream toolchain files that
`-M`/`-E` or a compiler cache may not otherwise enumerate. Use the complete
returned flags, and do not enable options that ignore the seed or weaken cache
correctness. The producer must include all opaque inputs in its complete
manifest and verify the selected launcher honors this command identity;
unsupported cache behavior cannot justify publishing a reusable object.

## Verification

```bash
PYTHONPATH=python python -m pytest -q tests/python/test_bulk_aot_closure.py \
  tests/python/test_bulk_aot_cache.py tests/python/test_bulk_aot_store.py
python tools/check_compiler_structure.py
```

On Linux the real-GCC tests require GCC and a usable compiler-cache launcher;
missing tools fail the integration test. They independently compare raw GCC
dependency output and file hashes, compile the same source between collections,
exercise verified store reuse/invalidation, and inject a header mutation between
real dependency scans. Non-Linux runs skip those integration tests. No GPU or XC
numerical acceptance is claimed by these toolchain tests.
