# Offline CPU AOT dependency collection

`generativeqc_compiler.xc.bulk_aot_closure.collect_cpu_closure` collects actual
native C dependency data for a CPU `SourceVariant` and a caller-owned `.c` file whose
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

## Explicit GCC C++20 objects

Pass `language="c++"`, `compiler="g++"`, and explicit flags such as
`("-std=c++20", "-O1", "-fPIC")` for a matching `.cpp` translation unit.
The default language remains C, including its existing keys and ABI. C++ is
limited to the canonical native Linux ELF `/usr/bin/g++` driver and the same
plain optimization/PIC/absolute-include flags, with exactly one `-std=c++20`.
Clang++, CUDA C++, other standards, alternate ABI options, modules, PCH, LTO,
plugins, response files, cross/sysroot compilation and link closure are outside
this domain. A neighboring `.gch` file or directory is rejected.

The collector requires manifest coverage of the actual `cc1plus` and assembler,
checks their regular-file/ELF identity, and validates a two-job G++ `-###` object
plan with the exact source and frontend-to-assembler handoff. Only the validated
temporary assembly pathname is normalized. Driver selection is rechecked around
each query. The returned `compiler.executable` identifies the physical driver;
the launcher does not replace that identity. It binds queried specs, actual
include search paths and full translation-unit macro output, including
`__cplusplus` and `__GXX_ABI_VERSION`, to a distinct C++ context. Included standard
library configuration headers participate in the header closure. The producer
must still assert the immutable standard-library/include/configuration namespace;
discovery does not prove completeness or qualify a preexisting object.

Use the same collect/compile/collect/store handoff below. The returned GCC seed
must reach the real compiler-cache command. Stored-object execution in tests
uses a separate link step; it does not qualify a reusable executable/shared-link
closure, production bundle, runtime cache or scientific endpoint.

Clang is an explicit alternative: pass `compiler="clang"` and
`compiler_family="clang"`. The default family remains GCC, including its prior
recipes and diagnostics. The Clang executable must resolve to `/usr/bin/clang`,
be ELF, identify itself as Clang, and report a native x86_64/aarch64 Linux target
matching the host. Collection adds `--no-default-config` and `-fintegrated-as`;
explicit configuration flags remain unsupported. It binds the queried resource
directory and actual `-###` object job, admitting exactly one integrated `-cc1`
job in the driver executable, with matching target, resource directory and C
language. External assembler/frontend, plugin/module/sysroot jobs and unobserved
job models fail closed. Clang does not inherit GCC's cc1/as/specs assumptions.
The job diagnostic is data to validate, never a command to execute.
The Clang recipe also returns its resolved `compiler.executable`; use that path
for object compilation. Invocation-entry/system-driver selection is checked
around queries and before returning a key, so entry symlink retargeting fails
closed even when the original canonical driver remains unchanged.
Clang driver regularity is checked before reading ELF bytes, so a named FIFO
that satisfies executable lookup is rejected without blocking on its writer.

Clang's `-M` quoting differs from GCC: it normalizes literal header backslashes
to `/` and may leave filename tabs unquoted. Collection cross-checks actual
opened-header paths with `-E -H` and replays that trace. Backslashes, control
characters, terminal spaces and mismatched trace paths are unsupported even if
normalized or split decoy files exist. This does not relax the GCC grammar.
Spaces inside paths, `#` and `$` retain real compiler/object dependency tests.
The traced set must exactly match the depfile headers. A successful
`__has_include` probe that never opens/includes its header appears only in Clang
`-M`; it is deliberately non-reusable (`clang-header-trace-mismatch`). This
also rejects ambiguous unopened probes with real normalized/split decoys.
Ordinary include-consumed paths remain supported. Completeness is still the
caller assertion; the trace is an identity cross-check, not a snapshot producer.

## Completeness boundary

GCC `-M` includes system headers as well as nested user headers. The collector
parses its fixed-target make rule, hashes each resolved file, repeats dependency
discovery, and rechecks source, compiler, header and manifest bytes. Path parsing
preserves ordinary literal backslashes and decodes GNU make's odd backslash runs
before whitespace. GCC can emit terminal filename backslashes indistinguishably
from escaped spaces or line continuations. These ambiguous spellings fail closed,
even when a merged decoy pathname exists: escaped whitespace followed by a slash
or another unescaped whitespace, even backslash runs before whitespace, and
continuations without a preceding unescaped separator are rejected. This also
excludes
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

For admitted Clang jobs the frontend and assembler are integrated in the driver,
so executable coverage requires that driver instead of standalone GCC companions.
The caller's snapshot must still cover loaded libraries, resource files, loader
and include namespaces, opaque assembler inputs and all relevant configuration
state. Disabling default configuration narrows the supported domain; it does not
establish manifest completeness. Resource-directory and job-selection changes
are bound to context identity and checked again during collection.

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

Clang recipes transport the unsalted closure through
`-fdebug-compilation-dir=generativeqc-closure-<sha256>`, an accepted Clang command
option. This collector does not accept debug flags, so the option does not add
debug metadata or change scientific C expressions. It supplies command identity
for opaque inputs; the selected cache must honor it. Real qualification changes
a manifest-covered `.incbin` input with identical C source/preprocessor output and
checks different actual object bytes through the same verified cache launcher.
Clang does not reuse GCC's `-frandom-seed` policy.
An uncacheable `.incbin` safe fallback proves fresh object bytes, but does not
prove a cache hit or the salt's hashing behavior. Qualification must also exercise
cacheable ordinary C: unchanged command replay hits, while a refreshed opaque
manifest changes the salt and causes a miss under the same verified launcher.

## Verification

```bash
PYTHONPATH=python python -m pytest -q tests/python/test_bulk_aot_closure.py \
  tests/python/test_bulk_aot_clang_closure.py \
  tests/python/test_bulk_aot_gxx_closure.py \
  tests/python/test_bulk_aot_cache.py tests/python/test_bulk_aot_store.py
python tools/check_compiler_structure.py
```

On Linux the real-GCC tests require GCC and a usable compiler-cache launcher;
missing tools fail the integration test. They independently compare raw GCC
dependency output and file hashes, compile the same source between collections,
exercise verified store reuse/invalidation, and inject a header mutation between
real dependency scans. Non-Linux runs skip those integration tests. No GPU or XC
numerical acceptance is claimed by these toolchain tests.

Linux Clang tests also require a real system Clang. They compare `-M` observations
with a separate cached object compilation's `-MD` depfile and `-H` input trace,
publish and retrieve that object through the existing verified store, and execute
a small polynomial density fixture against an independent Decimal reference.
This fixture qualifies the stored object's toolchain/reuse behavior, not Libxc,
installed production bundles, runtime cache completeness or CUDA endpoints.

Native Linux G++ tests require the real system driver and a verified compiler
cache, with no missing-tool skips on Linux. A C++ standard-header fixture is
compiled between matching collections, independently checked against its compile
depfile, published and read from the existing immutable store after deletion of
the original object, and executed against a Decimal oracle. Other controls cover
ordered flags, source/header/configuration/ABI identities, opaque assembler inputs,
unchanged compiler-cache replay hits and changed manifest seed misses. Non-Linux
skips are protocol-only evidence and cannot replace this qualification.
