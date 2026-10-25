# Calculation-time compilation inventory

This inventory describes the README energy plus analytic-force paths. Routing
and module families were inspected at source commit
`bbd41a42764e8d22d15c1348f70a633961d4dcf5`; the
[canonical source publisher][source-publication] defines the byte contract below.
This is source inspection, not a compilation-count, first-use latency, GPU execution or
scientific qualification result. `NOT_OBSERVED` means no calculation-time
compiler invocation was found in the traced route; it does not certify every
build, optional backend or future route as JIT-free. Measured compile counts and
times for these endpoints are **UNAVAILABLE** here.

The [generated method reference](../public_methods.md),
[r2SCAN-3c contract](../user/r2scan3c.md),
[WB97M-V contract](../user/wb97mv_cuda.md) and
[CBS contract](../user/ccsdt_cbs.md) own scientific admission. A method name,
packaged kernel or cache hit does not admit an otherwise unsupported force,
basis, spin, precision or resource domain. This page identifies compilation
owners, not a second method-capability ledger.

## Public routing and module families

`Calculator.singlepoint` uses `PreparedBatch`; the latter asks the native
**prepared item** for its supported properties. It runs native full forces when
that item supports them, otherwise its admitted Python compatibility consumer.
An older library without the query retains compatibility routing; a query error
is not silently downgraded. See [public routing][batch] (`execute`,
`_native_dft_force_eligible`, `_public_dft_cuda_force`) and the
[native capability owner][dft-method] (`native_cuda_stationary_binding`,
`admitted_cuda_stationary_binding`, `add_prepared_native_forces`).

| Family | Calculation-time work and trigger | Compilation owner / result |
| --- | --- | --- |
| N: native energy and native complete force | Prepared HF/KS SCF, integrals, DF, XC, response and admitted complete stationary-force controllers execute linked code. Native KS force eligibility checks the actual model, shape, precision, grid, Fock provider and bindings; it excludes fitted, range/nonlocal and D4 plans in the inspected CUDA binding rule. | Build-time generated/native C++ and CUDA in `libgenerativeqc`; calculation-time compiler **NOT_OBSERVED** in this traced route. [HF][hf-method], [DFT][dft-method], [generated build][generated-build]. |
| P: packaged ordinary CUDA stationary force | When native full force is not selected, an all-electron electronic MethodIR with an exact packaged profile loads the matching s/p or s/p/d stationary artifact. Matching includes full exchange composition and spin, not just an XC integer. | [Stationary loader][stationary-compiler] `load_stationary_aot_artifact`; native AO/grid and integral-derivative bridges remain separate. No NVCC discovery on this packaged public route. A declared missing, stale or corrupt package fails closed; it does **not** fall back to JIT. |
| J: generated ordinary CUDA stationary force | An admitted compatibility force without packaged plan coverage, or the ECP route, generates bounded primitive derivatives plus stationary AO/geometry/Becke contractions. Large admitted native-integral domains can reduce the generated primitive demand to nuclear repulsion; they do not remove the stationary wrapper. | [Stationary compiler][stationary-compiler] `compile_stationary_cuda` writes primitive and wrapper `.cu` files, calls `compile_cuda_object`, then `link_cuda_objects`. NVCC compiles PIC relocatable objects and links a shared library on artifact misses. |
| C: composite CUDA stationary force | WB97M-V's admitted range-exchange plus VV10 force uses native integral/RSH derivatives, native AO/grid and resident VV10 owners, plus generated nuclear and semilocal geometry/Becke stationary work. A different nonlocal partition-iteration count selects another stationary wrapper identity. | [Composite consumer][composite] calls `compile_stationary_cuda`; its primitive/wrapper/object/link cache is J. Unchanged prepared identities reuse owners; changed identities close and rebuild them, which can still hit persistent artifacts. Current final source assembly is `_canonical_gradient_sum` on already host-resident gradients, **not** an additional TensorIR CUDA reduction compile. |
| T: extra generated tensor CUDA programs | The ordinary ECP compatibility route plans separate local/nonlocal ECP contractions and final reduction. These are conditional additions to J, not ordinary all-electron P requirements. | [Stationary consumer][stationary-consumer] `_plan_stationary_cuda_tile`, `_tensor_execution` → [TensorIR][tensor-execute] `compile_cuda` → NVCC shared compilation on a tensor artifact miss. |
| G: stand-alone generated CUDA grid | Diagnostic callers that supply no native grid library can compile a generated grid executor. The inspected public CUDA force routes supply the native library instead. | [Stationary consumer][stationary-consumer] chooses `compile_grid` only when `native_grid_library is None`; [grid compiler][grid-execute] owns emission/runtime compilation. Do not count this diagnostic option as a mandatory public endpoint compile. |
| D: separate r2SCAN-3c corrections | Public construction separates canonical D4 and gCP from the electronic graph; the correction batch adds their energy/force contributions. | [Public constructor][calculator] and [dispersion owner][dispersion]. Native D4 and CPU gCP execution is explicit; calculation-time compilation **NOT_OBSERVED** in these correction consumers. CPU gCP is part of this declared composite contract. |

The source name `compile_ks_execution_plan`, for example, denotes mathematical
planning, not an NVCC subprocess. Similarly, emitting primitive sources or
constructing a TensorIR reduction program alone is not compilation.

## README CUDA paths, Direct and DF

Each row covers the named energy call and its additional analytic-force work.
DF means the admitted fitted primary with its required auxiliary/metric
derivatives. Range-separated DF can retain an exact long-range correction;
it must not be described as all-range fitted exchange. See the
[density-fitting contract](../user/dft_density_fitting.md).

| Method | Primary | Energy compilation | Additional force compilation / selection |
| --- | --- | --- | --- |
| HF | Direct | N; **NOT_OBSERVED** | N; native HF derivative provider, **NOT_OBSERVED**. |
| HF | DF | N; **NOT_OBSERVED** | N; native fitted response/metric derivatives, **NOT_OBSERVED**. |
| PBE | Direct | N; **NOT_OBSERVED** | Native full force when admitted; otherwise P (`pbe`, both spins) for all-electron, J + T for admitted ECP compatibility. |
| PBE | DF | N; **NOT_OBSERVED** | P (`pbe`, both spins) for admitted all-electron compatibility, with native DF derivatives. Native Direct full-force eligibility must not be transferred to DF. |
| PBE0 | Direct | N; **NOT_OBSERVED** | Native full force when admitted; otherwise P (`pbe0`, both spins) for all-electron, J + T for admitted ECP compatibility. |
| PBE0 | DF | N; **NOT_OBSERVED** | P (`pbe0`, both spins) for admitted all-electron compatibility, plus native DF derivatives. |
| B3LYP | Direct | N; **NOT_OBSERVED** | Native full force when admitted; otherwise P (`b3lyp`, both spins) for all-electron, J + T for admitted ECP compatibility. |
| B3LYP | DF | N; **NOT_OBSERVED** | P (`b3lyp`, both spins) for admitted all-electron compatibility, plus native DF derivatives. |
| r2SCAN | Direct | N; **NOT_OBSERVED** | Native full force when admitted; otherwise P (`r2scan`, both spins) for all-electron, J + T for admitted ECP compatibility. |
| r2SCAN | DF | N; **NOT_OBSERVED** | P (`r2scan`, both spins) for admitted all-electron compatibility, plus native DF derivatives. |
| r2SCAN-3c | Direct | N + D; **NOT_OBSERVED** | P (`r2scan`, both spins) for the admitted canonical all-electron electronic graph; separate D correction gradients. Canonical s/p/d basis does not by itself imply JIT in this source. |
| r2SCAN-3c | DF | N + D; **NOT_OBSERVED** | P for the admitted electronic graph, native fitted derivatives, then D. This is still a gated complete composite. |
| M06-2X | Direct | N; **NOT_OBSERVED** | Native full force only under actual model/shape qualification; otherwise P (`m06_2x`, both spins) for admitted all-electron compatibility. Other domains require separate admission. |
| M06-2X | DF | N; **NOT_OBSERVED** | P (`m06_2x`, both spins) for admitted all-electron compatibility, plus native DF derivatives. |
| WB97M-V | Direct | N; **NOT_OBSERVED** | C: NVCC primitive/wrapper/object/link misses; native RSH and VV10 derivatives. No ordinary P profile. |
| WB97M-V | DF | N; **NOT_OBSERVED** | C with the admitted mixed fitted-primary/exact-range source and native DF response; same generated-module family, different scientific/source identities. No implicit switch to exact HF/Direct. |

The P assignments can be checked without loading a native library or probing a
GPU: resolve each MethodIR for each spin, apply the public constructor's
correction separation for r2SCAN-3c, construct its `StationaryGradientPlan`,
then call `stationary_aot_profile_for_plan` with its native semilocal code.
The [generated profile inventory][profiles] and
[profile manifest][profile-manifest] are the coverage authorities. Coverage
does not prove the installed package contains a valid matching artifact.
Actual P loading additionally authenticates target architecture, plan/domain,
precision, partition iterations, manifest/source identities and binary bytes.

The issue's earlier `7522994` description of s/p/d r2SCAN-3c JIT and a generated
WB97M-V final reduction must not be generalized to this inspected source.
Use the current selectors above. No historical README latency is reclassified
as a fresh empty-cache measurement by this inventory.

## CBS complete energy plus forces

[CCSDTCBSCalculator.singlepoint][cbs] groups equal orbital-basis identities and
calls `Calculator.singlepoint_components`. It does not independently compile
a CBS formula or run a Python scientific reference as its production backend.
The following are distinct models, not interchangeable fallbacks:

| Requested reference/correlation model | Admitted backend | Traced compilation boundary |
| --- | --- | --- |
| Conventional exact RHF + conventional MP2/CCSD(T), no auxiliary records | CPU or CUDA in their admitted basis/derivative domains | Native HF, MP2 and RCCSD(T) grouped execution and analytic response; runtime compiler **NOT_OBSERVED**. |
| `hf_reference="exact"` with explicit auxiliary records: exact RHF + correlation-only DF | CUDA | Native exact reference, `correlation_cuda` MP2 groups and `df-ccsd(t)` correction group; runtime compiler **NOT_OBSERVED**. |
| `hf_reference="df"` with explicit auxiliary records: full RI-JK HF and DF correlation | CUDA | Native `full_cuda` reference, same-operator orbital response, three-center/metric derivatives and grouped correlation; runtime compiler **NOT_OBSERVED**. No exact-RHF fallback. |

Call chains continue through [MP2][mp2-method], [RCCSD][rccsd-method],
[RCCSD(T)][rccsdt-method] and [DF-CCSD(T)][df-ccsdt-method]. Fast changes the
admitted convergence/preconvergence policy, not this compilation ownership.
CPU DF, open-shell, ECP and frozen-core CBS E+F must not be inferred from the
conventional CPU row; follow the [CBS model contract](../user/ccsdt_cbs.md).
Legacy CUDA RI-MP2's energy-only path is not a complete CBS E+F model.

## CPU compatibility consumers

CPU energy and native complete-force routes use N. If the public admitted KS
force instead selects `_public_dft_cpu_force`, its
[CPU stationary consumer][cpu-consumer] constructs `CppCompilerAdapter` from
`CXX` (default `c++`). Native fitted primary derivatives still require a
generated nuclear-only consumer. Direct component execution selects packaged
s/p/d derivative dispatchers when available, otherwise bounded generated
primitive consumers. The [streaming component owner][cpu-streaming] also
checks the native component ABI before choosing its packaged executor; absent
coverage uses `compile_runtime`. The same compatibility consumer also prepares
generated [TensorIR weight/contraction/reduction programs][cpu-tensor],
[XC geometry contractions][cpu-xc] and [Becke grid contractions][cpu-grid];
each calls `compile_runtime` on an artifact miss. Range plans add their
[range exchange][cpu-rsh] derivative consumer and its
[range value owner][cpu-range]. Nonlocal pairs use the native provider;
the CPU geometry/grid chain can additionally prepare a nonlocal-grid Becke
contraction when its partition iterations differ. These paths are not a
claim that every CPU DFT E+F call compiles, nor a CPU fallback for CUDA refusal.

The C++ adapter uses Unix-style shared/PIC flags and the runtime artifacts here
are `.so` libraries. CUDA identities query NVCC/PTXAS and a host compiler;
the sccache private-worker implementation uses POSIX process/lease facilities.
Source inspection and mocked Windows host tests do not establish that these
JIT consumers execute on Windows or another unqualified platform.

The canonical source publisher encodes the caller's string as UTF-8, compares
existing raw bytes, and publishes through a binary stream without locale or
newline conversion. Explicit CRLF, mixed newlines and Unicode remain part of
the caller's source identity. The semantic cache validates the corresponding
byte lengths and hashes on fresh-process replay. Pre-existing translated or
otherwise malformed entries remain rejected and unchanged; the fix does not
silently repair or clear them. The
[byte-publication decision](../../.agents/notes/implemented/compatibility/2026-10-23-generated-source-byte-publication.md)
records the historical Windows failure, its resolution and host-test limits.
These source-byte contracts do not establish Windows native JIT support.

## Cache layers and toolchain requirements

The experimental [integral AD owner](../user/experimental_array_api.md#differentiate-through-gaussian-integrals)
is a separate opt-in compilation route. Its RHF/UHF CPU and CUDA variants
generate integral/normalization parameter derivatives and stationary source
weights. Second order uses primal scalar instruction tapes with generated local
AD rules. `compile_runtime` compiles the `.cpp` or `.cu` artifact on a miss;
the identity includes the contraction and tape runtime headers. Native SCF,
orbital response, explicit-integral J/K and eigensolver execution use the linked
library. This route does not change the ordinary method rows above.

The [build guide](../developer/build.md#compiler-cache-generated-shards-and-compile-units)
owns launcher precedence, minimum versions and private-worker lifecycle. The
following separates the endpoint artifact owners from that build contract.

| Layer / owner | Location and identity | Hit versus miss |
| --- | --- | --- |
| Live prepared owners | `PreparedBatch` / stationary execution objects; scientific/topology, target, schedule, workspace and live-state identities | Reuse admitted live owners; a changed geometry/identity can rebuild ownership without compiling if persistent artifacts hit. Snapshot grids, AO maps, DF factors and resident value caches are numeric caches, not compiler artifacts. |
| Packaged AOT P | Beside the loaded native library; [stationary loader][stationary-compiler] manifest and authenticated binary | Load matching packaged code with no JIT compiler. Invalid/missing declared package fails; clearing a writable JIT directory does not remove installed AOT. |
| Python semantic source cache | `<stationary-cache>/semantic-sources/<recipe-key>/`; [source-cache owner][source-cache] records recipe, dependency closure and UTF-8 source hashes | Valid hit skips deterministic emission/IR/AD. It does **not** skip the downstream binary identity check. Corrupt entries fail integrity checks. `GENERATIVEQC_STATIONARY_SOURCE_CACHE=0` disables this reuse in the ordinary stationary consumer, not the binary cache or compiler launcher. |
| Stationary CUDA objects and linked runtime J/C | `GENERATIVEQC_STATIONARY_CACHE`, default **working-directory-relative** `.cache/stationary-cuda`; source `.primitive.cu` / `.stationary.cu`, keyed `object.o` and linked `runtime.so` directories with `artifact.json` | [Native-runtime cache][native-runtime] binds source/header bytes, target, flags, toolchain/host facts and selected compile environment; link identity also binds object/library inputs. Valid hit verifies binary hashes and skips compilation/link; absent entry invokes adapters and atomically publishes. Corruption fails closed. |
| CPU stationary / generic runtime artifacts | Same explicit stationary override, CPU default `.cache/stationary-cpu`; generated `.cpp` and keyed `runtime.so` directories | `compile_runtime` binds source/headers, compiler identity and target, flags, libraries and relevant environment. Bundled multi-unit compilation has its own complete-source identity. Hit validates bytes; miss uses the C++ adapter. |
| Tensor T artifacts | Caller-provided stationary cache; keyed `program.cu`, `program.so`, `static.bin`, `plan.json`, `artifact.json` | [Tensor compiler][tensor-execute] binds plan/lowering/dependencies, compiler facts, generated source and static-data bytes. Binary and static data are verified on hits; misses call NVCC. No tensor compile is implied by composite host summation. |
| Compiler cache underneath adapters | Shared build sccache/ccache stores are distinct from Python artifacts. Runtime sccache leases private persistent `slot-N` stores under `$XDG_CACHE_HOME/generativeqc/sccache-jit` (cache home defaults to `~/.cache`); `GENERATIVEQC_JIT_SCCACHE_ROOT` overrides the pool. | A Python artifact miss can still hit a compiler cache. Private JIT slots must not overlap the shared sccache tree. Sequential calls can reuse a slot; concurrent calls lease separate stores. Neither shared global counters nor a warm AOT build prove endpoint cache state. |

[CudaCompilerAdapter][cuda-adapter] → [run_cached_compiler][cache-launcher] and
[CppCompilerAdapter][cpp-adapter] → `run_cached_compiler` own finite compilation
subprocesses. The public compatibility CUDA adapter sets a 600-second invocation
timeout; CPU adapter construction retains its 300-second default. These are
per-invocation deadlines, not a complete endpoint or total multi-module compile
budget. Caller resource/work admission remains mandatory. A cache miss must
find a verified sccache (0.16.0+) or ccache; an older/unusable sccache can fall
back to verified ccache. Launcher identity is execution provenance, not an
extra scientific artifact-key input. Artifact hits do not resolve the launcher,
but **may still require the actual compiler toolchain**: public C/J routing
finds NVCC first, and native-runtime/TensorIR caches compute toolchain and host
compiler facts before lookup. Thus “no compiler-cache launcher needed on hit”
does not mean “toolkit-free JIT artifact replay.”

An artifact's stored `compile_seconds` describes its original build. Replaying
that metadata does not measure compilation in the current process. Source
lookup/generation spans and inclusive artifact lookup/compile spans must not be
added as independent endpoint time or treated as exact compiler subprocess time.

NVCC discovery is [explicitly implemented][provenance]: `GENERATIVEQC_NVCC`,
`CUDACXX`, `CUDA_PATH/bin/nvcc`, then `PATH`. Toolchain identity also queries
the sibling `ptxas`; object identity uses `NVCC_CCBIN` or discovered `gcc` for
host facts. Actual compatibility remains the selected toolkit's obligation.
No NVRTC invocation was observed in these traced public call chains; NVIDIA
runtime/provider DSOs, driver loading or PTX handling are not evidence of an
application NVRTC compile. Optional autotuning, extensions, Array API and
Hessian/HVP consumers are outside this E+F inventory.

## Verification and remaining measurements

Existing executable contracts include
[launcher selection/miss wrapping][test-launcher],
[semantic source reuse/invalidation][test-semantic],
[packaged force routing and missing-package refusal][test-aot], and
[native artifact integrity/launcher-free hits][test-native-cache]. Mocked
compiler tests validate routing/cache behavior, not device execution or latency.
Keep independent binary/static-data integrity and ABI tests; no permanent
function-body hash, AST shape or exact internal compile-count test is needed
to maintain this inventory.

Completing this inventory still requires fresh-process, source/binary-bound
empty/primed artifact **and** compiler-cache measurements, actual compile
counts/time where available, and the corresponding independent accuracy gates.
Those measurements must record
installed-wheel versus build-tree loading, actual selected routes, cache roots
and overrides, private JIT slots, compiler facts and endpoint work. Retain
UNAVAILABLE counters rather than filling them with zero. This inventory supplies
none of those timing or scientific acceptance results.

[batch]: ../../python/generativeqc/batch.py
[calculator]: ../../python/generativeqc/calculator.py
[dft-method]: ../../src/methods/dft_method.cpp
[hf-method]: ../../src/methods/hf_method.cpp
[generated-build]: ../../cmake/GenerativeQCGeneratedSources.cmake
[stationary-compiler]: ../../python/generativeqc_compiler/method/stationary_cuda.py
[stationary-consumer]: ../../python/generativeqc/_stationary_cuda.py
[composite]: ../../python/generativeqc/_stationary_composite_cuda.py
[tensor-execute]: ../../python/generativeqc_compiler/tensor/cuda_execute.py
[grid-execute]: ../../python/generativeqc_compiler/dft/cuda.py
[dispersion]: ../../python/generativeqc/dispersion.py
[profiles]: ../../python/generativeqc_compiler/method/_generated_stationary_aot_profiles.py
[profile-manifest]: ../../manifests/stationary_aot_profiles.json
[cbs]: ../../python/generativeqc/cbs.py
[mp2-method]: ../../src/methods/mp2_method.cpp
[rccsd-method]: ../../src/methods/rccsd_method.cpp
[rccsdt-method]: ../../src/methods/rccsdt_method.cpp
[df-ccsdt-method]: ../../src/methods/df_rccsdt_method.cpp
[cpu-consumer]: ../../python/generativeqc/_stationary_cpu.py
[cpu-streaming]: ../../python/generativeqc/_stationary_cpu_streaming.py
[cpu-rsh]: ../../python/generativeqc/_stationary_rsh_cpu.py
[cpu-range]: ../../python/generativeqc/_stationary_range_cpu.py
[cpu-tensor]: ../../python/generativeqc_compiler/tensor/cpu.py
[cpu-xc]: ../../python/generativeqc_compiler/xc/native.py
[cpu-grid]: ../../python/generativeqc_compiler/xc/grid_native.py
[source-cache]: ../../python/generativeqc_compiler/common/semantic_source_cache.py
[source-publication]: ../../python/generativeqc_compiler/common/source_cache.py
[native-runtime]: ../../python/generativeqc_compiler/common/native_runtime.py
[provenance]: ../../python/generativeqc_compiler/common/provenance.py
[cuda-adapter]: ../../python/generativeqc_compiler/common/cuda_adapter.py
[cpp-adapter]: ../../python/generativeqc_compiler/common/cpp_adapter.py
[cache-launcher]: ../../python/generativeqc_compiler/common/compiler_cache.py
[test-launcher]: ../../tests/python/test_compiler_cache.py
[test-semantic]: ../../tests/python/test_stationary_semantic_cache.py
[test-aot]: ../../tests/python/test_stationary_aot_no_compiler.py
[test-native-cache]: ../../tests/python/test_native_runtime_cache.py
