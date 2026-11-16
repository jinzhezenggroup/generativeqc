# Decision: share exact workers across native CUDA response packages

Status: implemented; full wheel and real-device endpoint qualification pending
Date: 2026-11-16

## Problem and inventory

The follow-up to `2026-11-15-uccsd-cuda-exact-kernel-sharing.md` surveyed eight
other generated CUDA translation units. They repeated node kernels across
analytic response graphs, contraction-provider variants and precision/batch
variants. Exact emitted-code equivalence can eliminate duplicate compilation
without deleting any method, reducing semantic work or changing execution order.

The comparison parent is `a278661ba`. All counts below cover complete translation
units, including their existing hand-written generated accumulation kernels:

| Unit | Kernel definitions before | After | Launch sites, unchanged | CUDA bytes before | After |
| --- | ---: | ---: | ---: | ---: | ---: |
| RCCSD | 3,203 | 1,174 | 3,537 | 3,017,317 | 2,083,974 |
| Triples response | 597 | 73 | 597 | 502,416 | 239,804 |
| Triples Fock response | 117 | 36 | 117 | 108,797 | 64,977 |
| DF actions | 550 | 183 | 550 | 507,511 | 353,604 |
| DF core | 322 | 173 | 322 | 294,111 | 236,715 |
| DF hoisted | 471 | 227 | 471 | 503,926 | 418,863 |
| DF occupied pairs | 124 | 69 | 112 | 130,291 | 115,528 |
| DF AO | 13 | 11 | 13 | 18,356 | 18,448 |
| **Total** | **5,397** | **1,946** | **5,719** | **5,082,725** | **3,531,913** |

The occupied-pair emitter already elides proved copy roundtrips, so its definition
and launch counts were not equal before this change. Keep that existing proof
and schedule unchanged. DF AO removes just two definitions; long authenticated
symbols slightly increase its small source while reducing compiled duplication.

## Decision and ownership

- Reuse the existing `CudaKernelPool`, keyed by the complete emitted signature
  and body after parameterizing only the entry name and diagnostic node number.
  Its key preserves dtype, dimensions, indexing, coefficients, contraction and
  reduction policies, scalar fallback and all finite/domain checks.
- Require an explicit C++ identifier for each translation-unit owner. Several
  response units use the same C++ namespace. A body hash alone would give their
  equivalent kernels identical external names and produce duplicate symbols.
  Owner qualification is execution/build provenance, not different mathematics.
- Keep the UCCSD owner spelling `shared`; its previously qualified generated
  source remains byte-identical. Migrate every constructor caller directly; no
  obsolete zero-argument compatibility facade is retained.
- Pool the local RCCSD response graphs, but leave its original iteration kernel
  inventory unchanged. Preparation/reused/prepared-contraction runners borrow
  that inventory through explicit `kernel_prefix`/`emit_kernels=False` ownership.
  Mixing these contracts with local pool ownership remains an error, rather
  than silently emitting another inventory or changing phase-error retention.
- Preserve the DF packed/scalar/batched and mixed-precision variants. Optional
  providers, capacity admission, original error resets/retention and bounded
  scalar fallbacks remain unchanged. Different executable bodies do not merge
  merely because they have the same shape or algebraic family.

## Validation and evidence

A one-off migration proof compares every new worker body against the complete
parent inventory and checks all 5,719 corresponding launch configurations and
operands, allowing only entry-symbol substitution and the original diagnostic
node argument. It also compares all remaining host orchestration, including
provider dispatch, arena planning, work counts, error resets and phase ownership.
This is retained investigation evidence, not a permanent source-fingerprint gate.

Permanent regression coverage separately compiles and links identical generated
arithmetic in two translation units sharing a C++ namespace, then executes both.
Invalid owner identifiers fail before source emission. The scalar-provider audit
follows actual native ABI entry points to their reachable kernel paths, so it
still checks both packed/scalar primary branches and strict independent replay
without requiring private worker names or a specific deduplication count.
Existing executable numerical, arena-guard, singular/nonfinite, invalid-index
and caller-specific diagnostic tests remain enabled.

The focused compiler/provider/DF/response suite passes 95 tests; 71 skip under
the existing backend/explicit-qualification gates. The compiler structure
check covers 604 modules with zero dependency errors. Those skips are not GPU
qualification, and compilation must not be described as physical CUDA execution.

### Complete compiler-object qualification

All eight before/after CUDA objects compile with NVCC 12.4.99, verified sccache
0.16.0 and the existing shared cache. The eight new objects also pass one host
relocatable link together, exercising real generated symbol ownership in addition
to the standalone regression test. Complete-object observations:

| Unit | Before seconds | After seconds | Object bytes before | After |
| --- | ---: | ---: | ---: | ---: |
| RCCSD | 164.92 | 107.16 | 14,350,024 | 8,932,360 |
| Triples response | 34.83 | 21.10 | 5,517,832 | 1,253,088 |
| Triples Fock response | 12.44 | 10.09 | 1,179,104 | 494,368 |
| DF actions | 30.43 | 20.45 | 5,627,696 | 2,875,944 |
| DF core | 24.30 | 19.02 | 3,894,544 | 2,598,112 |
| DF hoisted | 30.71 | 24.97 | 5,180,600 | 3,512,376 |
| DF occupied pairs | 13.44 | 12.41 | 1,373,184 | 1,089,368 |
| DF AO | 5.84 | 5.75 | 254,616 | 240,928 |
| **Total** | **316.91** | **220.95** | **37,377,600** | **20,996,544** |

These are one serial pair per complete unit, not isolated benchmark repetitions;
CPU regression tests overlapped part of the run. Treat wall times as local
observations, not an uncontended speedup estimate or whole-wheel timings. The
16 NVCC object requests produce 64 CUDA/device/PTX/CUBIN cache misses and zero
CUDA hits. Shared statistics also contain three unrelated C++ probe misses;
per-command snapshots preserve that distinction. No cache was cleared or forced.
The total object size decreases 43.8%, independent of elapsed-time noise.

Both variants use identical arguments:

```text
-std=c++20 -O3 --fmad=false --expt-relaxed-constexpr
-gencode arch=compute_89,code=sm_89
-gencode arch=compute_89,code=compute_89
-DGENERATIVEQC_HAS_CUDA=1
-DGENERATIVEQC_HAS_CUTENSOR=0 -DGENERATIVEQC_HAS_CUBLASLT=0
-DGENERATIVEQC_HAS_CUTLASS=0
-I <checkout>/src -I <checkout>/include -I <shared-generated-headers>
-c <variant-source> -o <variant-object>
```

These are local compile-only comparison settings, not changes to wheel optional
providers or the `80;90-real;120-real` architecture inventory. Header inputs are
shared between variants. `/usr/bin/time` reports the sccache client's RSS, not
peak memory of daemon-owned compiler workers; do not quote it as compiler RSS.

A second RCCSD/triples-response pair, after CPU tests ended, adds only
`--resource-usage`. It correctly reuses 12 CUDA/device/PTX frontend cache entries
and recompiles four CUBIN stages; **it is not another cold-build comparison**.
Matching the actual ptxas entries shows no register increase among 3,203 original
RCCSD entries (three decrease), with maximum 48 unchanged. Of 597 triples-response
entries, 33 increase by one to four registers, 36 decrease by one, and 528 stay
equal; maximum 40 remains unchanged. These compiler resource observations do not
prove unchanged whole-method runtime performance or occupancy on other targets.

## Remaining costs and rejected alternatives

Do not merge all generated sources into a giant translation unit or replace
scientific programs with lower-precision/CPU/reference work. All method inventory,
architectures, numerical flags, launch work and whole-wheel acceptance gates
remain intact. Complete compiler endpoints, not a sampled kernel, are the unit
of compilation qualification; no runtime method speedup follows from these counts.

The next distinct candidate is the generated CPU evaluator/header boundary.
UCCSD's 402,844-byte CPU header contains 315,645 bytes of five evaluator bodies;
its 2,796,775-byte response header contains 2,349,642 bytes of 46 evaluator bodies.
The CUDA unit currently parses both headers even though it needs metadata and
output types, not those CPU evaluators. Separating that interface requires its
own dependency, shape/work-metadata, compiled-consumer and endpoint validation;
this change does not claim to remove that remaining cost or diagnose runner OOM.
