# Decision: separate source reuse from execution-ABI reuse

Status: implemented (source identity and build-image preservation); additional
Stationary/DF kernel merging is not implemented or qualified.
Date: 2026-10-27

## Problem

PR #2835 separated automatic XC point producers from force consumers. Its
394-to-6 consumer reduction follows exact source, weight, reduction, spin and
point-layout equivalence. Curated Stationary profiles and DF shell triples need
their own mathematical/ABI analysis; their inventory sizes are not proof of an
equivalent reduction opportunity.

The audited baseline is `a16443baced95f5434c31b74cd993c970c39927b`.
The retained [census and validation](../../../../benchmarks/results/aot-reuse-audit-20261027/README.md)
record the compiler input digest and dirty-worktree status explicitly.

## Measured boundaries

### Stationary

With both legacy and native curated AOT enabled, the 64 CUDA translation units
are one shared s/p primitive inventory, 23 shared s/p/d primitive shards,
14 legacy s/p wrappers, 14 legacy s/p/d wrappers and 12 native wrappers.
WB97M-V's additional source contract excludes its two profiles from the latter
inventory. This count excludes #2835's automatic point/force package.

Primitive lowering/compilation already has shared ownership. Linking primitive
objects into several DSOs does not imply a corresponding installed-byte saving.

Exact staged geometry source comparison, without deleting functional constants
or normalizing emitted identifiers, finds three groups:

- LDA: LDA RKS/UKS;
- GGA: PBE, PBE0 and B3LYP, each RKS/UKS;
- meta-GGA: r2SCAN, M06-2X and WB97M-V, each RKS/UKS.

These sources include the AO jet pullback and local Becke partials. Their
functional independence assumes an external `DevicePointValue` producer.
Including actual spin blocks, ordered runtime/external sources, mathematical
weight program hashes and reduction hashes separates all 14 profiles again,
even before distinguishing s/p from s/p/d dispatch. Thus geometry equivalence
does not justify aliasing the existing full wrappers or removing the automatic
profile admission guard in `native_stationary_shared_contract`.

Factoring geometry further would require a new geometry execution interface
carrying the correct source offsets and response inputs, plus compatible
producer precision/admission. It must retain ordinary, cooperative and phased
Becke behavior. #2835's partition-response failure demonstrated why a prepared
XC point does not imply a prepared partition seed.

### DF shells

The 80 ordered classes are orbital angular momenta 0..3 at A and B, with
auxiliary angular momentum 0..4. Their rectangular raised-A caches request
2,100 axis-polynomial entries. The union contains 100 power triples and 100
distinct emitted polynomial expressions, using the existing Gaussian-moment
algebra. All 80 ordered cache layouts remain distinct.

The native templates already share geometry/Boys and contracted derivative
logic. Their host `DfShellDispatch` signature does not erase:

- Cartesian component ordering and full/triangular weight indexing;
- the raised-A cache and raised-B displacement identity;
- Boys maximum order, including the auxiliary-g F11 derivative domain;
- moment versus qualified Rys routes, root counts and shared state;
- three schedule variants, FP32/strict retry and screening eligibility;
- scattering A/B derivative slots and recovering C by translation.

Orbital exchange has 50 unordered angular orbits, but this is only an adapter
design opportunity. A kernel alias requires a proved weight permutation and
derivative restoration, with work/resource counters retained. Neither 80-to-50
nor 2,100-to-100 is a qualified kernel or endpoint reduction.

### Architecture identities

Stationary and DF already generate a single source tree for a fatbin build.
Across separate Stationary builds, however, source recipes unnecessarily bound
the target even though both emitters take no target-dependent mathematical or
schedule input. Object/link recipes correctly retain the target, compiler,
flags, headers and binary hashes.

Direct-shell lowering cannot generally discard its target: a force Rys plan
may derive a Fock companion from target-dependent `default_schedule`, and target
resource validation still applies. Identical resolved lowering inputs would be
required before introducing shared direct-shell emission.

## Implemented changes

1. Version the integral and stationary wrapper source recipes to v2. Remove
   CUDA target from those source identities only. Optional integral target
   metadata remains request provenance. Keep every exact scientific selection,
   dependency hash, precision policy, domain and shard ABI.
2. Use one wrapper source-cache owner for JIT and AOT. Explicitly include
   primitive declaration, reduction ownership and staged/inline point mode.
   The optional `GENERATIVEQC_STATIONARY_AOT_SOURCE_CACHE` CMake path shares
   primitives and scientific wrappers across builds. Default AOT behavior stays
   uncached; native binding metadata and installed manifests are not cached.
3. Keep full recipe dependencies on the CMake generation edges. A cache hit can
   skip imports that formerly appeared in a depfile; this must not hide an
   invalidating input from the incremental build graph.
4. Group direct-shell source/profile identities by numeric SM while retaining
   all requested compile images. `120-real;120-virtual` previously overwrote
   the first entry; it now preserves both regardless of order. Unsuffixed
   `120` subsumes both image requests.
5. Provide `tools/audit_aot_reuse.py` as an opt-in census. Its source digests are
   diagnostic evidence, not permanent CI fingerprints or dispatch authority.

Existing atomic publication, byte bounds, corruption rejection and deterministic
concurrent-publisher checks remain owned by `semantic_source_cache`. No new cache
framework, mathematical algebra or native allocation owner was introduced.

## Evidence and limits

The one-off baseline/candidate comparison covers 64 scientific source records:
24 primitive units, 28 legacy wrappers and 12 native scientific wrapper bodies.
All 45,970,400 bytes are preserved by per-record byte counts and SHA256 equality.
The native outer binding's compiler-provenance identity can change deliberately;
this is not a claim that complete native registration files are byte-identical.

Regression coverage includes separate-process cross-target hits with IR/AD
forbidden, AOT output directories and unchanged-file timestamps, corruption,
method/domain/dependency/ABI invalidation, bounded publication, CMake dependency
selection and image-list permutations. Real NVCC 12.9.86 object and shared-library
builds verify distinct SM80/SM120 identities and binary hits. A CMake probe uses
verified sccache 0.16.0; `cuobjdump` finds both SM120 ELF/SASS and PTX after a
combined real/virtual request.

No force kernel, public capability or numerical expression changes. No real GPU
was used. This work makes no complete-endpoint speedup, installed-size saving or
additional kernel-count reduction claim. New kernel execution sharing would
need independent force oracles, full endpoint/work evidence, precision/resource
gates and lifecycle/failure qualification through Slurm.

## Revisit when

Revisit geometry extraction when an explicit producer/geometry-consumer ABI
can remove substantial duplicate compilation without changing response dataflow
or regressing complete endpoints. Revisit DF orbital symmetry when the ordered
weight/derivative adapter has independent permutation and translation evidence.
If either source emitter acquires target-dependent scheduling, include the
resolved schedule in its recipe before allowing cross-target hits.

## References

- [#2835](https://github.com/jinzhezenggroup/generativeqc/pull/2835)
- [Current build configuration](../../../../docs/developer/build.md)
- [Stationary source-cache contract](../../../../docs/developer/stationary_cuda_diagnostic.md)
- [DF shell contract](../../../../docs/developer/df_shell_derivatives.md)
