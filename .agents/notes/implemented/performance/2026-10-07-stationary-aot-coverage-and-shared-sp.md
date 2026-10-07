# Decision: exact stationary AOT coverage and shared small-domain primitives

Status: implemented
Date: 2026-10-07

## Problem

At base revision `2b0feff1d5577e58f3a44fbecbc1974be92df4e3`, the public
force selector excluded every global hybrid before testing existing exact-plan
profiles. PBE0/B3LYP therefore used JIT despite their RKS/UKS package entries.
The s/p/d primitive objects were shared across profiles, but s/p packages still
regenerated and compiled the same primitive inventory in each monolithic TU.

## Decision

Expose a non-loading exact-plan catalog query. Only an unrepresented plan may
choose bounded JIT; represented plans proceed to the existing fail-closed
loader. Query the actual MethodIR, point code and spin, not the method label or
the mere presence of exchange. Domain, partition, precision, target and binary
provenance remain loader/executor obligations.

Generate and compile one separable s/p primitive object, then link it into each
small specialized wrapper, using the established s/p/d ownership pattern. Keep
the legacy single-TU emitter for inspection and byte-equivalence tests. Build-time
manifests verify both source halves and hash their combined provenance; runtime
must not emit source or AD to load AOT. The hashed compiler contract changes, so
old manifests are deliberately rejected rather than trusted across this change.

The previous compatibility hash itself called `integral_block`, recreating
TensorIR/AD even for a cold binary hit. Replace that generated graph hash with a
complete source closure including method and tensor compiler modules. Record
the graph hashes at manifest-writing time, validate their source inventory on
load, and reuse them in runtime work evidence. Extract the existing reduction
coverage gate so native reductions validate identical source sets without
constructing unused reduction IR. This keeps the scientific formulas intact.

Expose a deterministic bounded CMake profile subset, defaulting to the existing
catalog. Do not multiply the imported Libxc catalog by every derivative/spin/
schedule. An explicit empty subset removes this stationary AOT inventory.

## Rejected alternatives

- Blanket AOT for all hybrids: changed exchange weights and unrepresented
  compositions have different scientific plans.
- Catch loader failures and JIT: this hides broken package provenance and restores
  an unintended runtime-toolchain dependency.
- Deduplicate by point code: PBE and PBE0 both use code 1, with different plans.
- A single branching XC megakernel: not needed for primitive reuse and would
  require separate CUDA resource and numerical qualification.

## Invariants

Exact scientific plan identity stays distinct from primitive compilation
ownership. Generated scientific expressions, primitive dispatch, FP64/no-FMA
policy, integral ownership, active-AO policy and bounded small-system fallback
stay unchanged. Linking a shared object into multiple DSOs saves compiler work,
not necessarily binary bytes. A profile subset is a packaging policy, never
scientific promotion or permission to compile silently in ordinary execution.

## Evidence

Host tests exercise PBE/PBE0/B3LYP public routing in both spins and s/p/d domains,
alias versus changed exchange weights, missing package errors, source-equivalent
split wrappers, forbidden primitive regeneration, source-half corruption and
deterministic CMake subsets. Existing loader tests cover plan, target, precision,
domain and binary checks. Real-GPU endpoint and large-system interleaved timing
evidence must be recorded separately; source sharing alone is not a speedup claim.

At this implementation stage, the focused host suite passes 255 tests and the
compiler ownership audit reports 487 modules with zero dependency errors.
Cold-loader tests clear the compatibility cache and forbid integral/reduction
IR and wrapper generation for PBE0/B3LYP in both spins and primitive domains.
The frozen capacity endpoint-owner checksum is re-pinned for this metadata/
coverage-only change; numerical limits, complete-source requirements, allocation
budgets and source-accounting gates are not relaxed. The independent hybrid
oracle preserves the requested AO representation, including five-function d
shells, rather than silently comparing a spherical endpoint to Cartesian SCF.

Final combined host validation: 633 passed and 61 explicitly gated cases skipped
in 105.32 seconds, using ccache for standalone C++ probes. The eight new GPU
cases are among the skips, not numerical qualifications. Native/CUDA ownership,
SCF/electronic-structure, vendor/provider, default-promotion and high-order
complexity audits pass. Reproduction commands and raw logs are retained locally
under `.artifacts/`; hardware qualification remains a separate explicit gate.

Node n2 CUDA 12.9 qualification build: all eight PBE0/B3LYP RKS/UKS s/p and
s/p/d DSOs link successfully and pass the no-GPU package provenance/native-cubin
audit. The selected inventory contains exactly one shared s/p primitive object
and 23 shared s/p/d objects (25,395,072 object bytes). Shared emitted sources
occupy 4,904,417 s/p bytes and 23,002,437 s/p/d bytes. The eight linked artifacts
total 173,393,440 binary bytes and 173,412,009 bytes including manifests. These
are bounded-subset compile/footprint counts, not full-catalog scaling or endpoint
speedup evidence. Binary/source/contract receipts, compiler-cache statistics and
an offline cuobjdump resource census are retained in ignored `.artifacts/`.
No register/resource eligibility is inferred from successfully linking a DSO.

The complete current-source native build reaches its explicit 1,500-second
deadline before producing the core runtime library. Therefore the new real-GPU
energy/force/reuse/displacement gate is not executed or claimed passed; do not
substitute a mixed-revision native core to promote this path. Keep the PR draft
and both issue qualification lanes open. Incremental staging with `rsync -a`
initially copied changed source bytes with timestamps older than the manifests;
the audit correctly rejected stale contracts. Advancing the changed input's
mtime and rerunning the real CMake manifest rules produced matching receipts,
without clearing compiler caches or editing expected identities to obtain hits.

## Consequences and revisit conditions

For N packaged profiles, the s/p primitive compile count becomes one instead of
N, alongside the unchanged shared 23-shard s/p/d inventory. Wrappers and device
links remain proportional to selected profiles. Further sharing of AO/grid/
Becke/scheduling bodies and executable parameter-alias reuse needs independent
ABI/resource work. Issue #1123 still requires broad catalog/object census,
dependency-closure integration and hardware numerical/resource gates; this is a
bounded production packaging slice, not closure of that entire lane.

## References

Issues #2077, #1123 and #2071; `docs/developer/stationary_cuda_diagnostic.md`;
`tests/python/test_stationary_aot_generation.py`;
`tests/python/test_stationary_aot_no_compiler.py`.
