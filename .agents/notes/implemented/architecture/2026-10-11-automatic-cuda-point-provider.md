# Decision: static automatic CUDA point/XC execution plumbing

Status: implemented experimental native plumbing; public CUDA KS/forces remain blocked
Date: 2026-10-11

## Problem and boundary

The common automatic CPU/CUDA point source is qualified independently, but
ordinary CUDA XC/KS/forces still cannot consume an automatic registration. A
standalone scalar probe is not an XC contraction or prepared molecular endpoint.
The native executor must accept semantic/feature facts and immutable launchers,
not implement new functional-name mathematics or call a CPU/Libxc oracle.

## Decision

`dft/xc_point_cuda.py` owns the existing `DevicePointValue` ABI and compact,
channel-major physical-spin traversal. Curated global kernel names and their
three existing template arguments remain intact: the kernels delegate through
a force-inlined device helper. This avoids invalidating resource classification
merely by adding a provider template argument to their global kernel symbols.
`xc/automatic_cuda.py` binds `automatic_point.py`'s exact scientific source to
that same helper. RKS/UKS are layouts, not separate AOT mathematical variants.
Ordinary providers have no response/fxc entry; unsupported component scaling
is rejected. This plumbing does not reopen public CUDA resolution.

The CPU and CUDA generators now consume one structural registration census in
`xc/automatic_semilocal.py`. CUDA AOT emits a static library-owned
`CudaXcPointProgram` per registration with the canonical semantic identity,
exact provider-source identity, ingredient mask, feature width, work-policy
version and ordinary single/batched callbacks. RKS and UKS are runtime layouts,
not separate mathematical compilation variants. Thirty-two translation units
bound compilation-unit size; their objects are shared by the native library and
qualification driver rather than compiled twice.

`CudaXcPlan` binds the descriptor once and reuses its existing AO, density,
feature and potential contractions. Automatic descriptors do not inherit any
curated optional precision, response, local-AO, batch or graph qualification.
`prepare_point_batches` retains the bounded one-tile incumbent. Direct batched
callbacks exist for layout qualification, not schedule promotion. The source
hash is not a persistent executable-cache key: toolchain, transitive headers,
linked libraries, target and resource receipt closure still belong to the
existing AOT/artifact owners.

`GENERATIVEQC_ENABLE_AUTOMATIC_XC_CUDA_AOT` is explicitly off by default. Its
disabled registry has no scientific entries or shard dependencies. Enabling it
builds the complete structural domain, not a hand-picked positive whitelist.
The first full native executable measured about 100 MB: silently adding all
unadmitted entries to ordinary CUDA builds would prejudge #1123's bounded
packaging policy. This opt-in is experimental execution plumbing, not the final
installed-bundle solution.

## Remaining production integration

- Connect supported targets and measured resource facts to the existing native
  resource/evidence admission owners. Descriptor source hashes are insufficient.
- Qualify ordinary CUDA RKS/UKS cold/replay/changed geometry and J-only work counts
  before changing public resolution; then qualify stationary CUDA forces.
- Extend the existing `bulk_aot*` artifact/cache/ledger owners for packaging,
  executable aliases and complete toolchain/header/library closure. Do not
  create a second cache owner or collapse scientific/provenance alias identity.

## Evidence and acceptance boundaries

The first native integration snapshot (`native-v2`) compiled all 197 automatic
registrations through NVCC 12.9 with release optimization and `--fmad=false`.
Slurm n1/RTX 5090 job 7420 passed 44 independent native fixed-density XC gates
(11 registrations x both spin layouts x original/changed geometry). Each record
includes cold, exact replay and both density-direction perturbations. PySCF
2.14.0 supplies independent AO jets and Libxc 7.0.0 supplies point E/vxc. Maximum
energy, potential and directional errors were respectively
`5.551115123125783e-16`, `8.326672684688674e-16` and
`1.9537216137434366e-11`. Enqueue asserts no hidden staging/synchronization;
canaries and semantic potential counts verify the exact bounded arena/work.
This is historical intermediate-snapshot evidence, not final source binding.
The final source adds totals-stride overflow protection, explicit disabled
packaging, retained prepared identity and direct single/batched point probes.

### Initial source-matched qualification

Scientific/native source: `fad1b85fa2aafda4162e4e758d92f3f98cc38453`.
Both remote source trees were checked against the complete tracked-file SHA256
manifest before execution. The n5 build uses NVCC 12.9.86, sm_120, release `-O3`,
`--fmad=false` and verified sccache 0.16.0; the full AOT option is explicitly on.
The native executable SHA256 is
`e0b2aae8874c5b75d000259ad6fdce277195fcf0bfef2549dd60b036b75a78d7`.

- **209 host/compiler tests pass**; the 100 real-device cases are intentionally
  skipped in that host run. Public API docstrings and the compiler ownership
  checker pass (516 modules, zero dependency errors).
- **Slurm job 7423**, n1 RTX 5090, passes all **100 real-device tests**: 44 direct
  AOT E/vxc single/batched layout gates (including 37-point / 16-point partial
  tiles, vacuum, empty spin, floors and FHC), 12 negative-density rejection
  gates, and 44 independent fixed-density molecular XC receipts. All eleven
  registrations cover both spins and original/changed geometry. The fixed-XC
  error maxima equal the intermediate values above. Repeated submissions mean
  ordinary-stream physical recalculation, not CUDA Graph/KS replay admission.
- The native curated LDA/PBE/r2SCAN/WB97M-V E/V and state suite also passes in
  job 7423 after the traversal refactor.
- **Slurm job 7427** runs Compute Sanitizer memcheck on automatic MGGA_C_TPSS
  UKS at changed geometry and on the curated native suite: **zero errors**.
  Device identity: RTX 5090, driver 580.95.05; scheduler visibility is preserved.
- All **197 registrations / 394 ordinary single/batched kernels** compile in
  the 32 shared AOT objects. Generated CUDA source is **6,774,685 bytes**;
  aggregate CUDA objects are **90,876,592 bytes**; the complete test executable
  is **100,127,440 bytes**. These are size measurements, not an installed
  bundle, executable alias-reuse proof or cold-compile/endpoint speed claim.
- Register range is **48–255**, maximum stack **256 bytes**, maximum spill store
  **176 bytes**, maximum spill load **224 bytes**. **Eight variants spill**:
  both entries of MGGA_C_R2SCAN01, MGGA_C_REVTM, MGGA_C_REVTPSS and GGA_C_OPTC.
  This broad compile-only census is not full-catalog numerical or schedule
  admission; resource/evidence owners must decide future packaged variants.
- The exact current CMake grid/XC source compiles separately with release flags.
  Its five active PBE scope pressure sets (`ao_jets`, `density_product`,
  `density_features`, `xc_points`, `vxc_contraction`) are identical to the
  source-82c34b929 retained receipt; no mixed scope supplies a missing FP64 one.
  New object size **9,327,640 bytes**, binding
  `0049108cd44ffbcfaec15f02e3ed97caea0f19c99ebf8884f71f878aa2e521dc`.
- A fresh default-off CMake configuration compiles only the inert CXX registry,
  with no automatic CUDA scientific shard in that target.

Ignored artifacts live under `.cache/1118-runtime/` in the runtime worktree:
`qualified-source-manifest.txt`, `compiler-final.log`, `final-gpu/`,
`final-native/automatic-aot-resource-summary.json` and
`final-native/native-resources-final/report.json`. Remote frozen inputs/results
are n5 `qc-1118-native-v1-20261011` (compiled final bytes) and n1
`qc-1118-native-v3-20261011` (qualification). No release/tag/asset is published.

### Upstream integration and repeated qualification

PR #2255 merged as upstream `e370c490f` while the native stage was being
published. Integration source is
`fecef17bf9ca0049ecd5f3c4da1677e2733cb66b`; it retains the upstream final
point/force evidence note and this stage's common structural registry owner.
The native grid emission remains byte-identical, but other merged compiler
modules change the importer/compiler-inventory identities. The initial receipts
therefore cannot be relabeled as current or replayed with overridden identities.

A new frozen tracked-file manifest was verified on n5 and n1
`qc-1118-native-merged-20261011`. All 197 registrations / 394 kernels were
regenerated and recompiled with current identities and the same release/cache
contract. Size and pressure values remain exactly those above. New binary SHA256:
`4bc35f00f1c68c1407e6cdd823b8e2198af8978bf827d14e4393f6dfcd079a6a`.

- **209 host/compiler tests pass**, with 100 real-device cases skipped locally;
  public docstrings pass; **518 compiler modules, zero dependency errors**.
- **Slurm job 7450**, n1 RTX 5090, passes **all 100 GPU tests**, the curated
  native suite, and both automatic/curated memcheck gates (**zero errors**).
  All 44 independent native receipts match current compiler identities; energy,
  potential and directional error maxima remain exactly the initial values.
- The current release native grid resource report again has all five PBE
  pressure sets identical to the retained pre-refactor baseline. Its object is
  **9,327,640 bytes**, binding
  `5b938e4c3f454656f0040c517077f45a4234ff36c399d42ba16078ca531123c1`.
- A fresh default-off CMake configuration still compiles only the inert CXX
  registry, not automatic CUDA scientific shards.

Current ignored evidence is `.cache/1118-runtime/merged-gpu/`, `merged-native/`,
`compiler-merged.log`, `structure-merged.log`, `docstrings-merged.log` and the
`merged-source-*` archive/manifest. The earlier source-bound evidence is retained
as history, not overwritten. Public CUDA KS/force and packaging/promotion gates
remain open; this requalification does not upgrade their claims.

Source/host tests protect one ABI/traversal owner, deterministic identities,
continued public rejection and curated capability behavior. Preserve curated
scientific/response paths with numerical/resource checks after the traversal
refactor. Native fixed-density evidence is not SCF, analytic force, public
capability promotion, installed artifact reuse or a complete endpoint timing.

## Debugging invariants

- The automatic point's gradient parameter is an array reference. A provider
  forwarding function must preserve `const double (&gradient)[2][3]`; spelling
  it as an array parameter decays it to a pointer and fails real NVCC compilation.
- Bound launch counts by both feature channels and the three scalar-total
  channels before converting to the native signed index ABI.
- A mixed CXX/CUDA AOT object library needs CUDA include usage requirements on
  its CXX registry dispatcher too; link its compile interface to CUDA::cudart.

## Rejected shortcuts

Do not duplicate the scalar point algebra, hand-write promoted mathematical
branches, put XC imports into the DFT owner, assume a passing point probe admits
a schedule, or silently stage production CUDA science through CPU/PySCF.

## References

- #1118, #1119, #1121, #1122, #1123 and #1124.
- PR #2255: common scientific source and physical spin-screen normalization.
- `tests/python/test_cuda_xc_point_contract.py`.
- `tests/python/test_libxc_automatic_cuda_xc.py`.
