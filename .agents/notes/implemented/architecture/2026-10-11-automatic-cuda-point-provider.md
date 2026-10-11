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
