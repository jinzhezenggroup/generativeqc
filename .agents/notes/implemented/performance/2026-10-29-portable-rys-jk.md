# Decision: retain generated J/K coverage in the untuned CUDA profile

Status: implemented
Date: 2026-10-29

## Problem

The [ECNU portability qualification](../compatibility/2026-10-29-ecnu-a100-portability.md)
made the A100 usable and removed model-based Becke admission, but the portable
profile still contained no generated integral kernels. Complete warm PBE0/def2-SVP
energy-plus-force calls consequently recomputed canonical AO-component integrals.
On the 48-atom water cluster, one warm Fock action executed 451,133,882 recurrence
evaluations. CUDA tracing attributed approximately 2.307 s to J/K, 0.300 s to XC,
and 1.087 s to direct integral forces in a 5.296 s endpoint. Python bookkeeping
contributed only about 0.0034 s. A separate Nsight timeline corroborated the
kernel attribution. This was missing generated coverage, not evidence of an
incorrect energy or force.

## Decision

Populate `portable_cuda` with the existing value-only Rys compiler programs for
the 20 canonical s/p/d classes before `dddd`. Mathematically eligible low-order
classes use four independent warps of lane-local packed tasks; remaining classes
use target-derived component schedules. Reuse the current scientific IR,
recurrences, scatter ABI, screening and resource validation. Do not add handwritten
scientific CUDA or GPU-model checks.

The profile remains portable and untuned. It is a conservative shared execution
policy, not an A100-specific tuning record. Exact SM120 profile resolution is
unchanged. Preserve the native `dddd` value fallback: generated `dddd` does not
have the required independent qualification. Derivatives and uncovered classes
also retain their existing implementations. The portable primary already uses
packed tasks where eligible; it does not inherit the tuned profile's optional
Rys-task K-family preference.

Set the portable profile's `optional_fock_variants` build policy to false.
The primary generated programs remain present; unused Rys-value, Rys-task and
restricted K-block alternatives no longer inflate their CUDA modules. The option
defaults to true for existing profiles and can be selected independently of GPU
identity or tuning status. Exact SM120 selections and optional inventory remain
unchanged. Source generation remains deterministic and requires no device probe.

Generated availability alone must not veto profitable cold reuse. Ordinary
strict hybrids through 128 Cartesian AOs retain automatic resident-source
admission, using the existing small-HF size domain and all existing screening,
cold-provenance and complete-capacity gates. Larger generated hybrids, pure J,
application-budget refinement and explicit source controls retain their policy.
This is a size/work policy shared by CUDA targets, not an architecture allowlist.

The fixed-density libcint qualification now accepts the actual compiled target,
profile and external manifest explicitly, and accepts real Slurm or LSF GPU
allocations. This avoids testing a hard-coded SM120 inventory against an SM80
artifact. The endpoint profiler accepts the same schedulers. Neither change
alters assigned device visibility. The existing capacity qualifier also follows
the capability-based Becke admission rather than its former SM120-only premise.

## Rejected alternatives

Simply enabling component-lane generated Rys coverage passed numerical gates but
regressed the 48-atom diagnostic cold endpoint from 22.210 s to 42.689 s, while
warm improved only from 5.296 s to 4.897 s. Forcing the optional Rys-task K family
improved those to 33.964 s and 4.365 s, still a cold regression. Generated coverage
suppresses automatic cold resident-ERI caching; conservative component kernels
did not recover that lost reuse. Do not promote either warm-only improvement.

Packed low-order primary J/K coverage improved the same diagnostic to 17.028 s
cold and 3.365 s warm without changing cache policy. The clean timing campaign
is separate from these intrusive traces. Copying SM120 schedules and calling
them A100 tuning was not considered acceptable provenance.

The first complete packed-only cohort exposed small cold regressions: 3 atoms
1.330 to 1.426 s and 12 atoms 3.073 to 3.470 s. The small cold-reuse correction
above was added after this evidence; do not relabel the packed-only ABBA receipts
as measurements of the final cache policy.

Cold tracing then isolated about 0.12 s of first-use overhead in generated J for
the 3-atom case; a subsequent J took about 0.006 s. Its SCF/XC work and force
components were otherwise comparable. The initial packed build emitted unused
alternative families into the same CUDA modules. Disabling those families reduced
the first complete call from about 1.42 s to 1.36 s, close to the 1.34 s control,
without changing the primary kernel mathematics. Retain the Nsight cold timeline
and explicit module/build inventories; do not move initialization outside the
endpoint timer or call a warm-only improvement a cold improvement.

## Qualification and evidence

Task-local raw evidence is retained in `.artifacts/ecnu-a100-gap-20261029/`.
The finite ECNU LSF allocation uses `gpu_chem`, node `gpu22`, one exclusive A100
PCIe 40GB, eight CPU slots, CUDA 12.9.86, GCC 14.4 and verified ccache. The native
GPU UUID matches the earlier native qualification. No concurrent build or GPU
test runs during timing cohorts.

The clean protocol retains cold setup plus energy/analytic forces, three warm
calls, a moved geometry and three moved-warm calls for 3/12/48/96 atoms. The
packed-only pilot uses two fresh processes per arm in ABBA order. The final
small-system check uses four ABBA rounds (eight fresh processes per arm and
24 warm samples per geometry); final 48/96-atom confirmation uses two fresh
processes against the retained two current-control processes on the same GPU.
Frozen
GPU4PySCF/PySCF references, basis and matched quadrature remain those of the
earlier qualification. Acceptance gates remain 1e-8 Hartree energy and 1e-7
Hartree/Bohr maximum force error. Warm SCF/Fock counts and AO work are retained;
missing recurrence counts for the generated path must not be reported as zero.

The fixed-density gate covers every generated primary class against independent
libcint integrals: separated/coincident/reversed centers, RHF/UHF, combined J/K
and separate J/K/HF-K consumers, task counts 1/33/129, and ordinary/compensated
output. It does not use the native recurrence as its oracle.

All 352 selected final/reference-control endpoint records passed. Maximum errors
were 1.0687e-11 Hartree and 3.1529e-11 Hartree/Bohr. All 2,880 fixed-density
libcint cases passed on the final library, including coincident/reversed centers,
both spins, all four consumers and both accumulation modes. Matrix gates are
`atol=2e-11, rtol=2e-10`, with the independent expected matrix scaled by the
submitted task count.

Complete endpoint medians, comparing the previous PR implementation with this
final implementation, are:

| Atoms | Previous cold (s) | Final cold (s) | Previous warm (s) | Final warm (s) |
| --- | ---: | ---: | ---: | ---: |
| 3 | 1.319917 | 1.335392 | 0.101487 | 0.087628 |
| 12 | 3.058159 | 3.068018 | 0.499689 | 0.472015 |
| 48 | 22.130968 | 17.039359 | 5.243538 | 3.361453 |
| 96 | 69.625773 | 45.083262 | 18.835222 | 10.948880 |

Small cold medians remain +15.5 ms/+9.9 ms (+1.17%/+0.32%); report these rather
than claiming a cold win. Sample ranges overlap. A deterministic 20,000-resample
bootstrap of the median difference gives 95% intervals of [-10.1, 37.6] ms and
[-21.8, 30.5] ms, respectively. These limited samples do not resolve a stable
small cold regression. The much larger +0.10/+0.40 s pilot regressions were not
promoted. This uncertainty is not a guarantee of identical startup on other
machines.

Warm reductions are 13.66%, 5.54%, 35.89% and 41.87%; moved-warm reductions are
13.16%, 4.91%, 35.67% and 42.00%. Warm calls use one Fock in every arm and retain
the same semantic AO counts (discovery timing is allowed to differ). Small cold
Fock counts match at 13/19. The 48-atom cold solve changes 18 to 17 Focks; final
96-atom cold solves use 17 and 18 versus 18 in controls. Moved 96-atom solves use
12 versus 13 Focks. Preserve those counts: the whole moved/cold speedup is not
an isolated kernel measurement.

The qualified binary's embedded identity equals its exact retained source inventory:
`056a57493efb119dfbdbcdb90b4fd5400e89c1024392897b279cc92f826ff2f2`.
Earlier trials used an external manifest, whose hash and CMake setting must be
combined with the recorded base-source and binary identities. Final primary
generated source is byte-identical to the packed trial's primary source; only
optional families and the separately qualified small cold policy differ.

Before publication, the remote PR branch acquired a separate device-ordinal
normalization fix and stronger capacity regressions. Integration preserves them.
The sole production-inventory difference from the GPU-qualified source is
`python/generativeqc/experimental/integral_autodiff.py`: retain the return value
of `checked_integer` so NumPy integer ordinals reach the CUDA driver as ordinary
integers. Its real driver-boundary host test covers both scalar types. All J/K,
Becke and ordinary benchmark endpoint sources remain unchanged. The integrated
inventory is `b2ae0550b2e95eff39850166b2ad4a2e8e01ba8ba6e4f9df5142ee7605420d83`;
`integrated-source-identity.json` records the exact one-line production delta.
Do not claim that the integrated whole-source inventory is byte-identical to
the earlier qualified binary's source inventory.
The post-integration host run passed 858 capacity, device-selection and native
budget checks; 1,230 optional device-campaign cases were skipped. Integration
pre-commit checks passed. The remote report-provenance helper and additional
thread/shared-memory/grid-limit negative controls are preserved.

Final raw archive SHA256:
`e7f993df80180ddf42e4e2f6ffe3c7123e1442f2d83349e3b4c4adf29bdfc42d`.
Local host validation includes 411 capacity/profiler checks; 205 profile/Rys
checks with one optional skip, four additional malformed-flag cases; 53
codegen/CMake/cost checks; and 10 cold-source policy checks. Pre-commit, compiler
dependency and CUDA ownership checks pass. Source and binary receipts, build
graphs, compiler-cache statistics and failed/rejected trials remain local.

## Scope and remaining work

Matched-grid regression checks do not establish grid-converged production
accuracy or an overall ranking against GPU4PySCF. The retained reference's
96-atom warm samples take 28.968/12.644/6.784 s with 29/10/3 SCF cycles, whereas
native frozen warm calls use one Fock. The reference also uses a different
physical A100 on the same node and CuPy contractions without cuTENSOR. Do not
compare only its fastest sample or attribute the whole difference to kernels.

Direct integral derivatives and grid/stationary forces remain substantial costs.
Revisit them with independent force gates and complete endpoint evidence. Other
untuned architectures get resource-legal generated coverage, but this campaign
does not claim measured speedups on GPUs that were not executed. Retain bounded
fallbacks and truthful tuning metadata when expanding the generated inventory.
