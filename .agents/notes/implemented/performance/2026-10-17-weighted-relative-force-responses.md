# Decision: default relative-coordinate responses for weighted Direct forces

Status: implemented; complete-endpoint qualification passed
Date: 2026-10-17

## Problem

The qualified relative/density order-six/seven stack leaves approximately
4.63 seconds of aggregate low-order (zero through five) device work in the
retained 96-atom warm profile. This is not an isolated order-five measurement.
These workers already precontract arbitrary AO weights and share primitive
geometry and Boys preparation. Repeating that optimization is not useful.
Their remaining nuclear-center AD independently expands equivalent responses.

## Decision

Differentiate the existing weighted scalar in the Gaussian separation frame
`A-B`, `C-D`, `P-Q`, then apply the constant exponent-fraction Jacobian to
recover physical nuclear derivatives. Only `P-Q` differentiates the Coulomb/Boys
ladder. Exponents and externally supplied weights stay fixed. Physical positive
Gaussian exponent fractions define the frame. This is a generated AD schedule,
not a new ERI recurrence, molecule specialization or density-fitting substitute.

The generic builder retains its original nuclear-coordinate default and also
accepts an explicit relative derivative basis. Generated strict FP64 roots are
emitted alongside unchanged nuclear/FP32 functions. Native workers retain
64-component partitions and the original `ssss` root, which has fewer scalar
operations than its relative rewrite. Long-range and FP32 consumers are unchanged.

Before publishing any atom contribution, validate all accumulated relative
responses for every source. A nonfinite result retries the original nuclear
FP64 worker once, without a partially published force. Prepared owners capture
the arithmetic selector independently of Fock policy. The rollback control is
`GENERATIVEQC_DIRECT_WEIGHTED_FORCE_RESPONSES=0`; unknown nonempty values also
retain nuclear roots. A new owner is required when changing a captured selector.

## Qualification contract

Relative responses are enabled when unset after complete-endpoint qualification.
The same candidate binary is compared with explicit zero against genuinely unset
selection, retaining the already qualified order-six/seven defaults. A separate
frozen original image detects cold loading cost hidden by same-image comparison.
Preserve canonical basis/grid, strict FP64, screening, convergence, five warm and
moved-warm calls, independent arrays, and 1800-second population caps.

Initial host checks compare arbitrary signed weights and coincident centers
against retained raw-component AD and independent displaced Hermite values.
Order-four/five tests independently check both response graphs, including every
bounded component partition. Failure injection verifies no partial publication
for separate/combined J/K, both spins and repeated/distinct atom mappings.

Static scalar-IR counts fall from 9849 to 6750 for the first `dpds` partition
and 10485 to 6973 for the first `ddps` partition; `ssss` grows from 72 to 75 and
is deliberately excluded. These are compiler counts, **not hardware instructions,
executed FLOPs, or measured endpoint speedups**. Primitive-pair admissions and
screening do not change. These static counts were not the promotion criterion;
the subsequent GPU and complete-endpoint evidence qualifies the default.

### Preliminary real-GPU evidence, not endpoint acceptance

Slurm 9941 ran a frozen generated-root probe on node1/RTX 5090, using 4096
actually executed primitive/source calls per launch and 100 CUDA-event-timed
warm launches. Both variants use the same eight physical Gaussian geometries
and signed cotangents; input data are intentionally cached in this microprobe.
Two repetitions retain the same qualitative result:

| Root | Nuclear device time | Relative device time | First-run change |
| --- | ---: | ---: | ---: |
| psss | 6.053 us | 4.100 us | -32.27% |
| ppps | 26.645 us | 20.479 us | -23.14% |
| dpds | 306.942 us | 200.656 us | -34.63% |
| ddps | 321.237 us | 204.668 us | -36.29% |

All output coordinates agree with retained nuclear GPU roots within
7.106e-14; this parity check is not a replacement for the independent host
oracles or complete-endpoint force gates. Memcheck, synccheck and racecheck
report zero errors/hazards for the generated-root probe. Ptxas reports lower
`dpds` stack/spill requirements: stack 7824 to 4504 bytes; spill stores 18456 to
9424 bytes; spill loads 21736 to 10696 bytes. These are compiler resource reports,
not observed global-memory traffic. Register count remains 254 for this class.

The launched binary's SASS audit finds no calls, loops or arithmetic predicates:
only an untaken exact-grid bounds exit and a dead post-EXIT sentinel branch.
Multiplying its DADD/DMUL/DFMA instruction counts by the verified 4096 active
invocations gives derived FP64 work of 70,189,056 to 46,686,208 FLOPs for `dpds`
and 74,448,896 to 46,927,872 for `ddps`, counting DFMA as two FLOPs. These are
audited straight-line execution derivations, **not hardware counter readings**.
NCU reports `ERR_NVGPUCTRPERM` on node1, even though its metric-query subprocess
returns zero. An attempted secondary node2 counter query stops before profiling
because NVML reports a driver/library version mismatch; no node2 measurement
is claimed and no system driver/permission setting is changed.

The fully linked candidate passes 12 independent Libcint cases, plus a native
memcheck and synccheck case in Slurm 9951. The unfiltered full-process native
racecheck reaches its 600-second timeout: it is **not passed**. Do not repeat
that overly broad campaign. The replacement native gate traces actual force
launches, requires the low-order production kernel to execute, and scopes
racecheck to launched force kernels. Slurm 9957 passes the replacement gate in
41.88 seconds with zero reported hazards/errors/warnings. A separate Nsight
trace proves one separate-source and two combined-source low-order production
launches in this UKS Cartesian fixed-density oracle. This is a scoped production
force-kernel race gate, not a successful unfiltered whole-process racecheck.

Complete qualification must additionally compare the frozen original library
against the candidate, not only zero/unset modes of the same new image. The
new core is 969,268,544 bytes versus 941,694,144 bytes before the change; a
same-image toggle alone cannot detect extra cold module-loading overhead.

## Complete source-matched endpoints

Freeze integration base `8165e9035114c574bef9c1efe150a81a4ef735d3`: the qualified
#2588 stack reconciled with master `743c06bf8`, not unmodified master. Original
native identity is `15d35b0bc932941f6dd82f39089afec36b5b5cacf519d7072eb6be6973fea2b8`;
candidate identity is `f7c524732fe0852a5f4a6ebc18efec6526d349c9e1103be72fb28c5b87382ac5`.
All 1889 frozen production inputs are authenticated independently of later
test, benchmark, documentation and dependency-audit changes.

Slurm9957 uses node1's assigned RTX 5090,
`GPU-45f177f9-33ac-0e4e-f473-2beeaf6d413d`, with a finite 02:30:00 allocation.
The earlier generated-root microprobe uses a different assigned card; do not
combine its kernel times with endpoint timings. Canonical spherical def2-mTZVPP,
moving unpruned Becke 48 x 16 x 32, strict FP64, original screening/convergence
and the 1800-second population cap are unchanged. PBE0/HF retain the existing
def2-SVP controls. Occupied-XC #2577 is disabled and is not composed.

Complete E+F seconds, same new image, nuclear zero versus genuinely unset:

| Direct case | Cold | Warm median of five | Moved | Moved-warm median of five |
| --- | ---: | ---: | ---: | ---: |
| r2SCAN-3c96 | 244.066872 -> 243.818824 | 30.205867 -> 29.489152 | 144.150942 -> 142.982348 | 30.284782 -> 29.499596 |
| r2SCAN-3c48 | 59.686129 -> 59.478220 | 8.057675 -> 7.719456 | 41.518604 -> 41.360098 | 8.078146 -> 7.716254 |
| PBE048 | 22.944838 -> 22.875742 | 3.927602 -> 3.850807 | 21.731207 -> 21.681610 | 3.904460 -> 3.829424 |
| HF48 | 11.032736 -> 11.057492 | 0.746025 -> 0.743083 | 7.229579 -> 7.222777 | 0.745728 -> 0.743843 |

The 96 warm/moved-warm gains are 2.373%/2.593%; the independent reverse-order
population gives 30.292204 -> 29.462275 and 30.303544 -> 29.462602 seconds,
2.740%/2.775%. Twice-summed relative MAD is 0.164%/0.459% forward and
0.260%/0.290% reverse. This is a reproducible incremental component, **not an
additional 10% win or GPU4PySCF parity**. The HF cold increase of 0.224% is
retained. No matched 48-atom control exceeds 5% regression.

Independent original-image comparisons retain actual cold loading costs:

| Direct case | Original cold / warm / moved / moved-warm seconds | Candidate seconds |
| --- | --- | --- |
| r2SCAN-3c96 | 244.223847 / 30.224710 / 152.496412 / 30.237207 | 243.818824 / 29.489152 / 142.982348 / 29.499596 |
| r2SCAN-3c48 | 59.963030 / 8.101658 / 41.464039 / 8.051703 | 59.478220 / 7.719456 / 41.360098 / 7.716254 |
| PBE048 | 22.876326 / 3.920532 / 21.705589 / 3.910077 | 22.875742 / 3.850807 / 21.681610 / 3.829424 |
| HF48 | 11.006622 / 0.742930 / 7.222093 / 0.745608 | 11.057492 / 0.743083 / 7.222777 / 0.743843 |

Original-image HF cold/warm/moved increases of 0.462%/0.021%/0.009% are retained,
not hidden by selector-only comparisons. The 96 original-image moved trajectory
is **14 iterations/14 Focks versus 13/13**; its 6.239% reduction is not an isolated
force-kernel gain. Same-image and reverse trajectories match: 96 cold24/24,
moved13/13; 48 cold19/19, moved13/13; PBE0 cold17/18, moved12/12; HF cold21/22,
moved16/17. Every native warm/moved-warm call uses 1/1. Never normalize trajectories.

All 14 complete populations, 168 calls, independently pass finite full-array
oracle checks: maximum errors 2.546586e-11 Eh and 5.697376e-11 Eh/Bohr, below
the unchanged 1e-8/1e-7 gates. No native population is incomplete. Separate
diagnostic profiles and the fresh same-card peer are not counted as these
clean endpoint populations.

The compiler/host scientific cohort passes 202 tests; the expanded allocation,
stack, lifetime and default cohort passes 2254. Structure/reproducer coverage
passes 202, including all 37 finite-population wrapper tests. Documentation
passes 22 tests/55 subtests and Sphinx warnings-as-errors. Cohorts overlap;
do not add their counts into a unique-test total. The host HF driver borrows
the shared provider's preparation declaration to capture force arithmetic;
the dependency audit permits that interface, not device/recurrence implementation.

Reproduce with `benchmarks.direct_force_reuse --weighted-responses`; the mode
removes inherited order-six/seven overrides, compares zero/unset of this
independent arithmetic selector, records the axis outside empty worker-owned
directories, and preserves bounded process-group retirement on every exit.

## Source-matched profile and actual work

Both diagnostic cold/first-warm profiles independently pass the same full-array
gates. They use 24/24 and 1/1 native iterations/Focks in both arms, respectively;
they are not additional completed 12-call populations. Profiled cold/warm wall
is 245.562949/30.457728 -> 244.225622/29.562162 seconds, not a replacement for
the clean timing cohorts. All native work-census fields match between arms,
including actual traversals and unavailable coverage; enumeration is unchanged.

Separate warm device-kernel sums, seconds:

| Scope | Nuclear | Relative |
| --- | ---: | ---: |
| SCF exact J | 4.240152 | 4.213972 |
| SCF XC/AO/grid | 3.959457 | 3.960257 |
| Generated low orders0–5 together | 4.636523 | 3.832672 |
| Order6 J' | 2.610593 | 2.607896 |
| Order7 J' | 0.490333 | 0.491967 |
| Order8 J' | 0.454112 | 0.453755 |
| One-electron pair response | 2.276319 | 2.272499 |
| Other stationary/grid response | 9.301394 | 9.302022 |
| Resident final validation kernel | 0.017567 | 0.017568 |

The low-order aggregate improves 17.337%, not an isolated order-five gain.
There is no new order-six/seven optimization or SCF/XC improvement in this
component. The candidate's other-response group decomposes without overlap:
geometry4.152341 + Becke2.485601 + GEMM1.929906 + AO radial0.508335 +
other0.225839 seconds. It is not all Becke response.

Actual warm primitive-pair traversals by orders0–8 are identical in both arms:
224278138, 392134128, 400832292, 259426880, 119559374, 38484032, 8666248,
1250544, 95976. Low orders total 1,434,714,844. These count actual consumer
entries, including retry traversals, not enumerated candidates or FP64 FLOPs.
Low recurrence-preparation and AO-contraction totals remain unavailable/null;
do not infer them from primitive or component capacity. Order6 retains
1,882,836,519 density terms and 3,153,588,800 Hermite joins; order7 retains
453,127,296 density terms, 870,721,600 joins and 4,136,704 fallback AO contractions.
Density attempts, accepts, mask declines, preparation and zero numerical
declines remain unchanged. The arithmetic reduction is separately demonstrated
by the generated-root straight-line SASS audit, not by reducing these counts.

Exclusive warm complete-wall ledgers reconcile to the measured profile wall:

| Exclusive wall owner | Nuclear seconds | Relative seconds |
| --- | ---: | ---: |
| Native SCF/publication | 8.295620337 | 8.270884857 |
| Stationary force/publication, excluding nested owners | 20.253138975 | 19.430116766 |
| Final-state snapshot | 0.605277133 | 0.588812521 |
| Final-state decode | 0.295528425 | 0.290079477 |
| Final-state validation | 0.555418068 | 0.529388513 |
| D4 | 0.445219143 | 0.445737306 |
| gCP | 0.002146751 | 0.001963812 |
| Calculator construction | 0.000668937 | 0.000664469 |
| Public wrapper | 0.004578791 | 0.004401923 |
| Driver return/final sync | 0.000131688 | 0.000111958 |
| Endpoint preparation | 0 | 0 |

Cold preparation is separately 9.084387 -> 9.119926 seconds. Explicit
`cuda{Device,Stream,Event}Synchronize` runtime-API counts remain 13756 cold and
4807 warm per arm. Their union host intervals are 103.468559 -> 102.773864 seconds
cold and 4.486878 -> 4.461192 warm. These **overlap** the scientific wall ledger;
they are not an extra summand or a complete measurement of implicit/driver API
synchronization. No-kernel device windows are 1.922241 -> 1.859647 warm and are
not labelled synchronization time. Device sums, wall ledgers, API intervals and
microprobe times are distinct views and must never be added.

Compilation uses verified ccache4.5.1 for actual CXX/CUDA commands, retaining the
existing shared cache and checkout-root normalization. Sccache is unavailable.
Before/after global cache statistics are retained, not attributed entirely to
this build. Native private helper REG:0/STACK:0 metadata is not evidence of zero
resources; the concrete generated-root ptxas reports above are the resource
comparison. No resource cap, precision, screening or convergence is relaxed.

Full arrays, traces and immutable source/cache/device receipts remain in ignored
`.artifacts/weighted-relative-low-20261017/`; lossless scalar evidence is published
with the implementation PR, explicitly not an offline force replay. No new
tracked archive, storage budget, release or tag is introduced.

## Alternatives and invariants

- Do not repeat page-width/task-enumeration experiments: they do not remove this
  derivative arithmetic.
- Do not duplicate geometry/Boys algebra in handwritten kernels.
- Preserve exact full-range J/K, source weights, derivative signs, symmetry,
  RKS/UKS, spherical/Cartesian representations and bounded fallback behavior.
- Do not infer a cold gain from static arithmetic reductions or normalize away
  changed SCF/Fock trajectories.

## References

- PRs #2587 and #2588: prerequisite qualified relative/density force stack.
- Issue #1117: broader GPU performance tracking, currently closed.
- Local ignored evidence: `.artifacts/weighted-relative-low-20261017/`.
