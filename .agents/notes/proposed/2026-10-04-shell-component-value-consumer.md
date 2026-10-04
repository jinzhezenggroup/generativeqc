# Experiment: share high-order value sources across Cartesian shell components

Status: proposed; default off, device-qualified, three-atom endpoint regresses
Date: 2026-10-04

## Motivation and ownership

The [measured canonical domain](2026-10-04-canonical-shell-component-reuse.md)
contains 487,996,523 order-5--8 primitive/component products at TZVPD12, versus
2,921,528 primitive products counted once per physical shell quartet. Existing
32-lane ordering exposes only a 1.373 common-preparation reuse bound. These
counts justify a bounded physical-shell consumer, not a claimed 167x speedup.

`GENERATIVEQC_CANONICAL_COMPONENT_VALUES=shell` (alias `1`) admits an optional
canonical value route for orders 5--8. The native owner prepares angular-bucket
shell pairs and geometry-only screened row prefixes using the existing CUB
sort/scan helpers. Conservative shell keys are maxima of the exact resident
Cartesian AO bounds. Every component still applies the original AO product
predicate. Equal shells and equal shell pairs retain triangular component
ownership; independent full J, full K and range K orbit scatters are unchanged.

The compiler emits a reusable FP64 geometry/Hermite/radial source using the
existing recurrence and six-index contraction. One CTA walks a physical shell
quartet in tiles of at most 128 components. Its leader prepares all component
Hermite rows and the order-specific scalar auxiliary; lanes consume immutable
shared storage and accumulate primitives in registers. Barriers protect every
overwrite, including tail/masked lanes and sequential radial operators. Each
contracted AO value is scattered once. This first implementation specializes
the total order, with through-f coefficient bounds; it does not claim new exact
shell-class recurrence algebra or optimized cooperative radial production.

When the independently qualified `GENERATIVEQC_CANONICAL_RSH_VALUES=shared`
storage is also present, high-order full/LR values share geometry/Hermite data
and consume separate radial roots sequentially. Otherwise ordinary full/range
calls each use component sharing. No full-minus-LR subtraction is introduced.
Explicit reachable-recurrence or Hermite-convolution **value** selections retain
priority and decline this experiment; their force-only selections do not. Other
orders, mixed J, public-AO fallbacks, generated complete owners and all derivative
consumers retain their old routes. Environment selection is read only during
preparation and belongs to checkpoint/resource identity.

## Actual work and memory contract

Let P_q be primitive products, A_q admitted AO components, and T_q the number of
nonempty 128-slot tiles in the physical shell quartet's canonical component
capacity. With radial count R, the old preparation count is proportional to
`R sum_q P_q A_q`; new geometry/Hermite preparation is `sum_q P_q T_q` for a
joint call, radial preparation is `R sum_q P_q T_q`, and primitive-component
contractions remain `R sum_q P_q A_q`. Separate calls repeat geometry/Hermite.
Sparse admission is not compacted within a shell, so T_q is **not** assumed to
equal ceil(A_q/128). New shell-prefix lookup, AO checks, full Hermite rows,
union radial states, barriers and changed load balance all cost work.

For P retained shell pairs across a batch of B systems, explicit device metadata
is `88 P + 4(7B+1) + W` bytes, where W is the actual CUB workspace query. This
includes retained preparation scratch, not just final row pointers. The plan
charges it after earlier canonical, force and optional range owners. Budget or
allocation decline releases this group and retains every earlier owner. Host
vectors and their capacities are included in preparation/retained accounting.
No whole AO-quartet list or four-index integral tensor is allocated.

The compiler source has `6144 + 8 choose(L+4,4) + 64` bytes: 7,216 / 7,888 /
8,848 / 10,168 at L=5/6/7/8. This is shared per CTA, not per thread. The native
validity flag, compiler frames, registers, spills and linked resources are extra
and not yet measured. A serial leader may itself become a bottleneck; shared
storage can limit occupancy. Neither source bytes nor preparation ratios imply
endpoint speedup.

Borrowed, null-by-default counters record shell visits, geometry/Hermite
primitive preparations, radial preparations and primitive-component contractions.
The old canonical radial counter retains its contracted-AO meaning; its candidate
counter now includes final AO checks under conservative shell maxima. Candidate
counts therefore need not equal the old exact AO-row prefix domain. Admitted
radial counts must agree. Missing molecular force work/FLOPs remain unknown.

## Evidence and minimum promotion gates

The n2 ccache-backed host selection passes 734 cases, including the
emitted shared producer with several AO components and successive Full/LR/SR/Full
root overwrites, independent Gaussian Laplace/Wick integrals, diffuse/repeated/
translated centers and adjacent arithmetic/allocation suites. Used coefficient
values match the retained independent-per-component recurrence bitwise in the
non-FMA host build. This does not establish CUDA synchronization or performance.

`--component-values-only` requires actual source reuse and independently
screened full/LR/SR matrices in Cartesian/spherical bases, both spins, batch
offsets, original/displaced geometries, signed contracted primitives, output
masks, empty domains and frozen selection. It also injects a final-allocation
resource failure and requires all earlier bytes/owners and matrices to survive.
Qualification must run isolated and joint-range variants, memcheck/synccheck,
checkpoint gates, then same-binary original/displaced cold and five warm calls
at 3 and 12 atoms and a larger admitted point. Complete E/F acceptance remains
1e-8 Eh / 1e-7 Eh/Bohr. No default, speedup or parity is promoted.

Node3 is excluded from builds and GPU jobs. CPU/CUDA compilation uses verified
ccache on n2; device qualification uses finite Slurm on permitted nodes. Earlier
#1839/#1842 observations and #1845's 3-atom regression retain their own source
and GPU identities and are not added to this unmeasured candidate.

## First linked-resource observation

The frozen implementation is `8283c4f42338ba48a3c3f444c3308bafaffce734`, with
production source identity
`b3fe154ee64a20a81e0bf092e234f86a1f440a5e44f87cddb21dd1591d471793`.
The n2 sm120 build completes with 453 verified ccache compiler commands. Its
library SHA-256 is
`fc54990ccda8be7d5274885ca2eda8e5e8f03b342f020c10a025bb8feed49a95`.
The retained cache-stat window contains 408 hits and 45 misses; no cache was
cleared or disabled.

`cuobjdump --dump-resource-usage` on this linked library reports all sixteen
component kernel variants (four orders, two spins, separate/joint ranges).
Their static stacks are 512 B/thread, registers are 155/156/162/160 at
orders 5/6/7/8, and shared storage is 8,241/8,913/9,873/11,193 B/CTA. The
same library's corresponding canonical Cartesian kernels report static stacks
of 4,136/6,200/7,160/9,064 B/thread and 174--185 registers, with no explicit
shared storage. This comparison uses one linked binary, not another board's
calibration. Reported LOCAL=0 does not establish zero dynamic local-memory
traffic or spills, and these static resources do not establish achieved
occupancy or endpoint speedup.

Exact build/cache receipts, binary hashes, raw resource text and the strict
sixteen-variant extraction are retained under
`.artifacts/shell-component-values-20261004/receipts/`. The extracted source
workspace formula above excludes native/compiler storage and consequently is
smaller than the linked shared-storage result. Finite n1 Slurm 5753 begins
independent device/sanitizer qualification of the verified deployment; its
completion and complete-endpoint results are still pending.

## Synchronization qualification and retained failures

The first linked implementation failed n1 job 5753 after the retained values,
range-force and optional-allocation checks passed. Memcheck 5754 reported an
illegal instruction in `component_jk_kernel<6,false,false>` at offset 0x1a180;
synccheck 5755 reported divergent threads at the post-consumer block barrier,
including masked tail lanes in the smaller allocation fixture. A normal pass
of that smaller fixture did not qualify its synchronization.

Revision `93cd0dfd2ee94fc396a60ccfc9c4fa0d9775fb39` added an inline
`__syncwarp()` before the shared-source block handoff. It still failed native
5758 and synccheck 5759, at the corresponding order-six offset 0x1a210.
The linked SASS showed a NOP at the inserted inline warp join. Do not repeat
that inline-only fix or describe the initial matrix passes as sanitizer passes.

Revision `64780b7355f628892886d5e08c32efe7c38ca34b` retains the warp/block
handoff in a `__noinline__` device helper. Its production source identity is
`149c7e55fa02f6a08eca1ad973b60305a3804612f7a57d05e61e03b4c35b3fa6`;
the library SHA-256 is
`935e13e4aa7b52e4fbf52a047f4fd03c4bb5e0825743826618155c3f8a0d0dd1`.
The isolated n2 build again verifies 453 ccache compiler commands, with a
retained statistics-window delta of 451 hits and two misses. Linked SASS now
contains `WARPSYNC.COLLECTIVE`, `WARPSYNC.ALL` and the block barrier inside
the helper. This observation does not on its own establish a compiler defect
or prove a correct device schedule.

On n1 Slurm 5760, this revision passes the retained-value check and both
separate/joint component fixtures, including allocation rollback. Each fixture
reports 26,424 geometry preparations, 32,296 radial preparations and 2,590,104
primitive-component contractions in its counted separate calls; the optional
joint matrix calls are checked but not included in those totals. Retained force,
sanitizer and checkpoint completion remain pending at this observation.

The corrected linked component kernels retain 512 B/thread static stacks and
the original shared-byte counts. Registers are now 155--157 / 156--158 /
162--164 / 160--162 at orders 5/6/7/8 across their spin/range variants. These
remain static resources, not dynamic traffic or a timing result. Failed logs,
corrected build receipts and final SASS are retained separately under
`.artifacts/shell-component-values-20261004/`, `sync-v2/` and `sync-v3/`.

Job 5760 subsequently completes successfully on n1's assigned device 2. Both
separate and joint modes pass memcheck, synccheck and initcheck with zero
errors; retained full-range values and short/long-range derivatives pass; all
nine selected checkpoint-policy cases pass. The qualification receipt binds
these checks to the source identity and library hash above. The initial two
failed revisions remain unqualified. Complete endpoint performance and the
molecular source-reuse census are separate gates and are not implied by these
device checks.

## Molecular work census after qualification

Finite n1 Slurm 5762 evaluates the frozen 12-atom, 232-spherical-AO full-TZVPD
input with identity density, using the same qualified library for retained,
component and component-plus-joint calls. This is a resident value census,
not SCF, forces or an endpoint timing experiment. The diagnostic explicitly
requires joint storage in the joint mode.

Both retained full and LR calls admit 407,065,289 contracted AO quartets.
Component calls retain exactly those radial admissions while checking
418,127,294 candidates beneath conservative shell maxima. Each separate call
executes 1,354,668 high-order shell visits, 5,308,640 geometry/Hermite and
radial preparations, and 487,996,523 primitive-component contractions. The
joint call executes the same 5,308,640 geometry preparations, 10,617,280
radial preparations and 975,993,046 contractions. The preparation count is
larger than the ideal once-per-shell count because component capacity tiling
remains; it is not inferred from admitted components alone.

Maximum matrix differences from the same-binary retained calls are
3.056e-13 (full), 1.777e-15 (LR), and 3.091e-13 (joint), below the diagnostic
1e-8 gate. These checks complement the independent small-system CPU oracle;
the same-binary comparison alone is not an independent scientific oracle.
Charged total device bytes are 23,288,465 retained, 23,754,544 component,
and 24,803,120 component-plus-joint. Baseline component counters remain
unavailable (`null`), and FLOPs and endpoint performance remain unavailable.
The primitive-component/preparation ratio is a reuse observation, not a
speedup. Raw input identity, probe source, source/binary hashes, assigned
device and all five calls are retained in `sync-v3/census/runs/5762/`.

## Complete small-endpoint result and retained decision

Finite n1 job 5761, assigned device 2, completes the same-binary three-atom
full-def2-TZVPD ABBA comparison, bracketed by GPU4PySCF references with CUDA
LibXC verified on every call. All 72 complete E/F calls pass the unchanged
1e-8 Eh / 1e-7 Eh/Bohr limits; native maximum errors are 3.127e-13 Eh and
5.871e-11 Eh/Bohr. Every warm replay takes one SCF iteration.

Baseline cold times are 16.226265 / 15.444448 seconds; candidate cold times
are 17.070636 / 17.061314 seconds. Baseline process warm medians are
1.923335 / 1.925643 seconds; candidate medians are 2.024086 / 2.022240 seconds.
Ratios of the two process medians are 1.077713 cold and 1.051273 warm, with
1.101970 displaced and 1.050345 displaced-warm ratios. The candidate therefore
regresses on this small complete endpoint despite its observed source reuse.
The static stack reduction and preparation-count ratio must not be substituted
for these endpoint results or used to claim a larger-size crossover.

Keep the implementation default off and the PR experimental. Twelve-atom and
larger complete endpoints for this consumer remain unmeasured. A future decision
to revisit it needs profiling of the serial producer, component contraction and
barrier costs, followed by complete endpoint qualification. Derivatives are
unchanged; no combined gain with #1839/#1842 or
any other source/GPU measurement is inferred.

Exact raw endpoints, census input/probe, qualification logs, source/binary and
cache receipts, and executable reproduction scripts are retained in the
[reviewable evidence bundle](../../../benchmarks/results/wb97mv-shell-components-20261004/README.md).
The bundle's offline check repeats all endpoint acceptance gates without running
CUDA. The earlier local artifact paths retain the detailed failed SASS/logs too.
