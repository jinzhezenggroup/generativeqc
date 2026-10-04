# Experiment: share high-order value sources across Cartesian shell components

Status: proposed; default-off implementation, device/endpoint qualification pending
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
