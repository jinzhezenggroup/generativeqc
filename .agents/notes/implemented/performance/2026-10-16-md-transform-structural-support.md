# Decision: retain only structurally supported MD-J transforms

Status: implemented; standalone sparse-layout endpoint qualification complete
Date: 2026-10-16

## Problem and source boundary

The completed #2558 force comparison leaves cold r2SCAN-3c/96 dominated by
SCF Direct J (199.7 seconds over 24 builds) and XC/AO/grid (109.1 seconds).
Those are historical device measurements on the #2558 stack, **not current
master endpoint measurements**. The evidence is retained at
`benchmarks/results/direct-force-reuse-20261016/README.md` on commit
`fe0af786591b81f4b60cad32eab14b2d92aa7e00`.

The current comparison starts independently from the frozen master revision
`b84ea05eb230addcc8599aab913a5c13f997dbe7`, including #2553 and #2534. Earlier
resident trials used `d5fc58a1a39388f6f4ffe59b2ce607ccb7d04df1`. Neither contains
the unmerged #2541/#2549/#2556/#2558 force stack or #2520 XC changes. Do not
reuse that stack's endpoint timings as measurements of this candidate.

Existing density-first MD-J contracts density before public AO expansion and
reuses the generated reciprocal Coulomb recurrence. Its dense primitive
transform layout exceeds the existing 128 MiB optional cap at the target basis.
Raising that cap would not solve public resource admission and is not proposed.

## Decision

Retain the union of possible Hermite derivative powers over the actual sparse
Cartesian terms of each public AO. A derivative can contribute only if each
axis power is no greater than the sum of the corresponding two Cartesian
powers. This is exact polynomial support, independent of geometry, contraction
coefficient values, density and screening.

Intern support maps by the actual pair of AO angular-expansion signatures, not
by molecule, atom, shell identity, named basis or public representation. Each
primitive keeps its supported FP64 coefficients and one explicitly computed
`weight * 0.0` slot. Structurally unsupported entries refer to that slot, retaining
negative zero and overflow/nonfinite behavior. Geometry-dependent zeros and
spherical cancellations still execute the original coefficient summation.

All consumers, including the partially screened source J/K fallback, use the
same borrowed map. Generated Coulomb mathematics, contraction order, Schwarz
mask, existing density-error allowance and normal-K execution are unchanged.
Map uploads are charged to the same optional rollback group and cap. The public
active-ledger guard remains intact: free live bytes do not authorize optional
storage. Neither the resident cap nor any default budget increases.

Crucially, the default README calculator supplies neither `resource_plan` nor
`resource_budget`. `PreparedBatch` creates a persistent public ledger only when
one of those is supplied. The README already admits private MD at 48 atoms;
the old dense transform exceeds its cap at 96 atoms. Sparse storage alone
crosses that cap and activates the existing default source. Do not mistake a
public Python endpoint for an explicitly resource-budgeted ledger owner.

## Executable host evidence

The inventory test uses the canonical JSON's actual shell/primitive counts and
the production `molecule::ao_expansions`, rather than an estimated shell list.
Bytes include primitive/pair descriptors, index maps, transforms, both Hermite
scratch arrays and retained source metadata; they are not whole-process VRAM.

| Atoms / representation | Dense bytes | Candidate bytes | Original cap |
| --- | ---: | ---: | ---: |
| 48 / spherical, 544 AO | 45,878,756 | 31,303,512 | 134,217,728 |
| 96 / spherical, 1088 AO | 182,845,316 | 124,792,120 | 134,217,728 |
| 48 / Cartesian, 576 AO | 50,249,316 | 30,380,018 | 134,217,728 |
| 96 / Cartesian, 1152 AO | 200,228,996 | 121,109,074 | 134,217,728 |

The spherical 96-atom case retains 6,389,600 primitive coefficient slots,
including 431,952 weighted-zero slots, instead of 13,712,336 dense entries.
Only 1,690 uint16 map slots are shared across its 131,328 shell pairs. These
are retained/preparation inventories, **not measured executed Fock work,
hardware FLOPs or an endpoint speedup**. Additional indexed coefficient loads
could offset preparation/storage savings, especially in residual source work.

The new executable tests check all through-d Cartesian support, spherical term
unions, reordered public components, slot bounds, real-basis capacity and actual
optional-allocation rollback with every allocation failure injected dynamically.
The production CUDA preparation body also executes under host index emulation
against an independent long-double polynomial-to-Gaussian-derivative oracle for
36 angular/layout/geometry combinations. It checks signed structural zeros and
nonfinite weighted-zero propagation. This is host mathematical/storage evidence,
not GPU execution or a substitute for the existing resident Libcint tests.

343 focused host/API/resource tests pass, with 9 opt-in GPU skips. The translated preparation also passes host
ASan/UBSan with leak detection. Ignored diagnostic receipts live in
`.artifacts/direct-md-next/`; baseline/candidate source exports and cached
ALL-target builds live under `/data/jzzeng/gqc-direct-md-20261016/`.

Both changed MD CUDA translation units compile for sm120. Offline `cuobjdump`
object-resource reports print LOCAL:0 for both versions, but this must **not**
be described as a final zero-local-memory guarantee: relocatable objects still
contain explicit PTX local recurrence arrays. Final linked resources and device
profiles are needed to account for those workspaces and any spills. MD
preparation uses 64 registers in both; density transform uses 54 -> 48 and
projection 42 -> 40. The residual source module's maximum increases 102 -> 110
registers, with its shared-memory maximum unchanged at 1,036 bytes. The MD
module's maximum shared memory remains 10,264 bytes. These are compiler object
resources, not occupancy measurements or device-time evidence. Both native
ALL-target links and the source-matched real-device qualification complete.
Nsight launch metadata reports zero local bytes for preparation, but the
explicit 5,760-byte PTX local arrays remain in both sources; that zero is not
proof of zero local workspaces or achieved traffic.

`tools/check_scf_structure.py` checks 250 shared modules with zero dependency
errors. The support planner is a standard-library-only metadata leaf. The host
provider may borrow that leaf and the public basis metadata ABI, never the
device integral recurrence header to obtain an AO expansion stride. Public
admission, domain/cap and opt-out promotion audits remain intact. Executable
host tests retain public tight-budget/future-owner and every-allocation rollback
coverage without granting optional storage to those owners.

## Real-device evidence

The earlier local-controller availability check was insufficient: authorized
nodes have separate Slurm controllers reachable over SSH. Node5 is usable.
No GPU command ran on excluded node3 or outside finite Slurm allocations;
assigned device visibility is preserved.

All four earlier ALL-target builds complete. The b84 sparse resident candidate
passes 16 independent Libcint RKS/UKS, spherical/Cartesian, nonsymmetric-density,
K/fallback and reciprocal-census tests. The same 16 pass memcheck (0 errors),
synccheck (0 errors), and racecheck (0 hazards/errors/warnings) on node5.

A bounded b84 raw-J diagnostic with an explicit matched 1-GiB total test budget
(not the default budget) reports 96-atom dense-density event medians
8.9069043 -> 4.2186631 seconds and MINAO atomic-density medians
7.9448594 -> 4.2370767 seconds. These are NOT SCF or complete endpoints.
The first 256-MiB trial did not admit 96-atom MD and remains retained as such.
The original 128-MiB MD cap is unchanged in both trials.

The diagnostic's MD-vs-normal 3e-11 raw-J gate fails on atomic densities:
maximum differences 6.90e-11 (48) and 9.46e-11 (96). These failures remain
retained. Normal J is not an independent oracle: output-specific screened
Libcint contractions at the three largest-disagreement shell pairs per size
show MD errors about 1e-20 or smaller while normal J differs by those same
1e-10-scale amounts. Old dense MD and sparse MD agree over the full 48-atom
atomic matrix within 1.60e-13. This localizes the discrepancy away from sparse
storage; six selected entries are not whole-matrix or endpoint qualification.

The b84 baseline population completes all 12 calls: cold 409.462 seconds, warm
median 87.519 seconds, moved 257.727 seconds, moved-warm median 87.531 seconds;
SCF/Fock counts are 24/24, 1/1, 13/13, 1/1 respectively. All independent E/F
gates pass. The source-matched cold profile attributes 198.520 device seconds
to J and 108.234 to XC/AO/grid. Warm order-6 spatial, order-7 cooperative and
order-8 device times are 21.781, 28.035 and 5.772 seconds. These are separate
device ledgers, not additive components of the exclusive host wall ledger.

The focused candidate library is
`4949afdcaa8b41893db1032f7d7d689777e9031c10b150b1ab7c9ee6e642414d`.
Complete original-protocol candidate populations and a matched device profile
are retained at `benchmarks/results/md-j-sparse-support-20261016/README.md`.
The earlier unmerged force-stack endpoint timings cannot substitute for that
comparison. Candidate cold/warm/moved/moved-warm complete seconds are
311.849/83.562/205.257/83.534: -23.84%/-4.52%/-20.36%/-4.57%. All four original
12-call populations and all independent E/F gates complete within the unchanged
1800-second population bound. The matched 48-atom controls change by at most
0.28%; PBE0 cold changes +0.244%. Actual iteration/Fock counts are unchanged:
96 r2SCAN cold 24/24, moved 13/13; every warm call 1/1.

Matched device SCF J changes from 198.520 to 99.807 seconds over 24 cold builds,
and from 8.231 to 4.158 on warm. XC and force work are unchanged: candidate
warm order-6/7/8 device seconds are 21.773/28.034/5.773. All their recorded
executed component/primitive counters match, including 2,400,504,912 order-6
primitive-pair products and 1,250,544 order-7 Coulomb preparations. Unknown
order-6 preparation counters remain null. Do not describe this patch as a force
recurrence optimization or use these counters as hardware FLOPs.

The result qualifies the independently useful cold component, not the 10% warm
aspiration or historical GPU4PySCF parity. The next step is a matched stacked
qualification with #2558, not arithmetic subtraction of separate campaigns.
Explicit public-ledger admission remains deferred; the default unbudgeted
README path is the scope that crossed the existing private cap.

Two first focused populations were deliberately stopped after discovering that
scancel did not terminate predecessor timeout/profiler descendants on these
controllers. The n5 predecessor was verified still using the GPU; the n4
predecessor profile completed at 15:51:47 after the new job began at 15:44:47.
The inflated 542.922-second cold control and all partial records remain retained
as invalid overlapping timing, not suppressed negative code evidence. Guarded
replacement allocations verify no pre-existing compute context. Neither source
nor scientific protocol changed. Synchronization observations from the intrusive
MD census are nested diagnostics: its pageable copy/fence shifts API wait
attribution and is not an independent production synchronization speedup.

## Deferred public-ledger experiment

An intermediate prototype added immutable optional numeric-storage permissions,
per-owner quota leases, selected CUDA KS candidates and original-AO-only MD
screening. It passed 31 independent resident tests and all three sanitizers,
including racecheck with no hazards/errors/warnings. Its patch and complete source
snapshot remain in ignored artifacts (`public-admission-deferred.patch` and
`/data/jzzeng/gqc-direct-md-20261016/candidate-public/source`). This architecture
is NOT part of the focused implementation. A mixed eligible/ineligible fleet
would additionally need per-owner eligibility in the native shape query before
promotion; a global permission alone is insufficient to enlarge an ineligible
owner's local budget.

The default README population of that prototype executes the private MD source,
not its newly authorized public-ledger path. Its cold 311.444-second return and
83.445-second first warm return establish potential, not qualification of the
new permission mechanism. The 96-atom population and subsequent intrusive
profile were deliberately stopped when this scope error was discovered. Their
partial raw records remain retained and must be reported as incomplete, not
silently promoted as full endpoint evidence. Separate sparse-only populations
use the already built, independently checked source above.

A test-only loose-screening failure in that prototype compared the incumbent's
density-screened fallback against MD's AO-only oracle. At screening 0.02 these
are different approximations: the existing `direct_screening.cuh` also rejects
shell-density contributions. The corrected lifecycle test retained the loose
case, checked fallback compatibility with the incumbent and independently
checked selected MD against Libcint. Zero and original 1e-12 screening retain
the independent fallback gate. No production tolerance was changed. This is
also consistent with, but does not fully diagnose, the earlier atomic-density
normal-vs-MD raw-J discrepancy.

The complete sparse-only populations, independent oracle inputs, source/build
identities, mutually exclusive host accounting, separate device profiles,
sanitizer logs, executed-work counters and negative/incomplete trial index are
retained in the evidence directory above. The optional permission prototype
does not appear in the implementation patch.

## Rejected alternatives

- Increasing the resident cap or spending the ledger's currently unused bytes:
  no storage/work reduction and no authorization for later-owner capacity.
- Factorized axis storage reconstructed inside every residual contraction:
  likely adds arithmetic exactly where recurrence reuse is needed; first
  establish the sparse stored-coefficient candidate's real device cost.
- Scheduling-only task/page/CTA experiments: already negative in #2486 and do
  not address repeated recurrence work or dense coefficient storage.
