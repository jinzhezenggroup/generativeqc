# Decision: opt-in rotating-partner VV10 pair traversal

Status: implemented experiment; default unchanged
Date: 2026-10-15

The subsequent user-authorized guarded default promotion is recorded in
`2026-10-15-vv10-packet-default.md`. The opt-in decision, measurements and
limitations below remain historical evidence; promotion does not convert the
96-atom timeout into successful endpoint qualification.

## Problem

The parallel bilateral schedule evaluates off-diagonal scalar pairs once, but
each warp loads the same partner in all lanes, reduces every partner's three
or six partials through five shuffle stages, then publishes from lane zero.
Increasing rows per lane amortizes these reductions but trades occupancy and
tail utilization for retained inputs. The existing wide-tile experiment is
not a reason to promote another schedule without complete-endpoint evidence.

## Decision

Add the explicit `GENERATIVEQC_VV10_PAIR_SCHEDULE=parallel-packet` selector.
Retain four rows per lane, 128-row unordered tile pairs and the canonical
generated bilateral pair roots. Load a 32-partner packet with one partner per
lane, then rotate both immutable partner inputs and that partner's accumulated
partials through all lanes. After 32 rotations each lane publishes its original
partner. Diagonal tiles still evaluate ordered pairs and publish only row sums.

This exchanges register traffic for the old per-partner warp reductions and
lane-zero publication. It does not reduce scalar scientific pair work:
for N nonzero partners and tile lengths m_t, both schedules evaluate
`(N*N + sum(m_t*m_t))/2` scalar pairs. Observable positive-zero rows retain the
separate `Z*N` response sweep. No new global workspace, transfer, allocation,
host synchronization, precision policy or approximation is introduced.

## Invariants

- All 32 lanes rotate, including padded and inactive partner slots. Predicating
  shuffles on a live partner would lose contributions or use invalid masks.
- A complete rotation returns both the partner and its partials to the original
  owner; publication must not use an intermediate packet's identity.
- Negative-zero screened rows remain excluded. Positive-zero weighted rows
  still own feature and quadrature-weight response.
- Native numerical admission, disjoint output leases, alias-safe expired-gradient
  storage, asynchronous failures and exact-key invalidation remain authoritative.
- The ordered and existing parallel/wide routes remain available; the new route
  is explicit-only pending larger complete-endpoint qualification.

## Rejected alternatives

An isolated 4/8/16-row, retained/unretained-input sweep confirmed that merely
widening tiles is not uniformly beneficial. At 10,000 synthetic points, 16-row
retention was about three times slower for features and geometry; eight-row
packets likewise lost to four-row packets. At 80,000 points, eight-row packets
had no meaningful advantage over four-row packets. Keep the smaller register
and tail footprint rather than adding another adaptive-count launch.

## Evidence

Base: `origin/master` at `5f80ba629` on 2026-10-15. Device work uses finite n4
Slurm allocations on RTX 5090, CUDA 12.9.1, sm_120, FP64 canonical pair roots,
and a verified reusable ccache launcher. Local artifacts are under
`.artifacts/vv10/`; these are experiments, not a release or external publication.

The production-generated admission probe passes 752 device tests, including
partial warps/tiles, signed/zero/screened weights, rejected-domain fallback,
asynchronous numerical failure, output canaries and an independent long-double
evaluation of the original pair energy and derivatives. Each of memcheck and
synccheck passes 58 packet cases with zero reported errors. The unchanged
pair-root/reference host suite passes 100 tests before integration.

The integrated host checks pass 176 tests (two optional device checks skipped
on the unscheduled host), and compiler structure reports no dependency errors.
The full source-matched Release build succeeds. Its sm_120 resource report uses
128 registers for packet features and 166 for packet geometry, with zero stack,
local or shared storage. The incumbent four-row kernels use 84/128 registers;
this is a communication/layout change, not an occupancy improvement.

The initial isolated 80,000-point schedule interval decreases from about
141.3 to 120.5 ms for features and 200.9 to 163.6 ms for geometry. These intervals
include output initialization/finalization but exclude local-scale production,
allocation, transfers, SCF and molecular force assembly. They are not complete
endpoint speedups. Full molecular measurements and source-matched qualification
are recorded separately before any stronger claim.

### Complete consumer and molecular qualification

Source inventory `acaa3866a5dbfe3822d4cb2b01c07e3aca2cacd253c6a8efb8ddcbed3d2884ea`
and library SHA-256
`b8d9febe0e9da4ea9e3e7bb3c6597b1ecdf1b444ba72f6df0c8499d1bc8a61a5`
pass the full source-byte preflight. n4 Slurm 947 passes 39 native/lifetime tests
and six complete molecular independent E/F/finite-difference tests. The original
3/12-atom protocol then completes all 48 off/on calls with maximum independent
energy error 3.36e-12 Eh and force error 5.54e-11 Eh/Bohr, against the unchanged
1e-8/1e-7 gates. Neither 48 nor 96 atoms was attempted in this campaign.

Three interleaved calls per arm disable exact-component reuse and measure the
whole VV10 force-seed consumer, including preparation, transfers, local scales,
domain admission/compaction, pairs, seed download and retirement. Median wall
time at 320,000 points decreases from 2.52848 to 2.20718 s (12.7%); at 80,000
points it decreases from 0.16349 to 0.14288 s (12.6%). The 320k domain has 234,683
nonzero partners and 23,467 observable positive-zero rows: 33,060,378,882 scalar
pair evaluations per call in either arm. Every call reports one pair sweep and
zero cache hits. These are complete VV10 consumer timings, not whole SCF times.

For the 12-atom molecular population, changed-geometry time is 19.73289 versus
18.94765 s (4.0% lower); cold is 30.75992 versus 29.67734 s, while warm and
moved-warm are essentially unchanged. Both arms perform 36 SCF iterations,
37 Fock builds and four SCF pair sweeps cold, and 17/18/three after geometry
change. Small three-atom phases are essentially unchanged. A single off/on
cold comparison is not causal evidence because artifact-cache state can differ.
This modest molecular benefit is distinct from the larger isolated pair gain.

The compact archive, raw populations, source patch, complete source inventory,
compiler/cache receipts, executable experiment drivers and opt-in verifier are
retained under `benchmarks/results/vv10-warp-packet-20261015/`. The verifier
recomputes numerical gates, full phase populations, work counts and medians;
it is not a required-CI fingerprint of current production sources.

## Consequences and revisit conditions

### Original-limit 96-atom follow-up

A separate n4 Slurm 951 campaign uses the identical source-matched library,
an exclusive finite 65-minute allocation, eight CPU threads, no density fitting,
the original 48/16/32 grid (2,359,296 nominal points), five warm replays per
geometry and a separate unchanged 1800-second whole-point limit for each arm.
Both independently reconverged geometry E/F receipts match the exact protocol
and basis bytes before execution. They are accuracy-only, not reference timings.

Observed complete cold E/F is 1325.684 s for `parallel` and 1277.316 s for
`parallel-packet` (48.368 s, 3.65% lower). Both cold calls use 36 SCF iterations,
36 Fock builds and four SCF pair sweeps. Five-call warm medians are 59.591 and
59.689 s; each replay uses one Fock build and zero SCF pair sweeps. Every returned
call passes independent E/F gates, with maximum errors 8.641e-12 Eh and
2.335e-11 Eh/Bohr. There are no compiler-cache misses during either arm, but
this single fixed-order cold comparison is still descriptive rather than a
repeated/interleaved causal estimate.

Crucially, **both arms time out in `moved/0`, with only 6/12 calls returned**.
Moved work is interrupted, not measured as zero; there are no moved-warm
observations. The packet schedule therefore does not resolve the 96-atom
endpoint or justify default promotion. SCF counters alone do not identify the
approximately 60-second warm endpoint bottleneck; collect complete-path
profiling before assigning it to an individual unmeasured kernel. Preserve this
failed endpoint rather than replacing it with a smaller grid, fewer replays,
or a longer whole-point deadline.

Raw journals, timeout outcomes, full independent force vectors, provenance,
drivers, work counts and the opt-in historical verifier are retained separately
in `benchmarks/results/vv10-warp-packet-96-20261015/`. The preceding 3/12 campaign
is unchanged.

### Promotion boundary

Summation order changes, as it already does for the incumbent bilateral atomic
schedule. There is no new bitwise-reproducibility guarantee. Keep default
selection unchanged until complete original-protocol populations, especially
48/96 atoms and changed geometries, establish a consistent endpoint benefit
under the independent energy/force gates and original deadlines.
