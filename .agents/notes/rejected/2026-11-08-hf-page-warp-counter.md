# Decision: reject warp-coalesced page counter claims as an HF cold win

Status: rejected; experimental production code removed
Date: 2026-11-08

## Cold-only attribution

Earlier DPPP/PPPP kernel totals mixed cold and warm work. LSF 229318 instead
collects exactly one complete HF Direct96 cold constructor/preparation, SCF,
analytic forces and host export, stopping CUPTI before owner cleanup and never
executing a warm call. The native process rejects PySCF/GPU4PySCF imports.

This is the authenticated historical source
`790fd246fed0314d6f32c39c50d1243a3f8490d949ffc2f37b70e72745bffd89`, ELF
`47c4c19033fc37580e43706f1428c7fff3aa685b7c41f067238dba5b1bfe033c`, with
the earlier-handoff control unset. It is not qualification of merged head
`c30b371dbbeb7c132500cd1741b52a7d00fb57b5`. Public tolerances remain energy
1e-9, density1e-5, screening1e-12. There are eight screened plus seven strict
iterations/Focks; the seven initial-guess iterations/Focks are separate.

Profiled complete latency is14.541139s, collected kernel duration13.681372s.
Generated Fock kernels plus native full-range DDDD account for10.161169s;
Fock-purpose exact-page compaction accounts for1.567850s across1425 launches.
Shared-XY DPPP and PPPP contribute1.117565s and0.862078s; PPPS0.953991s,
DPPS0.781701s, PSPS0.753132s, DPDS0.727015s and DPDP0.645061s remain material.
These are kernel aggregates, not isolated exchange time, hardware-atomic
traffic, or seconds that can simply be subtracted from endpoint latency.
The Fock-family sum covers the complete cold collection, including initial
guess kernels where applicable; it must not be labelled strict-tail-only K.
All dropped/starved/errors/unknown/flush/disable collector counters are zero.
Independent reference errors are E1.4002e-9Eh/Fmax1.5476e-9Eh/Bohr.

## Source-owned mechanism screen

The existing page owner individually increments its survivor or signature
counter. The candidate groups currently active lanes by counter address,
reserves their total with one atomic and assigns disjoint lane-prefix slots.
The hardware-intrinsic legality guard permits match-any only where supported,
with scalar fallback; there is no SM80/product-name profitability dispatch.
Page domains, predicates, capacities, signature populations and integral
arithmetic do not change. Different task order is already permitted.

Before building, the clean page owner is byte-compared to the frozen image's
owner. Cached Slurm16834 builds original and candidate interposers from those
owners, with distinct launch interception and positive actual-host-call checks.
This avoids a full native rebuild, but does not turn a mixed core/interposer
experiment into current-head qualification. Both arms use an interposer.

Each card runs one finite scheduled ABBA population, with four fresh processes,
each measuring complete cold E/analytic forces/host export plus three warm
calls from its own frozen cold density. Neither a fixed-density kernel-only
measurement nor a faster selected repeat decides promotion.

|Card/job|Control cold/s|Candidate cold/s|Candidate/control mean|
|---|---|---|---:|
|A100 LSF229323|14.583722 /14.697732|14.622118 /14.653436|0.999798502|
|5090 Slurm16835|16.774069 /16.809494|16.790227 /16.826012|1.000972977|

All32 independent E<1e-8/Fmax<1e-7 checks and all12 paired warm<=3% guards
pass. Maximum E/F errors are1.4029e-9/1.5477e-9 on A100 and
1.3993e-9/1.5476e-9 on5090. All eight cold endpoints retain8+7 target
iterations/Focks and separate7/7 initial-guess work. No work-count reduction
is inferred from a changed queue order.
Native quartet/recurrence totals were not acquired; missing counts are not
zero. There is no new HF48, displaced-geometry or final-default qualification.

Both cards pass324 independent queue populations: empty/full/sparse/complement
masks, divergent rounds,53 independent counters, partial warps and three graph
replays, including block sizes1/11/31/32/33/64/65/128/256. Slurm16835 also
passes memcheck, racecheck and synccheck with zero errors/hazards. A100 is not
claimed to have run those sanitizer tools.

## Decision and evidence boundary

The cold differences are noise, far below the retained2%/noise acceptance
standard. Remove the helper and production changes, retain no runtime opt-in,
and do not sweep warp/block sizes or repeat the same candidate. This does not
prove that task compaction or global atomics never matter. The remaining major
measured cost spans many Fock families, rather than only DPPP/PPPP recurrence.

Complete outputs, arrays, source-owned candidate patch, interposer source,
source/binary hashes, cached build stdout and reducers remain in ignored local
`.artifacts/hf-strict-k-next-20261108/`. They have not been added to the almost
full benchmark-evidence budget or published as Release assets. This note and
issue2976 retain the decision; no new production speedup or refreshed README
ranking is claimed. PR3048 remains draft. A future accepted extension still
requires complete source-bound qualification, both-card protection and durable
raw-evidence admission, rather than citing this diagnostic as final evidence.

Archive SHA256s:

- Cold profile229318:
  `fd2dd7876e19119858ec03d57e3dd701325f0bfc56e090c24e4ca0b18c25f6e8`.
- A100229323:
  `4a043143e030e447d94814dfef4c6b9b9f3a1cf22dafc64f6b2719156cb0f0ea`.
- 509016835:
  `b6d8330c6be7ec8345dd26de8f30a5381e54374a2b4242f380d10ec81455d2fe`.
- Control interposer:
  `4364cd69ca0923dfbee443b4b47b487560a5a2e5c9b50a74bc4575c06fcc3519`.
- Candidate interposer:
  `c296c63191908b8d546518ad643b276c05a8402ec5fad7b1147876300012290d`.
- Independent queue fixture:
  `52eee5d70a9d0176020e0d50440ce89b81558bec45a6735abfd8cc104a6fc726`.

Initial infrastructure failures remain visible: the old login curl does not
support header-file syntax, so initial health failed before GPU submission;
the first local allocation named an SSH alias rather than the Slurm node;
Slurm16833 found no ccache and did not compile. The corrected build verifies
and uses the available sccache, without clearing its cache. Transfers that
time out after delivering bytes are accepted only after exact remote-hash
verification. They are not native execution failures or repeated benchmarks.
