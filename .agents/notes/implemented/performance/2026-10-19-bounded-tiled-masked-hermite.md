# Decision: keep bounded tiled masked-Hermite J experimental

Status: implemented, default off; no performance promotion
Date: 2026-10-19

## Problem

The original exact masked consumer repeats density transforms per shell task.
The subsequent resident-cache experiment covers only about 1.30% of the
96-atom `(2,1)` residual tasks within the original MD memory allowance and has
no reproducible complete-J benefit. A bounded shared-memory reuse scope is a
different alternative; simply increasing the resident allowance would not
qualify it under the existing resource contract.

See the preceding decisions:

- `2026-10-18-masked-hermite-residual-prototype.md`.
- `2026-10-19-bounded-residual-density-preprojection.md`.
- `../../proposed/2026-10-18-residual-j-tiled-density-first.md` for the earlier
  upstream comparison and hardware counters, which do not describe this kernel.

## Decision and ownership

`GENERATIVEQC_MD_J_MASKED_TILES=1` selects a default-off J-only `(2,1)`
compiler-generated consumer. It takes precedence over the older masked and
resident-cache consumers for that class. The native owner validates and freezes
the switch, retains all allocation/launch/error policy, supplies exact native
admission and provides the original AO-component fallback. The compiler owns
strip organization, mask planning and contraction scheduling. Mathematical
suffix, transform and radial definitions are shared with the existing owners,
not replaced by a separate integral implementation.

Each persistent CTA has four warps and a two-bra by four-ket shell-pair tile.
Two-primitive chunks bound scratch independently of molecule size and primitive
contraction length. Sorted directed density entries belong to six input strips;
eight tasks retain only cutoff/group maps. A primitive's density suffixes can
serve multiple tasks in the tile. At most four radial groups are admitted by
the existing profitability gate. Invalid bounds or rejected masks use the AO
fallback before publishing any masked output.

Within each primitive chunk, Hermite potentials accumulate over the opposing
primitives before AO back-projection. AO values survive all chunks and publish
once per task rather than once per primitive product. This is algebraic reuse,
not permission to change the AO mask or drop small contributions. It reorders
FP64 sums and therefore requires independent numerical acceptance, not bitwise
equality. A small Hermite R lookup can be reused across groups and directions;
it is populated using the native recurrence/index convention and is not an AO
four-index integral block.

There is no new resident allocation or larger MD allowance. Optional resident
preprojection remains a separate experiment. K, uniform reciprocal J, UKS
directionality, orbit deduplication and directed-rounding density admission
remain unchanged. The tiled path honors the native residual coarse-prefilter
policy, including its exact all-zero-density rejection and census behavior.

## Source-matched 96-atom evidence

All timings use n1 / RTX 5090 under finite Slurm allocations, preserving its
assigned device visibility. Each comparison uses one isolated binary based on
PR #2600 commit `aacbccc275274cb70e36b76a646750fe667acf3e`, retaining compact E
access, the original residual prefilter and angular-three reciprocal dispatch.
This pins the comparison source, not the PR's current merge status.

Use the same 96-atom / 1088-AO spherical r2SCAN-3c basis, frozen density,
`1e-12` screening, 1 GiB request budget and original 128 MiB MD allowance.
Density-array SHA256 is
`a0b22059e12adcd39771e5dcfaac0951afb298d8252ef6674889ad0349ed6090`.
The independent retained GPU4PySCF J is
`/data/jzzeng/gqc-mdj-residual-20261018/results/j-only-10083/gpu4pyscf.npy`.

Warm both owners, then run `0,1,1,0,0,1,1,0`: four complete resident-J samples
per policy, with density projection included. Work-count replays and profiler
replays are separate from clean timing. These are not SCF energy/force endpoint
measurements.

| Variant / Slurm job | Incumbent J median (s) | Tiled J median (s) | Time change |
| --- | ---: | ---: | ---: |
| Initial duplicated plans / 10273 | 3.9422342529 | 4.7231804199 | +19.81% |
| Compact masks and worker / 10275 | 3.9372786865 | 4.3888107910 | +11.47% |
| Primitive-chunk potential fusion / 10277 | 3.9524267578 | 4.4100822754 | +11.58% |
| Reused Hermite R lookup / 10282 | 3.9508771973 | 4.3277182617 | +9.54% |

The retained final lookup variant improves on the preceding experiment but
still loses clearly to the incumbent in its source-matched comparison. Do not
describe improvement between failed prototypes as a production speedup. Its
maximum incumbent difference is `2.985e-13`, with the same independent J gate
and unchanged logical output counts and owner bytes. Hardware counters below belong
only to the fused predecessor, not this last lookup variant.

The first variant also failed the native zero-density tile-census acceptance
test; the compact and fused versions restore that invariant rather than
removing or suppressing the test. The fused logical AO/Hermite output terms
decrease from `5513255552` to `3557663232`. This is a semantic dense-term count,
not a hardware instruction count or a speedup. Grouped radial terms remain
`6093089280` over `9081696` admitted masked tasks, about 91.46% of the
`9929600` `(2,1)` residual tasks. Residual primitive products remain `55373760`.
All angular-class uniform roots/directions/summands and admitted residual
tasks/products match the incumbent. Compaction order can change strip reuse
and refresh counts; it does not change the accepted integral domain.

The fused diagnostic replay refreshes angular-one/two density strips with
`801652368` / `1985788080` logical density/E summands. Both policies retain
`662672188` uncounted owner bytes; suffix/metadata bytes are zero. Maximum
independent J error is about `2.44485e-11`, and the fused/incumbent difference
is at most `3.02e-13`. Gates are `atol=3e-9, rtol=3e-11` against the independent
J and `3e-11` against the incumbent.

Static sm120 resource inspection reports 100 registers/thread, 18024 shared
bytes/block, and zero stack/local bytes for the uncounted fused kernel. The
compact predecessor also uses 100 registers/thread and 18808 shared bytes.
These are compiler resource reports, not measured occupancy or bottlenecks.
The final lookup variant uses 99 registers/thread and 19304 shared bytes/block,
with zero stack/local bytes in the static uncounted sm120 report.

## Fused-kernel hardware counters

Slurm job 10280 profiles the fused variant, before the R-matrix lookup
experiment, using the same `f18b486...` binary and frozen density. Nsight Compute
2025.2.1 collects one uncounted `(2,1)` launch per policy with 22 kernel-replay
passes. Use the authorized `user` account and sudo for restricted counters,
inside `srun`. sudo/PAM clears device visibility on this machine: pass Slurm's
actual `CUDA_VISIBLE_DEVICES` through `/usr/bin/env` after sudo and compare it
with the receipt before allowing execution. No profiling restriction or GPU
visibility is disabled. Passwords are not part of scripts or receipts.

The device is RTX 5090, driver 580.95.05, assigned visibility 1. SM frequency
is approximately 2.88--2.89 GHz. Clocks are not locked; these kernel-replay
observations diagnose execution, not a clean timing distribution. Both complete
J results still pass the independent oracle, with error about `2.44485e-11`.

| Metric | Incumbent `(2,1)` | Fused tile `(2,1)` |
| --- | ---: | ---: |
| FP64 pipe active | 86.56% | 59.61% |
| Reported memory throughput peak fraction | 5.14% | 10.07% |
| Achieved occupancy | 33.28% | 33.33% |
| Executed instructions | 269.242 billion | 358.571 billion |
| Mean active threads / warp instruction | 4.39 | 3.82 |
| DFMA + DMUL + DADD opcode instances | 12218058227 | 14179079243 |
| Global FP64 atomic opcode instances | 79282176 | 24946624 |

The arithmetic opcode sum increases 16.05%, total instructions increase 33.18%,
and FP64 atomic opcode instances decrease 68.53%. These are NCU opcode counts,
not FLOPs or semantic terms. The memory metric is aggregate device-memory
throughput, not DRAM bandwidth; measured DRAM traffic is only about 14--24 MB/s,
well below 0.01% of its reported peak. This does not support a global-memory
bandwidth saturation diagnosis. Achieved occupancy does not drop in this
comparison, unlike the older `(2,2)` masked experiment. Lower FP64 utilization
does not by itself identify synchronization, instruction issue or indexing as
the unique remaining cause. The SASS/stall exports are retained for attribution.

Reducing selected output terms has not reduced complete executed work. The
subsequent lookup experiment hoists invariant R indexing across cutoff groups,
rather than inferring a speedup from its arithmetic estimate or increasing the
tile size without evidence.

## Validation and synchronization

All 88 focused host tests pass, including real launch-error propagation and optional
allocation/fallback behavior. The standalone CUDA gate has 64 cases, each
testing counted/uncounted policies, directed/nonsymmetric/UKS density and
independent closed Hermite/Boys AO-orbit results. It includes three-primitive
chunks, a three-bra by five-ket ragged tile grid, fallback, system offsets and
original/zero/original density replay. Numerical, memcheck and racecheck runs
all pass; the latter reports zero hazards. The private probe additionally
checks fused output-term counts against independently enumerated AO masks.
Standalone probes compile sm80 PTX for execution on the assigned GPU; the
source-matched complete-J production comparison is compiled for sm120.
The molecular suite passes 114 cases with tiles off and 114 with tiles on.
Compiler structure checks find 567 modules and zero dependency errors.

Racecheck caught a real persistent-iteration race in the first prototype:
thread zero reset shared admission flags while another warp could still read
them at `continue`. Copy each shared decision to thread-local storage and
barrier before branching. Preserve this synchronization when changing the
persistent schedule. Memcheck and numerical agreement alone did not expose it.

## Consequences and revisit conditions

Do not enable this route by default, open a performance PR, or claim an SCF
speedup from these negative complete-J results. Preserve the original fallback.
The experiment demonstrates exact bounded reuse and broad mask coverage, not
profitable GPU execution. Reducing selected dense terms is insufficient when
suffix refresh, sparse transforms, group indexing and scheduling add work.
Hardware diagnoses must use counters from this kernel rather than transferring
the old masked consumer's measurements.

A later candidate must establish a reproducible complete-J win on the same
input and then rebuild source-identity-correct stationary-force AOT artifacts
for independent cold SCF energy/forces and convergence acceptance. Never edit
manifests to reuse stale force identities. Do not proceed to `(2,2)` merely to
increase experimental coverage before `(2,1)` demonstrates benefit.

## Receipts

Ignored local evidence is under `build/tiled-hermite/`; n1 retains source
archives, build commands/cache statistics, binaries, diagnostics and profiler
reports under
`/home/jzzeng/generativeqc-qualification/tiled-hermite-20261019-49fc/`.
`counter-summary-fused.json`, `counters-fused/` CSV/log/SASS exports and
`counter-reports-10280.sha256` identify the hardware evidence. Large NCU reports
remain on n1; no external publication or Release is used for retention.

| Variant | Library SHA256 | Source archive SHA256 |
| --- | --- | --- |
| Compact / 10275 | `2e6459ebcb690ca2ed6ba726104cb6061d8490bbe4e93d60e928b1363df99809` | `3067913bf49107c2c8a735a34c28bb5c82f2aa12cee37be428ea1f2d60258c55` |
| Fused / 10277 | `f18b486170338cc138f79c2dfd2b300deab7065e20bb54f95de8c7887d8290cb` | `930878e3536c3e0136fc39dbda4886b6feb0af267257ccb1f618840e988b5183` |
| R lookup / 10282 | `80f3f92c46391ee28aa532665ca6092fed73d2dffa453092f346302c3412dcb6` | `129f33bd564a546b1e3aa20fffba9c3ac774165f473431f2924921f143392f20` |

No GitHub posting, release, default promotion or external artifact publication
is authorized by a negative benchmark.
