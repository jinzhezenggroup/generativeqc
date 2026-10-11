# Decision: preserve the three-pass SPD force schedule and narrow only its low consumer

Status: rejected for promotion; numerical checks pass but endpoint benefit is unproven
Date: 2026-10-11

## Problem

The [preceding class-domain investigation](2026-10-11-pbe0-spd-class-force-domains.md)
removed unreachable high-angular private frames but introduced five whole
bounded-inventory traversals instead of the incumbent's three. It also moved
order six from a warp consumer into a serial-per-quartet cooperative consumer.
Its generic-decoder refactor loaded pair geometry before whole-block rejection.
These are different algorithmic changes, not an isolated frame specialization.

The frozen indexed-v1 pilots completed all independent numerical, density-seed,
resource and available-work checks, but their raw medians were slower:

| Atoms | Geometry | Incumbent (s) | Indexed (s) |
| --- | --- | ---: | ---: |
| 48 | warm | 7.488197099 | 7.684097031 |
| 48 | moved | 6.178028818 | 6.422670867 |
| 96 | warm | 15.659994189 | 20.617967615 |
| 96 | moved | 16.691532334 | 17.859703362 |

Those two-pair pilots are sufficient to reject promotion, not to make formal
regression claims. All slow samples and the incomplete racecheck remain
retained separately. Do not repeat the broad timed-out racecheck for this
already-rejected source or pool its samples with the new prototype.

## Decision

An explicit owner-prepared `narrow-low` selector replaces only the incumbent
orders-zero-through-six bounded consumer. It retains the original scalar
precontracted order-zero-through-three workers, scalar weighted order-four/five
workers, and explicit Cartesian order-six warp algebra. The specialization at
order six cannot instantiate unreachable order-seven-through-twelve recurrence
frames. Every logical 256-component tile and all eight 32-lane subtiles remain
owned, including the second tile of a 324-component dd/pp or dp/dp product.

The existing separate order-seven and materialized order-eight consumers, cursor
resets, workspaces and error propagation remain. The new launcher uses 128
threads and borrows the existing indexed block prefix; it allocates no quartet
list or new resident storage. Physical pair orientation, exact force screening,
enabled/generated-overflow masks and profile updates are unchanged.

Block-page decoding now has two explicit stages. The first resolves only system,
blocks and page position. The actual consumer rejects inactive and coarsely
screened blocks before the second stage reads pair ranges. The old one-step
adapter remains available for the rejected experimental indexed consumers.

The original generic Fock/RSH/force kernel body is restored byte-for-byte against
base `5a9b107ca8b9a654d45d3dc15fac45e962323f3b`, so same-image timing does not
compare against a newly slowed incumbent. Source equivalence is recorded in an
ignored provenance receipt rather than a test permanently tied to Git history.

The unset selector still chooses the incumbent. Missing prefix, derivative
cache, coherent topology or the original supported launch shape also keeps the
bounded fallback. Mixed f and range-separated routes are not promoted by this
prototype. The existing independent controls remain available alongside the
new `--spd-force-low-only` oracle entry point.

## Evidence and acceptance

Focused host probes compile the actual drain and actual Cartesian tile-view and
ordinal expressions, comparing them with an independent AO-pair inventory for
both spins and source layouts, empty queues, queues larger than a CTA, diagonal
171-component packets and off-diagonal 324-component tails. They verify that
local tile views do not mutate shared descriptors. A separate executable probe
uses null pair-range pointers during block decoding and tests both indexed and
triangular inventories. Split dispatch probes exercise all four selectors,
both source layouts, missing-prefix/cache/topology and submission failures.

The native oracle must require the requested prepared selector and optional
prefix, then mutate the environment after preparation without changing the
owner. Preserve all 64 independently contracted arbitrary-density J'/K' calls:
RKS/UKS, opposing/zero densities, source masks, unequal primitives, original/moved
geometries and Cartesian/spherical projection. Keep the 3e-10 derivative and
exact-zero disabled-channel gates. Independently complete FCI force cases remain
separate from host scheduling tests.

Timing uses a single native image/process/CUDA context and two independently
prepared resident owners, baseline `0` and candidate `narrow-low`. Both use the
same resident final-validation policy so this pilot cannot attribute #2220's
existing gain to force scheduling. Preserve authenticated owner-local seeds,
the one-Fock replay and 2e-11 density-drift gate. Both 48/96 atoms and both
original/moved geometries are required. Every complete E/F call independently
passes 1e-8 Eh / 1e-7 Eh/Bohr, resource and available semantic-work gates.

Only a positive pilot warrants five interleaved pairs and completed safety
qualification. The promotion gate stays
`max(2%, 2*(relative_MAD_baseline + relative_MAD_candidate))`; failures and slow
samples are never discarded. A verified improvement should be enabled by
default in its proven scope and submitted as a stage PR, not left opt-in because
unrelated consumers lack evidence. No gain means rejection, not a performance PR.

## Provenance and scheduling

New receipts live under `.artifacts/ks-weight-stage/spd-low-v1/`, distinct from
the frozen indexed-v1 artifacts. Verify compiler caching before compilation,
preserve source/artifact manifests and cache statistics, and run every GPU
qualification or timing command through finite `srun` on an allowed node.
Latest inspected master `02ff60da4d404549b1313f818dca9464168be85f` adds UHF
stability tooling, not these force/provider files; it does not trigger broad
retesting or make this frozen-base timing a latest-master claim.

## Initial narrow-low source freeze and host evidence

Production patch SHA256 is
`14c99f75793cefefe212e17c6cce9cea2b8031f950c99936572d6d0b80b03b75`;
qualification patch SHA256 is
`ae28f3835f95ccae1370243c41c7b88b156f196e7ea405f53a56830b8562f0cf`.
The restored generic section is 454 lines/28,451 bytes, SHA256
`3d05fe4c5b052175751fcb1ed8d744177f0715da617eb10821ba5a0184981f00`.
The first receipt mistakenly stopped at the default-argument `{}`; it is
retained as invalid signature-only evidence and superseded by the full-section
equality check. Neither the test extractor nor Git history is a numerical oracle.

The scoped campaign passes 99 host tests. After formatting, all six direct
page/capture/split/drain probes pass again. Ruff checks and configured formatting
pass; the structure checker reports 237 modules and zero dependency errors.
Local ccache 4.5.1 is verified and before/after statistics remain retained.
Initial missing-package collection attempts, an incorrect four-field test
fixture and the post-format exact-string fixture failure remain separate;
none is a GPU result. They are repaired without relaxing ownership assertions.

Remote CPU build job 7304 uses the source-reconstructed baseline archive, an
unchanged in-tree build layout and verified ccache 4.5.1. Its original finite
45-minute job and source manifests remain authoritative until terminal.
Qualification-observer CPU job 7305 completes successfully. This observer only
forwards the already-loaded runtime's original CUDA launch and records actual
narrow-low/generic-low/seven/eight and rejected-domain launches. Its GPU
admission campaign must verify both 48/96-atom original/moved complete calls,
including the 128-thread narrow-low width and equal seven/eight launch counts.
It is never preloaded in clean timing; admission receipts cannot qualify speed.
At that source freeze, no native numerical, safety or performance result was
claimed; later receipts below supersede that pending status, not its source identity.

## Completed numerical and actual-admission evidence

CPU build 7304 completes in 18m02s, exit zero, with actual verified launcher
commands and before/after cache statistics. Native library SHA256 is
`9bc3fbb8381c6ebe31b18bc8b811bb533cb834cc03c93bea23e7bf12398a6a08`;
standalone oracle SHA256 is
`74495d010ad83df065276afe508a2bd7ce665a03964996240862f942530c7d7d`.
The stationary AOT library/manifest retain their source-matched hashes.

GPU job 7307 completes all 64 independent arbitrary-density source API calls
and five independent FCI complete-force cases, with zero skips and exit zero.
All selector-capture, 3e-10 derivative and exact-zero disabled-channel gates
remain unchanged. GPU job 7308 then completes 20 independently gated complete
PBE0 calls across both sizes, both geometries and both owners. Each baseline
call launches generic-low/seven/eight once; each candidate launches
narrow-low/seven/eight once, with no generic-low, rejected indexed or class-page
launches and no incorrect narrow-low CTA width. These instrumented admission
calls supply no timing qualification.

## Pilot decision and retained 96-atom parity rejection

Clean two-pair 48-atom pilot 7309 completes all 18 E/F calls and eight timed
samples in one image/process/context. The independent analyzer passes every
numerical, frozen-seed, available-work, resource and actual-point check. Maximum
E/F errors are 6.593837e-12 Eh / 2.477352e-11 Eh/Bohr; maximum cross-arm density
drift is 1.268030e-11, below the unchanged 2e-11 gate.

| Geometry | Incumbent median (s) | Narrow median (s) | Raw time reduction |
| --- | ---: | ---: | ---: |
| warm | 7.420320829 | 7.444088876 | -0.3203% |
| moved-warm | 7.344353950 | 7.332061991 | 0.1674% |

Neither reaches the minimum 2% promotion gate. Two pairs do not establish a
formal regression or improvement. All individual slow/fast samples remain.

96-atom pilot 7310 fails before any timed sample: independently converged native
owners have maximum density difference 6.331513e-11, violating the same 2e-11
input-parity gate. The matrices' difference is symmetric, RMS 9.024053e-13, with
the largest difference at diagonal (101,101). Construction requires 18/17 Fock
builds; both original-geometry setup replays require one build/iteration and
pass independent E/F gates. Physical residual maxima are 7.016907e-13 and
1.769991e-12. These facts distinguish a failed input comparison from an E/F
oracle failure, but do not prove its cause or excuse its rejection. No original
or moved 96-atom timing claim is possible, even though both geometries passed
the earlier separate actual-launch/E/F admission campaign.

Do not loosen density acceptance, import host checkpoint seeds, add a
benchmark-only owner-state hook, pool the incomplete record, or rerun until a
favorable seed happens to appear. The force-only change and byte-identical
generic body do not by themselves prove that the seed difference is harmless
floating-point variation. Independent owner convergence is the relevant
diagnostic boundary if a future measured candidate merits further investigation.

This source has no qualified endpoint benefit, so no five-pair promotion,
additional sanitizer campaign, automatic force default or performance PR is
warranted. #2220's separately qualified resident final-validation default
remains enabled. A successor needs a different measured work/data-movement
mechanism rather than simply relabeling this private-frame specialization.

All terminal receipts and NPZ inputs remain under
`.artifacts/ks-weight-stage/spd-low-v1/receipts/`; the original source/build
manifests and every failed attempt are preserved. Master advanced to
`98f6c20b8` (#2232) at the next meaningful boundary. Its CC triples/CI changes
do not touch the inspected force/provider paths and did not trigger blanket
retesting or transfer this frozen-source qualification to latest master.
