# Proposal: exact VV10 force ingredient reuse

Status: automatic candidate; default-route endpoint qualification pending
Date: 2026-10-14

## Problem

The bilateral FP64 schedule completes WB97M-V/48 but the 96-atom grid still
has quadratic work. Frozen-density replays may present exactly the same VV10
ingredients while rebuilding the complete SCF/force endpoint. Reusing the pair
component can remove repeated source work without changing its scientific input.

## Candidate and ownership

`GENERATIVEQC_VV10_COMPONENT_REUSE=1` enables a candidate only for a resident
VV10 force owner seeded through the exact KS feature bridge. Coordinates,
weights, parameters and device are immutable for that owner. A changed explicit
pair schedule invalidates reuse. Tiled feature collection retains unconditional
evaluation; rVV10 is outside this candidate.

On device, compare all bits of omega, kappa, their three local derivatives,
weighted density and rho. These are all density-dependent inputs of the pair
closures and row finalizers. The current generated local-scale function supplies
the key; there is no second mathematical implementation, tolerance or hash.
Gradient direction is consumed independently by the newly executed AO response.
Density-screen negative zero and observable positive zero remain distinct.

The previous six local panels remain in pair scratch. Its dead energy-terms
panel retains rho, and the first dead partner-index slot retains whether the
previous generation failed. Scalar pair energy is placed in the energy-terms
panel so it cannot destroy omega[0]. The six output seed rows remain separate.
Only 28 bytes are added: an equality flag and three 64-bit work counters. The
native capacity query charges them to the existing hard force budget.

Admission state 2 means exact-key reuse: both bounded and generic pair kernels,
the scalar reduction and seed packing preserve prior outputs. All branch
predicates are uniform within a kernel; no CPU transfer/fence is added to seed
collection or execution. Optional extended work queries drain already-completed
force work; the existing six-field lifecycle query remains fence-free. The
complete endpoint still recomputes SCF, integral derivatives, AO/grid response
and reduction. This is not endpoint-result memoization.

## Required qualification

Compare all six channels to the independent discrete-functional oracle for
signed, zero and screened weights, separate streams, exact repeats, one-ULP
density changes, schedule changes and error recovery. A poisoned prior output
must never be reused after a new domain reset clears its old error slot. Run
memcheck/synccheck, complete molecular E/F, frozen-seed warm and moved populations
before considering default promotion. Keep the 1800-second point limit and
1e-8 Eh / 1e-7 Eh/Bohr gates. Count probes, actual hits and actual pair sweeps;
grid-squared capacity is not observed executed work.

## Rejected alternatives

Do not cache a whole endpoint, use tolerance-based equality, accept pointer or
generation equality alone, or allow a failed generation to become valid because
its error slots were reset. A host read of the hit predicate was discarded to
preserve the existing asynchronous resident-force execution contract.

## References

#2426, #2454; the frozen bilateral integration is `7001858f1`.
Ignored trial receipts are in `.artifacts/qualification/vv10-component-reuse-*`.


## Automatic candidate and clean complete populations

The candidate now considers exact force reuse automatically for at least 65536
points, matching the large-domain pair schedule crossover. Snapshot seeding,
variant, coefficient, schedule and exact-key/error checks still apply.
Explicit `GENERATIVEQC_VV10_COMPONENT_REUSE=0` disables reuse; `1` admits small
qualification fixtures too. Default-route endpoint validation is still pending.

Clean frozen source `129d1fa3c` opt-in populations each pass all 12 calls:

| Atoms | Cold with preparation (s) | Warm median (s) | Moved (s) | Moved-warm median (s) |
| --- | ---: | ---: | ---: | ---: |
| 3 | 6.785415 | 0.496386 | 4.804617 | 0.497128 |
| 12 | 43.423775 | 3.034359 | 32.095988 | 3.041815 |
| 48 | 575.782563 | 38.714667 | 406.116415 | 38.650074 |

Slurm 8266 (48) and 8269 (3/12) record their assigned distinct GPU UUIDs and
check for pre-existing compute processes before starting. Every cold/moved call
reports one force pair sweep; each of the ten frozen replays reports one exact
hit and zero sweeps. Maximum E/F errors across these populations are
4.093e-12 Eh / 9.160e-11 Eh/Bohr. The 48-atom warm median improves from the
preceding frozen bilateral source's 67.707403 s without changing cold SCF work.
The original 1800-second point limit and all independent gates are unchanged.

Native and all 43 consumer file hashes have been reconstructed from the frozen
Git tree, available on `codex/benchmark-vv10-reuse-qualified-source-20261014`.
These opt-in measurements do not relabel the forthcoming automatic source.
