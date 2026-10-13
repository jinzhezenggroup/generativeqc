# Proposal: exact VV10 force ingredient reuse

Status: experimental, default off; GPU and complete endpoints pending
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
