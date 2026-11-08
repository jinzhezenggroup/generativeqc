# Decision proposal: qualify SM80 shared Rys x/y without changing scientific algebra

Status: proposed; default promotion is not independently qualified at HF48
Date: 2026-11-08

## Problem

The non-CBS README A100 census does not establish universal superiority over
GPU4PySCF. HF96 Direct is the largest remaining cold deficit: roughly16s versus
6s. Repeating full64 or optimizing an isolated no-output consumer does not close
that gap. The actual DPPP page showed that recurrence evaluation, not the
tested output-lane gather, had an immediately reusable improvement.

## Proposed implementation

Reuse the existing independently derived unweighted x/y compiler companion for
exact SM80 in addition to SM90. Keep weighted Z SM90-only, DPPP192-thread and
PPPP96-thread mapping, the existing small-mask private fallback, primary kernels,
screening, accumulation and orbit scatter. Do not create another scientific
algebra. The generated companion capability remains authoritative.

Append an internal Auto request. Unset/empty XY with a valid off weighted-Z
control resolves Auto; explicit0 remains an immediate rollback. Auto requests
admission only on exact SM80. SM90 and SM120 defaults remain unchanged. Existing
restricted-FP64, capability, paged-materialization, no-incremental/separate-JK,
no-relevant-streaming, no-persistent-ERI and ordinary-output refusals remain.
Freeze the request/admission/class mask in the existing plan/cache identity.
No GPU product name selects this implementation.

The two standalone dispatch-fixture generators must migrate with the renamed
profile capability field. A host-policy unit test alone cannot catch a stale
generated C++ member reference.

## Evidence and its boundaries

The first8192 of147800 tasks from the authenticated A100 HF96 DPPP page retain
the actual consumer density, order800, observed early-page cutoff8e-8, ordinary
output and scatter. The full recurrence-plus-output median improves from
3.572224ms to1.640448ms at the actual density and from3.651072ms to1.681920ms
at a directed/asymmetric/zero density. All42 unfiltered samples pass full-matrix
gates. This is an early-page observation, not a strict-final census or endpoint.

Actual endpoint candidate source
`6f3ea6ead5de3dda0ab3a8e30c20540070f1cdbbdf0acf6c848dfa102c3000cc`,
ELF`5c222d026828561e768cb9926cd2694025ac94f904c46d2829a97d8e19bdd9f2`,
excludes all thirteen earlier adaptive-delta dirty paths. Incumbent source
`50ca688e7f9dc3d51c05f9aa336fab14226372fcbe8f970536062753a87ff6e7`,
ELF`2977049443dd116779ad583f67da20c902fcede2fdb0ae721d2995d8aee147ff`.

Four fresh processes per card/size, ABBA, complete cold E+analytic forces+host
export plus three frozen-own-density warms, both shared controls unset:

| Card/atoms | Control cold/s | Candidate cold/s |
| --- | ---: | ---: |
| A100/96 |16.142927356|14.624377469|
| A100/48 |6.068829596|5.702018391|
|5090/96|16.903542409|16.895718804|
|5090/48|6.325019540|6.339962678|

All120 native sample pairs per size pass E1e-8/F1e-7; every paired warm passes
the unchanged3% guard. Target iterations/Focks remain15 at96 and10 at48;
warms remain1. Invalid quartet counters remain invalid. HF still loses to
GPU4PySCF; these are improvements, not victories over that reference.

The independently authenticated tighter original HF96 oracle228320 passes
both cards' retained populations. Displaced/restored HF96 checks pass. A100
228865 and5090 16809 synthetic Libcint, memcheck, racecheck and synccheck pass
with zero errors/warnings. Mathematical fixture equality does not grant H100
performance authority. Actual A100 endpoint job is228864;5090 is16811.

## Independent HF48 failure: retain, do not reinterpret as pass

All sixteen A100 HF48 default observations, incumbent and candidate, miss the
additional absolute1e-7 force diagnostic by approximately1.818e-7. Energy
errors are about5.6e-10. Mutual native parity does not establish an independent
pass. The original README default contract has a different intensive force
gate1e-4 and RMS3e-5; passing it does not erase this stronger failed diagnostic.

GPU4 reference228912 with1e-11/1e-10 stopping controls and incremental default
Fock construction fails to converge after100 cycles. CPU16813 fails before
computation because the geometry import pulls in the native package;16814
also fails the excessively tight SCF criterion with default incremental Fock
construction. These failures remain evidence, not oracles.

Full-Fock/direct-screening1e-15 CPU reference16815 converges with orbital
gradient4.448e-11 and energy-1215.6178506877025. It confirms maximum native
default force error1.818374658e-7. The older GPU4 reference differs from it by
only6.531e-10 in force: insufficient reference convergence is not the cause.
GPU4 full-Fock228940 likewise converges and computes its gradient, but fails
afterwards in metadata export because its get_grad API needs an explicit Fock.
Its logged10-decimal gradient agrees with the CPU oracle to much better than
the force gate; the process still remains recorded as exit1, not successful.

Strict public native controls are a separate implementation diagnostic, never
a replacement for untouched-default qualification. Do not weaken1e-7, delete
the default failures, change the default convergence policy as a performance
trick, or claim a default gate passed because a strict solve did.

## Environment and source identity

CPU work is exclusively compilation or independent-reference validation.
Native production and timed endpoints forbid PySCF/GPU4 imports. Every real
GPU operation runs in a finite LSF/Slurm allocation with its assigned visibility.
General patchelf rewriting corrupted copied sanitizer executables; bounded
in-place PT_INTERP replacement with owned short loader fixes the environment,
not scientific code. Earlier failures and all original/repaired hashes remain.

Build16807's initial restore trap encountered generated pyc. A separate owned
CPU audit removes67 exact extra source files, retains243 caches, and verifies
original noncache source identity and original ELF before releasing the borrowed
build tree. Do not claim the first trap alone restored it cleanly.

Master PR3041 merged as`74d06855c`, integrated into this branch as`76343fcc4`.
Final overlay source`33db42ef97a4ce677b614a0900e18ba395c5837b7021c47df7f1e4ca1664d817`
corrects compiler docstrings and migrates fixture callers. Its independent build
and endpoint cohort carry their own identities;6f3 timings cannot be relabeled33db.
Final ELF is`60f40b1565c6147fec5cce0bcc77f16338a41eadcc436d8c8f675d08d1871c17`.
Build16816 and exclusive GPU jobs16818/229003 succeed, setup229002 authenticates
the complete reconstructed executable. Same-image explicit0/automatic ABBA
medians are A100 16.163515754/14.649752516s at96 and6.096549792/5.737883616s
at48;5090 16.905733509/16.886990728s and6.299592309/6.299205655s. Every paired
warm passes3%; all120 native sample pairs per size pass. Original/displaced/
restored HF96 passes independently on both cards. Strict HF48 controls require
21 target iterations and pass independently with maximum force error below
3.8e-11 on both cards, while every untouched-default HF48 observation still
fails1e-7. The strict result diagnoses an accuracy boundary, not a default pass.

Slurm16819 verifies all original noncache bytes, no remaining candidate-only
files, original source identity and original ELF after final restoration.
Relevant host suites pass61 tests; the compiler check covers587 modules with
zero errors. All adaptive-delta work and the original checkout remain untouched.

## Rejected alternatives and next decision

The tested combined-J/K output-lane gather is slower on both cards. Do not
expand it, repeat its census, or infer that every possible gather is hopeless.
Cooperative-roots and earlier adaptive work remain separate rejected/unqualified
experiments. No CPU seed, reference density, product-name special case, gate
relaxation, favorable-sample filtering or blanket full64 rerun is acceptable.

Finish source-bound qualification and report the stronger default HF48 failure
explicitly. A reviewable draft is appropriate while default promotion remains
blocked; neither an endpoint gain nor an exit0 sanitizer job resolves it.

## References

Issue2976; the source-matched capsule under
`benchmarks/results/a100-shared-rys-xy-20261108/`; local original arrays,
source/runtime deltas, failures and receipts under
`.artifacts/hf-sm80-shared-xy-v1-20261108/`.
