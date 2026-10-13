# Proposal: exact VV10 force ingredient reuse

Status: automatic route qualified; pending integration
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

## Measured qualification

The frozen integrated source `8493c02f0254cb4a401931acf5dcdbce02c88c78`
and library `3df2914c1eb415d446d63e340f9146eec4b4d28b063e028a113f82bbb93a9ee9`
pass 262 scheduled GPU phase/capacity/admission tests, 26 phase tests each under
memcheck and synccheck with zero errors, and six complete molecular E/F tests
against GPU4PySCF. The earlier molecular command skipped every test because it
selected the wrong suite/opt-in; that log is retained separately from acceptance.

Source `129d1fa3cc2a27e623bd7090189976de942f8817` adds benchmark retention of the
three measured force counters without changing the native inventory/library.
Both 3- and 12-atom controls complete all 12 calls and independent E/F gates.
Each cold/moved call executes one force pair sweep; all ten frozen replays have
one exact-key hit and zero force pair sweeps. These remain opt-in diagnostics.
They do not qualify native/96 completion or establish a default policy. The clean larger 48-atom population is reported below.

An initial 48-atom setup lacked its accuracy receipt. Cancelled Slurm 8255/8257
left Python children alive; their overlap contaminated Slurm 8258. Those three
own processes were explicitly terminated and their stopped outcomes retained.
The clean rerun records its assigned GPU UUID and verifies no pre-existing
compute process on that GPU before starting. No interrupted trial is reported
as an ordinary 1800-second timeout or clean endpoint timing.

On the PR tree, 96 host tests pass and 33 scheduled-GPU cases skip as intended.
The only native difference from the frozen experiment is removal of an unused
density-matrix comparison helper; force scientific arithmetic is unchanged.

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
qualification fixtures too. Default-route qualification is recorded below.

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


## Default-route qualification

Frozen integrated source `9aabd59a3a5f0e94ce9fb768646722d45e5da0a6`, library
`4c7b6ace6f07466af7967c0cc1f4453a7ff09a03c0ebcf13316527105e6779c7`, completes
all 36 original-default calls on Slurm 8284/8285, with every VV10 selector unset:

| Atoms | Cold with preparation (s) | Warm median (s) | Moved (s) | Moved-warm median (s) |
| --- | ---: | ---: | ---: | ---: |
| 3 | 16.330320 | 0.492156 | 4.608717 | 0.491725 |
| 12 | 42.786122 | 3.010895 | 31.844993 | 3.028606 |
| 48 | 578.965630 | 39.028821 | 409.380168 | 38.951268 |

Maximum errors are 4.548e-12 Eh / 9.160e-11 Eh/Bohr. Each cold/moved call has
one force pair sweep; every frozen replay has one exact hit and zero force
pair sweeps. The 48-atom warm median is 42.4% below the preceding bilateral
source's 67.707403 s. The original whole-point deadline and numerical gates
remain unchanged. No native/96 completion is established.

The first 3-atom cold includes 10.811690 s of force preparation. Separate
fresh-process retained-cache controls (Slurm 8290) give 6.378 s with automatic
reuse and 6.409 s with explicit opt-out; their warm medians are 0.492 / 0.599 s.
The first cold observation is retained in publication, not replaced by either
control. Preparation contains more than compilation, so no isolated JIT cause
is inferred from these intervals.

The same frozen integration passes 268 scheduled phase/capacity/admission and
SCF-work cases on n4, six independent complete molecular E/F cases, and 32 cases
under each of memcheck/synccheck with zero errors. n4 qualification explicitly
preloads the build's CUDA 12.9.1 user-space runtimes to avoid older optional
wheel SONAMEs; two failed prerequisite attempts remain retained. Formal timings
remain on n1 and check the assigned GPU UUID and absence of other compute PIDs.

The measured integration includes default-off SCF experiments and a broader
feature-only exact-cache guard; those do not activate in these plotted calls.
Compared with this PR, force arithmetic/ownership is unchanged; the integration
also retains an unused density-key helper. Native and all 43 consumer hashes
were reconstructed from the frozen Git source. See #2457 and
`codex/benchmark-vv10-automatic-source-20261014` for publication/provenance.
