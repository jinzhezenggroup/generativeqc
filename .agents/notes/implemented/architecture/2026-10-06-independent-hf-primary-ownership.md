# Independent HF lowering must own primary class work

Status: implemented, qualified; opt-in split execution is not profitable
Date: 2026-10-06

## Diagnosis

The independent lowering choices in #2020 reached HF's streaming launcher, but
normal generated pages still owned the primary fused incumbent contractions.
With no overflowing page, alternative streaming kernels launched with empty
class flags. Numerical parity and selected symbols alone could not detect this.
The default 48/96-atom warm and moved-warm HF observations admitted zero
quartets in every Rys-capable generated class; only native dddd did work.
These endpoints cannot qualify Rys throughput or selection on primary HF work.

## Decision

A complete strict-FP64 bounded independent J/K owner defaults every supported
generated class to primary streaming. The existing device flag reset and paged
exclusion make ownership disjoint: incumbent fused pages no longer consume
those classes before the two selected passes. This applies to explicit Rys
choices and the two-incumbent split control. Native dddd retains its exact
streaming contract. Compiler inventory remains the only coverage authority.

An explicit primary-streaming diagnostic mask still overrides this scheduling
choice. Default fused owners, mixed execution and partial/higher-l coverage
retain their existing paths. Immutable J/K choices remain prepared-owner
state; the existing diagnostic scheduling mask remains an execution control.

## Qualification

Require actual admitted work in a Rys-capable class on a small topology whose
ordinary pages fit their arena. The regression checks the native class ledger
under J-Rys and K-Rys without an explicit primary-streaming mask. Retain the
RHF/UHF, cartesian/spherical independent Libcint endpoint and geometry gates.
Repeat 48/96-atom HF fused, split-incumbent and split-Rys endpoints and collect
actual J/K class work. Previous default HF tail-only timings remain diagnostic
evidence of the ownership bug, not evidence for candidate performance.

Device-launchable graphs reject external event/D2H observer nodes. A diagnostic
observer therefore uses two tiny timestamp kernel nodes around each class and
samples outside execution. Work accounting is a separate run: counter atomics
may affect kernel time. Clean endpoint measurements load neither observer.

### Completed evidence

Frozen node1 release build at `2ab3e23fe` has library SHA256
`6cd79ab39a81d4eee229a87562a360c8ff74ea5bf257de74163655b598b9676d`.
All real-device work used finite Slurm allocations on the main partition with
one 5090. Job 6228 passed 18 independent Libcint/replay/primary-work checks;
job 6232 passed the unchanged PBE0 gates (432 inventory matrices, 12 prepared
J/K checks and 18 public endpoints). No compiled source changed during execution.

Clean HF endpoint job 6229 retained five warm and five moved-warm observations
per configuration, plus cold and moved setup, with all 72 numerical gates passing:

| Atoms | Owner | Warm median (s) | Moved-warm median (s) |
| --- | --- | ---: | ---: |
| 48 | Default fused | 0.742737 | 0.741348 |
| 48 | Split incumbent | 2.583615 | 2.599329 |
| 48 | Split Rys K | 3.072605 | 3.089016 |
| 96 | Default fused | 2.911810 | 2.915292 |
| 96 | Split incumbent | 5.919571 | 5.911804 |
| 96 | Split Rys K | 7.854530 | 7.876502 |

The HF API's `fock_builds` field is unavailable, not one. Actual execution is
established by separate timestamp job 6230 and admitted-work job 6231: every
warm/moved-warm callback executes 21 J classes and 21 weighted-K classes. All
18 selected Rys classes admit nonzero work. Warm J/K shell-quartet counts are
32,816,365 / 22,345,310 at 48 atoms and 142,757,104 / 69,602,656 at 96 atoms,
identical between the two split controls. These are admitted shell quartets,
not primitive quartets or root evaluations.

The timestamp-only class sums are diagnostic rather than complete build times:
48-atom warm J/K incumbent is 1182.927 / 1059.974 ms, with Rys K 1546.930 ms;
96-atom warm J/K incumbent is 2427.800 / 1986.947 ms, with Rys K 3955.497 ms.
Common density preparation/projection and non-class work are outside these
sums. Counter-instrumented times are not throughput evidence. Default fused
execution remains preferred; the independent route enables qualified experiments
and does not imply that splitting an existing fused algorithm is faster.

Receipts and the strict summary are retained in the frozen checkout's ignored
`.artifacts/jk-rys/{hf-endpoint-6229,hf-stamps-6230,hf-work-6231,qualification-summary.json}`.

Refs #2015, #2017, #2020, #2021; supersedes the pending HF qualification assumption in
`2026-10-06-independent-direct-jk-rys-selection.md`.
