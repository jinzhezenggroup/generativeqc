# Decision: separate application allowances from SCF AUTO

Status: implemented; wider public endpoint qualification remains in progress
Date: 2026-10-14

## Problem

The initial PBE0 request lived in a private KS-options extension and required
AUTO. HF rejects KS options, while several DF/RSH/composite force owners
require strict SCF. Neither restriction is a scientific reason to discard a
derivative error budget. Mirroring every backend's admission in Python would
create a second mathematical authority.

## Decision

Add a privately queried v2 method-descriptor extension. The stable public C
descriptor and ABI version remain unchanged. The three appended doubles carry
energy, maximum-force-component and optional RMS allowances. Preparation checks
sizes before reading the extension, rejects nonfinite/negative values and
force-only requests, and copies allowances into immutable SCF options. HF
bucket reuse includes this request in its execution identity. Non-mean-field
owners must not retain an extended descriptor by copying only its prefix.

Prepared Python signatures and the KS resource-request identity also include
the budget when present, without changing legacy identities when absent.
The dry-run capacity descriptor must carry the same v2 request as execution;
otherwise a strict-SCF request silently loses its optional FP32 grid reservation.
Native force-memory admission includes that optional storage envelope. The
resource-ledger test binds predicted and prepared capacities before executing
actual mixed grid work. Mutating the calculator's immutable policy invalidates
prepared reuse; the native-copy lifetime test instead mutates transient caller
descriptor storage after preparation.

The Python prefix retains its extension and nested pointer owners until native
preparation returns. Native options outlive those buffers. Strict reference
calculators never inherit either private transport, even after an AUTO
property fallback. Method resolution and auxiliary-basis selection remain
unchanged. Current libraries use v2; older compatible KS libraries retain v1.
Positive requests have one owner; a zero legacy extension cannot erase a v2
request. Legacy v1 still requires AUTO. V2 does not.

The facade can try strict SCF when AUTO is unavailable without dropping its
derivative request. Native XC warmup explicitly requires AUTO. Native force
capacity and execution share a budget/model gate independent of that choice.
Transport is not arithmetic admission and never produces fabricated work.

## Gradient publication

Private gradient v2 appends an execution-owned complete-model qualification
ID. V1 remains usable for PBE0, but refuses newer models rather than describing
them as PBE0. Unknown v2 IDs fail closed in the facade. The complete gradient
owner verifies the strict final density/Fock token and publishes only after
every source succeeds, including bounded numerical recovery.

This proof is independent of whether the SCF schedule exports a complete
operator inventory. A chunked strict SCF must not need invented operator rows,
or a gratuitous second endpoint, to publish its already verified mixed gradient.
Per-call observable errors are still absent unless an explicit audit executes.

## Scope and evidence

Host cross-language tests compile the C++ parser and consume Python descriptors.
They cover ABI size/alignment, every truncated size, malformed values, copied
request lifetime, nested KS ownership, strict seed selection, version rejection,
model-preserving fallback, and non-leaking audits. The independent graph
admission probe and old v1 checks remain. Related host suites passed 133 tests
before the final model-ID tests; the focused expanded policy/ABI suite passed
127, and concurrent-master resource/census regression suites passed 205.

Slurm 8260, transport-only library
`a9f4267b993196854100814f15061c79832123b5275d4b7c6c4e909b7ad5a75d`, passed
145 checks before three old malformed-v1 tests failed because they injected
into a transport no longer used by the current facade. Those tests now create
an explicit legacy request; separate v2 failure-injection tests were added.
The failed run is retained, not counted as a passing complete suite.

PBE/B3LYP/r2SCAN generated candidates have independent same-state gradient
evidence at 24/96 AOs (see the typed-semilocal note). Their new public endpoint
gate is intentionally no broader. HF, DF, RSH/VV10, 3c corrections, and the
README prepared/frozen/moved matrix are outstanding. The PBE0 milestone PR
must not be described as implementing the entire objective.

## Fair baseline requirement

Slurm 8270 passed 201 public/native/host checks and 64 complete singlepoint
observations using library `8ee40bb01f596cfeb3ba54c36edfc894efe6ad67e337a7ea93b84e2f9b5827ec`.
Its first comparison is retained only as a diagnostic: PBE/B3LYP/r2SCAN Fast
could use the newly admitted native force scheduler while ordinary strict
d-shell requests still used the incumbent phased owner. Attributing all of
that difference to precision would repeat the user's explicitly rejected
"ordinary optimization called Fast" failure mode.

Separate complete-model admission from arithmetic admission. Ordinary FP64
receives the same newly qualified native scheduling within its measured shape
domain, without requesting any FP32. Redo both numerical qualification and
complete endpoint timing against that comparator. Keep the faster incumbent
phased strict fallback outside the retained native domain. Never publish the
diagnostic pre-sharing ratios as the final precision speedup.

The first shared-baseline rerun, Slurm 8282, passed 200 checks but failed while
PySCF/SciPy computed the independent ammonia/PBE oracle (`eigh`/LAPACK internal
error), before that test executed a native endpoint. Its failure is retained;
it is not a complete passing suite. Slurm 8286 was manually cancelled to fix
the lifetime fixture after adding the intended prepared-policy mutation guard.
Final requalification and timings are separate receipts.

Slurm 8287, library
`89c9e3645fc38e1b42cac64cfa9dfc15899bb1a4a18a697dbf1dc8cf78100594`, passed
204 checks and all 64 alternated complete singlepoint observations. Each case
includes one runtime warmup and three measured repetitions per arithmetic mode;
construction, Hcore-start SCF, energy, all forces and cleanup are timed together.
This is not the README prepared/frozen/moved protocol. Maximum force error across
all observations was `4.238e-7` Eh/Bohr; every attempt passed independent energy,
maximum-force and RMS gates, without excluding warmups or mismatched iterations.

| Method | AOs | Shared FP64 (s) | Fast (s) | Ratio |
| --- | ---: | ---: | ---: | ---: |
| PBE | 24 | 0.571260 | 0.468040 | 1.221 |
| PBE | 96 | 3.677453 | 3.566240 | 1.031 |
| PBE0 | 24 | 0.599149 | 0.494175 | 1.212 |
| PBE0 | 96 | 2.127459 | 1.842673 | 1.155 |
| B3LYP | 24 | 0.626725 | 0.596624 | 1.050 |
| B3LYP | 96 | 2.201026 | 2.021003 | 1.089 |
| r2SCAN | 24 | 1.100092 | 1.038003 | 1.060 |
| r2SCAN | 96 | 4.520371 | 4.261834 | 1.061 |

Every shared strict median is below the corresponding retained pre-sharing
strict median. The new comparator therefore does not slow FP64 to inflate Fast.
The smaller gains are noise-sensitive, not evidence for a universal speedup or
the desired all-method completion. Raw attempts, XML, logs and source snapshot
remain under the ignored local/Slurm `readme-budget-fast-20261013` artifacts;
the follow-up HF build must not be substituted for this library receipt.

## References

- `../numerics/2026-10-14-typed-semilocal-fast-candidates.md`
- `../../proposed/2026-10-14-readme-budget-fast.md`
- `.artifacts/readme-budget-fast-20261013/`
