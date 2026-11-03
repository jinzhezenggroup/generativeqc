# Budgeted RHF DF96 admission and negative autotune qualification

## Result and scope

The 96-atom water cluster in `tools/dft_mp_v1/inputs/water32.json` now executes
complete native RHF density-fitted energy plus analytic forces through
Calculator's resource-planned Python path. Both orbital and auxiliary bases
are explicitly **def2-SVP spherical, 768 AOs**, neutral singlet, batch one,
FP64. The native DF hint is fixed at **4 GiB**, with a separate **2-GiB force
allowance** and **12-GiB per-owner device ResourceBudget**.

The default resident warm complete E/F endpoint is approximately **5.23 s**.
Of nine policy attempts, seven complete five alternating-order pairs each,
one fails fixed-budget admission, and one is retired during startup/warmup.
**No candidate qualifies a >=2% endpoint improvement. Keep the native defaults.**
There is no selected policy, published execution recipe, profile installation
or global-default change. This is a resource-domain extension with measured
negative tuning results, not a DF96 speedup claim.

RHF DF timings must not be compared to the direct PBE0-RKS timings in the
separate DFT96 receipts: those are different Hamiltonians. Single cold/setup
records per DF owner are descriptive, not a paired DF cold-speedup result.

## Native ownership boundary

Fitted Python requests previously queried Direct-HF's <=16-AO small-eigensolver
inventory before reaching the actual DF owner. That unrelated plan is not
allocated by DF. Dispatch fitted requests directly to the authoritative native
DF inventory; Direct-HF's limits remain unchanged.

The native DF extension admits only explicitly budgeted RHF sources with at
most **768 orbital and 768 auxiliary AOs**. Existing source/DIIS/tile leaves,
ordinary AO/metric provider bounds, detached singleton retry, force staging,
overflow checks and observed allocation admission remain active. Unbudgeted
large resident requests, large UHF, larger dimensions and composed KS/DF
resource admission remain unsupported. The Python-free native CLI retains its
separate earlier small-DF preflight; this endpoint qualification is for
Calculator and the Python tuner/replay frontend, not that CLI.

The old 512-MiB hint is insufficient once fixed metric/DIIS/source state and
the half-sized force budget are charged. Eligibility by dimension does not
silently increase a caller's hint. Host capacity formulas retain their
historical conservative metadata overestimate: no practical finite host-budget,
RSS or total-process-memory improvement is claimed.

## Complete endpoint observations

Slurm jobs **1469** and **1471**, n4 / RTX 5090 / SM120, run serial pairs between
isolated resident owners. Each owner independently converges a native density,
freezes updates, warms once, and returns complete host E/F for each timed call.
No CPU reference density, orbitals or integrals enter the production endpoint.

| Explicit policy | Baseline median -> candidate median, seconds | Speedup / lower 95% | Decision |
| --- | --- | --- | --- |
| SCF exchange `dense` | 5.231838 -> 13.082688 | 0.399905 / 0.399623 | Reject |
| Final exchange `dense` | 5.232685 -> 10.938416 | 0.478377 / 0.478271 | Reject |
| Response storage `panel` | 5.231801 -> 10.238987 | 0.510969 / 0.510804 | Reject |
| Shared J/K source `0` | 5.231303 -> 5.234161 | 0.999454 / 0.999225 | No gain |
| Value mapping `primitive` | 5.232900 -> 16.273581 | 0.321558 / 0.321339 | Reject |
| Value storage `dense` | 5.234947 -> 46.375104 | 0.112883 / 0.112601 | Reject |
| SCF exchange `occupied` | 5.232552 -> 5.234051 | 0.999713 / 0.999381 | No gain |
| Value storage `packed` | No timed sample | Not applicable | Fixed force-budget refusal |
| Derivative mapping `serial` | No timed sample | Not applicable | Startup/warmup retired |

These are **70 paired endpoint observations = 35 pairs**, partitioned into
seven separate comparisons. Do not pool baselines or multiply speedups. All
seven completed comparisons pass independent numerical, both resource-owner,
native identity and semantic-work gates; they fail only performance promotion.
No accepted candidate exists, so no winning-policy final holdout is claimed.

Every timed sample retains the full **768-dimensional auxiliary metric** and
reports **four actual complete native Fock builds**. This is observed work,
not inferred from SCF iterations. Native energy tolerance `1e-12`, density
tolerance `1e-10`, screening `1e-14`, maximum iterations 100 and DIIS history 8
remain fixed. The baseline owned numeric device peak is **3,995,852,217 bytes**;
both resident contexts coexist. This is not total driver/context/library/RSS
accounting or a claimed memory reduction.

## Oracle and independent replay

CPU Slurm job **1468** uses PySCF/libcint 2.14.0 with the exact immutable
Gaussian input tables, not a potentially differently rounded named-basis
lookup. The oracle independently solves fitted RHF with full Fock rebuilds,
`1e-16` screening, energy tolerance `1e-12`, gradient stopping tolerance
`1e-10` and analytic auxiliary-basis response. Recomputed physical-Fock
orbital-gradient norm is **2.2137677305011807e-11**. Reference energy is
**-2438.354141054309 Hartree**. The independent metric's minimum eigenvalue is
`0.09304817858971107`, well above its fixed full-rank floor
`2.3994330289515964e-7`; no cutoff or precision is tuned.

Strict fitted-HF acceptance remains absolute energy error <=`1e-9` Hartree,
maximum force error <=`1e-8` Hartree/Bohr and force RMS <=`3e-9`. Independent
fresh native default replay in GPU job **1470** passes E/F, resource, metric-rank,
Fock-work and parity gates, with energy error `3.183231456205249e-12` and maximum
force error `1.946109939865437e-12`. The CPU-only checkpoint remains ignored;
its 4,796,856-byte file has SHA-256
`bd6d0860f40af2ee5cd14c4a7259a9396983fc27f3831c182169c04955713f23`.

## Refusals and cleanup follow-up

Explicit packed storage cannot hold its required factors and minimum tile
under this fixed force allowance; its native worker raises the retained
`MemoryError: hf: DF memory budget cannot hold the metric and one contraction tile`
before E/F. The original comparison helper then masks that refusal with
`BrokenPipeError` while flushing an already dead control pipe. Preserve the
original worker log and report rather than rewriting that history.

The maintained helper now tolerates that pipe-close error and retires the
remaining socket/log owners without replacing the original rejection. A
failure-injection unit test and independent real-GPU refusal probe in job
**1472** validate the fix. Both executed and maintained helper bytes/hashes are
retained separately; the cleanup fix does not change the native/source identity
or reclassify the earlier measurements.

The serial derivative owner never reaches a timed pair. An operator retires
only that verified UID/job-1471 worker after **565 s** of startup/warmup,
having exceeded a one-off 300-s startup budget. The earlier attempted overlapping
Slurm control step did not start and is retained as cancelled; the eventual
CPU-only signal helper verifies the target's actual Slurm environment. It
does not run GPU work outside Slurm or override device visibility. No serial
endpoint latency, numerical failure or qualified speedup is inferred.

## Identity, validation and retention

Canonical checkout and observed native source identity both are
`e176d1356a5855e93c8615827e80bb2c9e179860a5a41117872ff2bebe42c22a`.
Normally cached Release/SM120 library SHA-256 is
`f47d774c71325d4a48f0043498a7191698e364cfe153cebf30b2676d6837366c`.
The source changes are the resource-owner extension, not generated kernel,
scientific precision, basis, metric or integral algorithm changes. Compiler
commands and before/after cache statistics are retained.

- Native compiled DF fleet/capacity tests pass, including large eligibility,
  force/energy bounds, eigensolver reservation, rejection and overflow contracts.
- Targeted CPU suites: **193 passed, 10 opt-in GPU tests skipped**.
- Native publication/ownership/planning suites: **663 passed**.
- GPU regressions in job 1469: **10 passed, 71 deselected**; cleanup probe 1472 passes.
- Both default-native replay and all seven completed candidate comparisons pass
  their independent scientific/resource/work gates. No new execution policy wins.

The compiled native packed-capacity fixture is shape-only: its initial 8-GiB
hint cannot fit the required 4-GiB force envelope; the corrected 16-GiB fixture
passes. Initial failure and final success logs are retained. This is not a
16-GiB GPU execution claim; the real campaign stays at 4 GiB and rejects packed.

`receipts.json.xz`: **134,752 bytes**, SHA-256
`9ac5dd4982c286e456d0e078387220d7ac14350638c665b9759226a9ff6e060e`.
The separately reviewed compact archive retains full E/F observations, metric
and actual Fock counts, native resource admission, oracle, refusal/intervention
logs, source/binary identities and executed controllers. Native libraries and
CPU checkpoint binaries are not published. The retained `plan.json`, oracle
script and launch scripts define the exact workload and replay procedure.
