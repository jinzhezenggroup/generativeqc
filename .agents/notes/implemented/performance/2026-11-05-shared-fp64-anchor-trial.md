# Decision: share provisional FP64 anchors independently of accuracy profile

Status: implemented opt-in trial; strict default unchanged
Date: 2026-11-05

## Problem

The [execution-parity audit](../../proposed/2026-11-05-standard-fast-execution-parity.md)
identified a non-precision difference: only budget-qualified Fast WB97M-V
could reuse fixed FP64 J/K anchors. The strict prototype at local commit
`2b1737752301bac37fd959b46b514e50cf6f13f2` already established a bounded
opt-in design, but its retained research measurements were not this source.

## Decision and invariants

Use one complete-model/provider admission helper for strict FP64 and already
budget-qualified outer solves. Both use the existing fixed FP64 density anchor
and linear full/range J/K reconstruction. Nonlinear XC/VV10 consume total
density, full targets refresh anchors and final physical closure remains full.
Strict precision, convergence thresholds and independent evidence requirements
do not change. Ordinary AUTO does not acquire strict anchor admission.

Migrate current consumers to `GENERATIVEQC_KS_FIXED_ANCHOR_DELTA`, without an
old-name alias. Explicit zero disables both routes; explicit one opts in only
eligible owners. The default remains budget-qualified Fast at 192 or more AOs;
strict reuse remains default-off. Preserve DF, UKS, device-chunk and public
resource-ledger exclusions, optional allocation fallback, failure invalidation,
warm isolation and force-pressure retirement. The native `fast_anchor_*`
transfer-diagnostic member names stay unchanged and also report strict reuse.

Bind the selector to new checkpoint execution identities. Extend the native
independent-CPU fixture to strict owners with and without an application budget,
in addition to relaxed Fast and excluded strict AUTO. Count all retired optional
bytes under force pressure, including frozen-VV10 storage, without revoking
the final density/weights/token or double-retiring storage.

The anchor counters are native C++ transfer diagnostics, not Python/C ABI
transport fields. Public endpoint selection requires independent execution-work
evidence, rather than treating an explicit selector or an absent counter as
proof that the path ran. Diagnostic CUDA traces are separate from timing samples.

## Experiment contract

Use an exact-source SM120 Release image with fast-compile and test hooks off
for timing, after a separate hook-enabled native lifecycle qualification. Run
paired OFF/ON fresh-owner water-24 and water-48 def2-SVP endpoints, three
alternating-order rounds per policy, with three same-owner warm calls each.
Cold timing includes construction, preparation, synchronized complete energy
and forces, host/density export and teardown; intervening warm work is excluded.
Warm timing includes synchronized complete energy/forces and host work export.
Record full, provisional, mixed and anchor/delta work rather than equating a
density difference with omitted J/K applications.

Generate a fresh independent stock GPU4PySCF oracle with the same complete
basis snapshot and moving grid; no native density or result seeds the oracle.
Predeclare paired strict energy/max-force/RMS-force gates of 1e-9/1e-7/3e-8,
the existing independent gates and a warm median ON/OFF limit of 1.05.
An observed cold reduction of at least 2% is interesting, not default promotion.
Preserve failed attempts rather than excluding them from the evidence.

## Evidence

Current-source host/native-policy/checkpoint/admission/catalog/default-promotion
suites passed **655 tests, with four device-gated skips**; the expanded endpoint
runner/failure-injection suite passed **82 tests**. The source-authenticated
native CUDA fixture passed in finite n1 Slurm job **15934**, including all four
precision/budget modes, independent CPU energy, warm/failure recovery, capacity,
force-pressure retirement and preservation of density/weights/final tokens.

Normal SM120 Release timing used a separately rebuilt hook-free library with
fast-compile off and verified sccache 0.16.0. Source identity is
`0ad22da997d05751e2e2afd420ca0d0ed95a8491fbfb066afb3a09a0ab02e372`;
the production library SHA256 is
`a1b97c319f7855a3763c1965e172ec0cf0b08e6ccc9483577060c1b68b4cb281`.
The dirty measured base is `c511f9a330ef4e6c7d48e6ed4a5f778bff260001`;
the retained patch reconstructs the measured code instead of relabeling earlier
prototype timings.

| Standard workload | OFF cold median (s) | ON cold median (s) | Median paired reduction | Warm ON/OFF |
| --- | ---: | ---: | ---: | ---: |
| Water-24, 192 AOs | 50.320683 | 48.756945 | 3.0507% | 0.999808 |
| Water-48, 384 AOs | 161.042278 | 156.908628 | 2.6165% | 0.998864 |

The reduction is the median of paired ratios, not the quotient of marginal
medians. All twelve Standard fresh-owner endpoints and their thirty-six warm
calls pass the matched independent energy/force gates. The J/K application
census is unchanged by ON/OFF: water-24 has 27 SCF iterations, 28 Focks, 28 J
and 56 K applications; water-48 has 28 iterations, 29 Focks, 29 J and 58 K
applications. Both have five full nonlocal target iterations; provisional
counts are 22 and 23. Each warm replay has one iteration, two Focks, two J and
four K applications, and no provisional nonlocal work. All are FP64 with no
execution retry or precision fallback. Delta reuse does not eliminate these
operator applications.

Separate CUDA/NVTX diagnostic traces of the frozen production image observe
21/22 delta-density preparations for the two Standard ON cold solves, zero
for OFF, and zero in all three warm calls. These prove actual public-path
selection, not an exact public native anchor-counter census. Profiler timing
is never a performance sample. Five additional FP64 matrices and one scalar
remain optional, charged storage; native fixture capacity is not a measured
public endpoint memory peak.

The complete trial remains **inconclusive**, despite encouraging Standard-only
observations. Every Fast OFF/ON attempt in this native-library-only fixture
fails closed because the packaged `wb97mv_rks` stationary CUDA force artifact
is absent. Converged SCF alone cannot qualify a complete energy/force endpoint;
these failures supply no performance comparison and do not establish an
anchor-induced regression. No endpoint peak-memory or single fresh total
compilation-cost receipt is available. Keep strict reuse default-off.

The [reviewed evidence bundle](../../../../benchmarks/results/shared-fp64-anchor-trial-20261105/README.md)
retains source/library/driver identities, independent basis/oracles, all selected
and interrupted/retried observations, predeclared gates, semantic work, trace
summaries/checksums and reproduction commands through the official evidence
publication validator. GPU/oracle/timing/profiler work uses finite `srun` and
preserves assigned visibility. The first 1800-second CUDA build timeout and
390-second water-48 oracle timeout remain explicit, followed by their successful
continuations. The 40-minute timing allocation also expires during the final
Fast owner; its separate retry preserves the original attempt rather than
overwriting it. These small shape-specific observations are not general
performance or default-promotion evidence.

## Rejected alternatives and remaining scope

Do not copy the earlier research timings or infer speed from anchor counts.
Do not remove application-budget capacity guards: mixed scratch needs its own
counted proof. Constructor resolution and large fitted force-tile admission are
still separate follow-ups, not silently claimed as unified by this trial.
Local Slurm only exposed the maintenance node, but authorized remote n1/n2
have independent Slurm schedulers; use finite remote `srun` jobs rather than
mistaking local scheduler availability for the availability of all allowed nodes.
