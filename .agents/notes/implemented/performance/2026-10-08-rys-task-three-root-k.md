# Decision: bounded three-root quartet-parallel exchange

Status: implemented
Date: 2026-10-08

The subsequent [unified K selector](../architecture/2026-10-09-unified-direct-k-selector.md)
retains this preference ahead of work scheduling on other eligible classes.
Lowering `incumbent` disables task preference only; also select task `fill` to
restore the defaults before both promotions. Historical measurements below
are not combined-policy qualification.

## Problem

The earlier five-class, 128-thread promotion was qualified on frozen base
`4385f727`, but its result does not automatically apply after main-branch AO/XC
promotions. On base `99a2a5cac926edc69c28453977d1bc9b70d8bd06`, exclusive-node
job 6680 gives incumbent/default 96-atom Cold medians 106.313781/107.403082 s.
Means are 106.354335/106.083368 s: only 0.25% better, with a 1.02% median
regression. Fock counts are `[17,17,17]`/`[17,18,18]`. This accurate negative
cohort remains in the receipts; fail-closed summary prevents its 48-atom run.
The prior frozen successes are not relabeled as current-main qualification.

## Decision

Extend the existing value-only lane-local producer to three Rys roots and at
most 64 Cartesian components, still restricted to s/p/d shells and at most p
on the fourth center. This admits `ddss`, `dsds`, `dpps` and `dspp` alongside
the original eight experimental capabilities. The measured preference on the
`sm_120` profile includes these four plus `psps,ppps,dsss,dpss,dsps`; other
classes/profiles retain incumbent and explicit `incumbent` fully rolls back.

36/54-component tasks use a 64-bit retained-component mask, including each
generated constant shift. Using a 32-bit mask here would omit high components
or invoke undefined shifts. Smaller tasks preserve their prior generated
32-bit source. The value producer reuses the same TRR/HRR state program and
existing strict high-accuracy Rys3 tables, not a new quadrature formula.
Three-root interpolation is inlined with bounded inner polynomial unrolling,
limiting coefficient lifetime without changing coefficient order or arithmetic.
The original one/two-root task mathematics stays byte-identical.

Restricted raw K permits at most 80 doubles in its private block contraction.
`ddss` needs 48 and `dpps/dspp` need 72; `dsds` needs 98 and therefore retains
generic scatter. This does not expand the old Hermite block experiment's
32-double limit. UHF, combined J/K and HF-weighted K retain generic scatter.
No new global ERI tensor, primitive cache, screening policy or queue is added.
The existing four independent warp queues, global bra head and admission
counters remain authoritative; no primitive/root loop adds collectives.

## Rejected alternatives

Do not fix the current-main negative by selecting favorable SCF trajectories
or weakening convergence tolerances. Unsetting the actual selector, complete
execution-Cold E+F, all retained converged observations, both mean and median
gates and real Fock counts remain mandatory.

Disabling inner interpolation unrolling for the two-root tasks reduces static
register counts (for example `psps` 254 to 158), but job 6683's fixed-density K
does not improve beyond noise: about 1.518 s at 96 atoms versus the original
128-thread implementation's approximately 1.52 s. That change is not retained
in the two-root producer. Static resource reports are not achieved-occupancy
measurements; NCU counters remain unavailable.

Initial latest-base jobs 6678/6679 unexpectedly overlap on node1 while asking
for the same RTX 5090 resource. The entire interrupted timing cohort is
invalidated before acceptance and retained, even observations made before
overlap. Subsequent real-GPU timing and qualification use exclusive-node,
finite Slurm allocations and sequential dependencies. This is a scheduling
exclusion, not a performance-based exclusion of a slow converged sample.

## Invariants and evidence

Preserve normalization, shell/component ordering, symmetry, FP64 precision,
Schwarz/cross-density screening, primitive-pair identity, bounded fallbacks and
construction-time owner freezing. Never retry a selected failed launch into
partially accumulated output. Class filter `all` must not broaden preference.

Prototype job 6685 compares the same binary on identical fixed densities:
96-atom incumbent/default medians 1.685846/1.339618 s (20.54% less), with
independent matrix gates. This is a mechanism experiment, not a Cold claim.
The final commented/formatted implementation has its own binary/source identity
and separately predeclared five 96-atom pairs and three 48-atom pairs.

Final-source job 6688 passes both 96-atom Cold gates with all five pairs:
incumbent/default medians 116.382170/101.296905 s (12.96% less), means
118.393776/102.965899 s (13.03% less). Fock counts are
`[19,22,19,18,19]`/`[17,17,17,17,19]`; the accurate 22-Fock control and slower
19-Fock candidate remain included. The full endpoint ratio includes changed
SCF work and must not be called a same-iteration kernel ratio. Maximum energy
and force errors are 9.095e-12 Ha and 3.786e-11 Ha/Bohr against the independent
cached GPU4PySCF oracle.

The same allocation's three 48-atom pairs also pass: medians
49.426070/45.583701 s (7.77% less), means 49.383835/46.033125 s (6.79% less).
Fock counts are `[19,19,19]`/`[19,19,20]`, retaining the slower 20-Fock candidate.
The largest 48-atom energy/force errors are 7.276e-12 Ha/1.657e-11 Ha/Bohr.

Final-source validation job 6689 passes 176 public/prepared Libcint K cases
(maximum matrix error 6.370e-12), 864 standalone Libcint matrices and 96 cases
per memcheck/racecheck/synccheck, with zero findings. Six separate fixed-density
processes compare incumbent, old component-Rys and the actual default with
identical admission vectors at four density scales. At 96 atoms, diagnostic
device-K medians are 1.681313/4.294012/1.339690 s respectively (20.32% less for
default versus incumbent), admitting 81,907,624 shell quartets. At 48 atoms they
are 0.951970/1.582291/0.749990 s, admitting 23,480,495 shell quartets. These
separate diagnostics neither add to Cold timing nor prove equal primitive/root
work. Maximum fixed-density matrix error is 1.910e-11.

Current CPU checks pass 248 tests with 18 oracle-dependency skips. All six
separately pinned incumbent, old component-Rys and old K-block profile source
hashes remain identical. Deliberate bundle corrections affect task sections in
sm_120 shards 1/2/3 and registry preference; shard 0 and portable artifacts
remain unchanged. Final endpoint, independent CUDA/Libcint, sanitizer and
work-count results are retained in the separately source-identified
[current-base receipts](../../../../benchmarks/results/rys-task-master-20261008/README.md).

## Revisit when

Any further root/component/contraction-bound or target expansion requires fresh
independent matrix/force gates, sanitizer checks, full admission vectors and
complete endpoints. Preserve valid negative cohorts, source identities and
actual Fock counts, rather than attributing all end-to-end improvement to a
same-work kernel ratio. Keep the
[earlier five-class decision](2026-10-08-rys-task-k-default.md) as historical
evidence rather than rewriting its measurements.
