# Candidate: use the existing one-electron schedule in the fitted force bridge

Status: implemented in candidate; PBE gain qualified, 3c timing inconclusive
Date: 2026-10-23

## Diagnosis

The generic paired stationary Hcore/Pulay bridge accepts all four generated
one-electron schedules, including implicit triangular addressing for the
nucleus-cooperative schedule. Both DFT method call sites nevertheless force
schedule zero. The prepared Direct consumer already selects the cooperative
Hcore schedule under the shared policy. This difference is independent of the
fitted J/K approximation and of whether final D/W are resident.

At source `4f6558075d6484ac65b6b6852f3f71765757f84a`, separate traced water96
PBE DF and r2SCAN-3c DF cold-plus-warm runs spend 1.067 and 2.251 seconds in
the paired derivative kernels respectively. All eight calls in the four-case
Direct/DF diagnostic matrix pass independent endpoint gates. These inclusive
diagnostics identify work; they are not clean speedup populations.

## Candidate and invariants

Pass the existing `one_electron_derivative_mapping_requested()` policy to both
standalone DFT bridges. Unset uses the already implemented cooperative mapping;
explicit thread, shell-warp and serial controls remain available. No generated
recurrence, weight, sign, screening or precision changes. The method's final
token, resident D/W binding, staging allowance and exact bounded host fallback
remain authoritative. Paired outputs remain separate and drain before return.

The composition test executes every schedule with borrowed device D/W and checks
both source vectors and the existing resource bounds. Independent molecular,
moved-geometry, sanitizer and repeated endpoint qualification must decide
whether this candidate is retained. The original raw diagnostic profiles remain
under `stage7-current-profile` in the task artifacts on n1.

## Completed endpoint qualification

At c08d702293ec433321e9a97e48c3aa39f6a4f03d, five alternating fresh-process
PBE DF pairs improve cold median 20.615377 to 20.108672 seconds (2.458%) and
process-median warm 6.573549 to 6.242300 seconds (5.039%). Both exceed the
predeclared MAD-based floor. r2SCAN-3c improvements do not exceed its observed
variability and are not counted as qualified gains. The initial slower pilot
remains separate and fully retained. All 128 paired, pilot, diagnostic and
moved calls pass independent gates. Existing numerical/sanitizer tests pass.

See `benchmarks/results/df-one-electron-schedule-20261023/README.md` and its
offline reproducer for full populations, source identities, failures retained
and qualification limits. This is an incremental scheduling gain; the broader
Direct/DF cold comparison against GPU4 remains open.
