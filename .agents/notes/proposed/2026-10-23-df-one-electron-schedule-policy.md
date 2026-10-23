# Candidate: use the existing one-electron schedule in the fitted force bridge

Status: implemented in candidate; complete endpoint qualification pending
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
