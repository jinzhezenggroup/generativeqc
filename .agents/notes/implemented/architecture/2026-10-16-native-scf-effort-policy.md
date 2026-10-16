# Decision: native NUM03 SCF effort and final observable gates

Status: implemented
Date: 2026-10-16

## Problem

The force-aware research consumer already uses native SCF and analytic forces,
but Python previously selected effort, evaluated every empirical level envelope,
maintained relaxation hysteresis and classified strict final observables. Merely
adding an SDK wrapper around the old Python selector would preserve that runtime
ownership gap in #2227 R4/R6.

## Decision

One native scalar policy in `src/scf/effort_policy.hpp` owns next-effort decisions
and measured final gates. An installed versioned C ABI serves C++ and thin Python
adapters. The value-based controller returns detached state rather than retaining
a physical calculator, model or device lease. Transition identities/reason labels
and immutable dataclasses remain presentation; status precedence and numeric
decision branches do not depend on those labels.

Preserve strict reproducibility/missing-force/stationarity precedence, the first
admissible empirical envelope, immediate tightening and delayed relaxation. Check
the RMS forecast too even though only energy and max-force control admission:
the existing public estimate rejects a nonfinite RMS, so dropping that check
would loosen policy semantics. Express the streak comparison before increment
to preserve mathematical hysteresis at the largest uint64 counter without wrap.

Final verification is distinct from the empirical envelope. It consumes actual
strict-result observables, retains all six unmet gates and gives model mismatch
unverified precedence. The caller still executes strict cleanup and compares
physical identities. Neither a boolean identity flag nor an empirical envelope
authenticates a physical model or supplies a rigorous force-error bound.

Native decisions require no dimension-sized numeric allocation; work is linear
in supplied level count. Python marshals the typed level table explicitly. This
is not complete physical endpoint allocation accounting or a work-independent
capacity claim.

## Compatibility and rejected alternatives

- Do not silently retry Python after a native/load error. Missing additive
  protocol is a declared old-library domain; failures in a selected protocol are
  errors, not evidence of missing support.
- Retain explicit compatibility for custom policy/predictor/level types and
  counters outside uint64. Truncating a large Python streak/relaxation limit can
  change its hysteresis. Exact typed identity comparisons use the native path;
  custom string inequality stays compatible rather than replacing it with a
  different equality operator during marshalling.
- Reuse each immutable loaded library path rather than repeatedly opening one
  library per geometry step. A new path selects a new binding. Do not support
  in-place hot replacement of loaded machine code.
- Do not replace empirical calibration with a new physical theorem or copy
  SCF/force/optimizer equations. Fitting, independent measurement, provenance and
  public standalone prediction stay explicit Python analysis work.

## Evidence and remaining ownership

Tests cover independent threshold tables, inclusive one-ULP gates, full final
reason-mask combinations, state/history publication, calibration domain and
forecast overflow, native failure/no retry, output transactionality, SDK ABI and
custom/unbounded compatibility. Installed SDK tests use public headers only.
The complete endpoint runner reuses the existing FIRE/SCF-force research harness,
including owner construction, genuinely moved coordinates, all attempted steps,
strict cleanup, final verification and teardown, with measured calibration
retained separately and source-matched Python/native libraries.

This does not migrate physical progressive/force-aware SCF stage execution,
geometry optimization or universal precision control. #2227 R1–R6 remain the
full objective; standalone metadata gates are not complete runtime parity.

## Revisit when

The shared native prepared controller can execute the full qualified progressive
and strict-closure consumer without losing physical identities, measured error
evidence, provider budgets or unsupported-domain refusal/compatibility.
