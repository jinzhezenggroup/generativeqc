# Decision: require the native SCF effort protocol for built-in policies

Status: implemented
Date: 2026-10-29

## Problem

The native NUM03 effort policy already owns built-in next-step decisions and
final observable gates. Its Python transport nevertheless treated a missing
library or additive protocol as permission to run the old Python decision
implementation. That preserved two production owners before the first release,
contrary to the current repository migration policy and #2227 R6.

## Decision

Built-in policy inputs in the native integer domain require the selected image
to export both current effort protocols. Missing images/protocols, ABI mismatch,
load failure and native errors stay visible. Never continue to a second image
or retry the Python policy after selecting a native owner. Immutable loaded
image bindings and the native numerical/decision implementation are unchanged.

This supersedes only the old-library allowance in
[the original native policy decision](../architecture/2026-10-16-native-scf-effort-policy.md).
Calibration fitting and standalone prediction remain Python analysis work.
Custom predictor/policy/level types, custom string comparisons, and integers
outside uint64 retain their documented Python semantics. Treating those values
as built-in native inputs would alter user code or truncate hysteresis counters.

## Validation and scope

Failure-injection tests cover missing libraries and each missing protocol through
both public decisions, with Python retry forbidden. Existing independent
threshold tables, one-ULP boundaries, all final gate combinations, native error
recovery and custom/unbounded cases remain the behavioral contract. Current
source-bound complete CPU trajectories qualify unchanged built-in execution.
This does not implement the optimizer, migrate physical SCF stages, qualify a
new CUDA context, or complete #2227 R4/R6.
