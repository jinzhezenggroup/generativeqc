# Decision: fail closed for native incremental KS resource inventory

Status: implemented
Date: 2026-10-11

## Problem

Native KS execution parses both incremental Direct-J/K selectors and admits the
mode only within its qualified strict-FP64 exact-direct CUDA domain. The native
CLI resource command bypasses method preparation. Its scalar KS inventory v1
has no incremental-mode argument, so it describes ordinary state even when an
incremental selector is enabled. Execution's allocator additionally owns anchor
and delta state, and incremental closure can retain extra history records.

## Decision and invariants

Validate both selectors before combining them with OR. Share only the exact
boolean-value classifier with method admission; each caller retains its own
error type and admission policy. An enabled selector returns unsupported JSON
before KS basis/layout work. A malformed selector remains a CLI usage error.

Unset, `0`, and `off` must preserve the complete ordinary plan and its identity.
Do not read tuning suffixes for disabled modes, or when an active mode has already
been rejected for lack of inventory. HF does not consume these KS selectors.

## Rejected alternatives and revisit condition

Adding a mode label to the ordinary plan would distinguish identities without
making its storage accounting correct. Calling full method preparation would
violate the metadata-only resource contract. Do not extend the scalar ABI or
copy new CUDA allocation formulas merely to silence the inconsistency.

Revisit when allocation-owned inventory represents incremental anchor/delta
state and closure-history lifetimes, with independent layout and admission
qualification. This gate changes neither the CUDA algorithm nor performance
policy and supplies no new GPU qualification.

## Regression contract

`tests/native/test_cli_runtime.py` covers both selectors and backend choices,
all four supported KS methods, eager malformed parsing, disabled byte identity,
active/inactive tuning suffixes, early rejection, and unaffected HF resources.
