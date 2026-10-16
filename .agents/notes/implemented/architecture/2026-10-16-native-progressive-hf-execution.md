# Decision: native bounded progressive-HF execution scheduling

Status: implemented
Date: 2026-10-16

## Problem

The deterministic HF controller already enforces immutable target identities and
independent physical verification, but Python also owned the physical stage and
prepared-owner cleanup schedule. Moving only forecasts or MINAO capacity queries
does not transfer that execution policy. R3 response/HVP is deliberately deferred;
this change advances the independent R4 runtime-policy slice of #2227.

## Decision

The selected target library exposes a private additive synchronous callback
controller. One native owner orders source preparation/execution, target
preparation, optional projection, mandatory target execution, conditional density
export, reverse-order owner closure and post-closure physical verification.
Callbacks only transport existing operations; no scientific handle or density
crosses this ABI. The controller owns no heap storage or global mutable state.

Source failure still permits the unchanged cold target. Target item-level failure
does not export a density or manufacture physical verification. Callback failures
stop scientific work but close every successfully prepared owner, including when
target cleanup itself fails. Python contains exceptions inside the callback and
re-raises the first exception with its traceback after cleanup; cancellation is
not allowed to escape through libffi.

## Compatibility and remaining ownership

Only the already-selected target library is consulted. A missing optional symbol
uses a named portable compatibility schedule. Native refusal, malformed receipts
and callback failure never retry it. Existing native numerical operations retain
their own admission and backend contracts. This is not new CUDA qualification.

Typed target/planning validation, projection transport, physical residual audit
assembly and final acceptance remain explicit Python owners. General
coarse-to-fine planning and force-aware execution are not migrated by this slice.
It does not complete R4, R6 or #2227. Keeping those omissions explicit is preferable
to calling a callback dispatcher a complete native molecular solver.

## Invariants and verification

- Only source success permits projection; it never replaces the exact target.
- Only target success plus convergence permits retained density export.
- Physical auditing starts after both prepared owners close.
- Every callback failure, malformed flag and failed cleanup preserves bounded
  cleanup, and a later/reentrant/parallel invocation has independent state.
- The native test exercises actual exported transport and transactional invalid
  arguments. Python tests exercise exception identity, cancellation, refusal,
  missing-symbol compatibility and independent PySCF RHF/UHF direct/DF E/F.
- Whole-controller endpoint timing must include preparation, projection, both
  solves, closure and physical verification; isolated callback timing is not an
  endpoint qualification or a speedup claim.

## Revisit when

Projection and physical-audit owners have complete native target/provider
contracts, or a general multistage policy replaces this intentionally bounded
two-stage HF contract. Preserve independent oracles and old-library behavior
until a deliberate compatibility decision retires them.

## References

- Issue #2227, R4 and R6.
- `docs/reference/python_contracts.md`, progressive lifetime and acceptance.
- `tests/native/test_progressive_hf.cpp`.
- `tests/python/test_native_progressive_execution.py`.
- `tests/python/test_progressive_controller_native.py`.
