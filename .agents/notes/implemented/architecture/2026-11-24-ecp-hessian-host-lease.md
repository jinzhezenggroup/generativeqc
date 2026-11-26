# Decision: Include scalar ECP setup in the native CPU Hessian lease

Status: implemented
Date: 2026-11-24

## Problem

An admitted scalar ECP Hessian could fit the CLI inventory while leaving native
prepared setup outside the shared host ledger. That fit did not arbitrate
against other live prepared owners.

## Decision

Authenticate the exact named MethodIR and ECP system with the existing native
Hessian gate. Use the same conservative ECP setup query as dry-run planning,
under the existing serialized primal workspace lease. The derivative's ECP
reservation and work caps are separate; neither is narrowed to the sampled
usage during preparation. A different geometry keeps its original provider and
topology while replaying under the lease.

Do not infer admission from the presence of ECP inventory alone. CPU DF lease
admission, all-electron-only Hessian bindings, force replay and unmodeled BLAS
or process storage remain outside this change.

## Evidence

The CsH def2-ECP regressions exercise the exact combined host ceiling, a
one-byte-short refusal before SCF, shared-pool rollback and unchanged HVP
publication from a leased prepared state. The CLI inventory also accounts for
the caller's direction buffer outside the native lease.

## References

- #2227 (R2), resource ownership for linked native CPU derivatives
- `tests/python/test_native_cpu_ecp_admission.py`
- `tests/native/test_cli_hessian.py`
