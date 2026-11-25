# Decision: Charge packed ERI values to the CPU Hessian host lease

Status: implemented
Date: 2026-11-24

## Problem

The CPU Hessian resource inventory already modeled explicit packed Coulomb
ERI storage, but execution excluded it from the native host lease. A modeled
fit alone cannot arbitrate against another prepared owner sharing the pool.

## Decision

Use the existing packed topology inventory and reserve the actual retained
value extent before copying and preparing the fleet. Keep the packed cap as
an independent sum over the fleet; it is neither a reservation nor an
authorization to switch to dense ERIs. The lease and its serialized workspace
are otherwise identical to the ordinary all-electron Direct Hessian lease.
Its scope remains energy replay and second derivatives, not standalone forces,
ECP provider lifetime, opaque BLAS memory or process RSS.

## Verification

The prepared Python contract tests exact and one-byte-short storage, shared
pool contention and rollback, and HVP parity with the dense native owner. The
CLI parity probe compares the dry-run host peak with the actual lease receipt
and refuses a one-byte-short request before SCF. Both are exercised without
changing the provider or derivative equations.

## References

- #2227 (R2), CPU Hessian/HVP host admission
- `src/api/ks_cpu_df_host_budget.hpp`, native fleet admission
- `tests/native/test_ks_host_hessian_frontend_parity.py`, installed parity probe
