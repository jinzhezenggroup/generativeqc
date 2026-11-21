# Decision: Include orbital ECP in each WB97M-V radial construction phase

Status: implemented
Date: 2026-11-21

## Problem

The shared ECP setup inventory was sufficient for a single early Fock
provider. Fitted WB97M-V may construct another Fock provider after the primary
and both grids exist. Its one-electron export includes orbital ECP regardless
of whether the late LR provider uses exact or fitted exchange.

## Decision

The radial query takes orbital ECP term count and uses the native ECP leaf.
Device setup is the maximum of S/T/V, ECP, and the relevant J/K source setup:
the one-electron export releases its CUDA staging before evaluating the ECP,
and both retire before the J/K source is allocated. Initial setup and late
excess above the late owner's retained allowance remain separate. The host
envelope conservatively includes ECP staging for each constructed provider.
Direct WB97M-V has no separate LR construction, and an automatic retry still
waits for the entire failed transaction to retire.

## Invariants

Do not mark ECP scratch as retained merely because the geometry and term
records stay live. Do not move the late ECP peak into initial setup; that
would let it borrow capacity occupied by the primary provider or grids.
The planner queries g auxiliaries using the WB97M-V **DF value** role only.
The native Hessian still authenticates the accepted physical SCF state and
linked second-order sources independently; this is not force admission.

## Evidence

The shape-only ECP test varies term count at a fixed basis shape and verifies
that initial setup and host bounds grow without inflating retained device
capacity. Late excess grows only when a separate LR owner is constructed.
The public WB97M-V ECP test compares budgeted HVPs to unbudgeted native CLI
results, asserts device ledger cleanup, and uses independent references.
Slurm job 2245 on SM120 passed all 37 selected tests, including Direct,
fitted-primary/exact-LR, dual fitted, and f-orbital/g-auxiliary ECP cases.
Job 2246 verified the budgeted dual-fitted ECP dense matrix against the CLI
and its product against the budgeted HVP.

## References

- `docs/developer/dft_hessian.md`
- `tests/python/test_ks_device_fleet_resources.py`
- `tests/python/test_wb97mv_hessian_cli_cuda.py`
- `2026-11-21-cuda-nonlocal-hessian-resource-ledger.md`
