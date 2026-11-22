# Decision: Charge CUDA Hessian response to the prepared batch's ledger

Status: implemented
Date: 2026-11-21

## Problem

The Hessian owner enforced its own numeric cap, but several response sources
used raw CUDA allocations. A composition with the retained SCF capacity would
therefore undercount the simultaneous device footprint, even when the
Hessian receipt correctly included those sources.

## Decision

Use the existing native ledger's allocation and release operations for the
Hessian arena, optional dense storage, induced Fock, one-electron, exact J/K,
fitted response, raw DF and ECP arenas. The synchronous bridge binds the batch's
existing ledger under its context lock. After canonical-frame validation it
admits response against the smaller of the endpoint cap and the ledger's
unpromised capacity, including the external result and direction buffers.
Freeing a response must retire its charges without invalidating the SCF lease.
This reuses the existing ABI, preserves numerical options and fallbacks, and
does not bind a ledger for an unbudgeted batch.

## Evidence

On RTX 5090 Slurm `2224`, RKS direct and fitted HVP and dense fixtures passed
through the actual C bridge. Each first rejects a ledger with no spare capacity,
leaving the host result and receipt unchanged. A successful request's response
bytes fit alongside live SCF buffers in the measured ledger peak. Returning
from the call releases the temporary response charges; closing the batch
releases the retained SCF charges. The two-grid shared minimum also passed,
as did seven public CLI/Python regressions covering fitted LR, independent
VV10, r2SCAN-rVV10 and default-grid r2SCAN-3c.
See `build/hessian-gpu-sm120/r2scan-orbital-f/hessian-ledger-final-regression-n5.log`.

## Remaining composition

Do not reinterpret an SCF-only resource plan as a whole Hessian plan. The
public composition for inventoried SCF models is described in
[the follow-up decision](2026-11-21-cuda-hessian-public-budget.md). A supported
SCF resource request for fitted KS remains separate from this native ledger.
