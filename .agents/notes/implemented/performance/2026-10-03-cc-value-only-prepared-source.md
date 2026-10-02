# Decision: request only values from the prepared CC interaction source

Status: implemented
Date: 2026-10-03

## Problem

The CC owner queried CUDA Direct source storage at derivative order zero, but
constructed its prepared Fock plan from `make_hf_fock_spec()`, whose specification
defaults to derivative order one. The exact admitted value-only allowance could
not hold that different request. The optional source caught `std::bad_alloc` and
used its existing host fallback. Widening MO source tiles reduced transform work
but left approximately 51 seconds of host source generation at 28 AOs.

CPU reference preparation inherited the same default. Its full-exact prepared
plan computed coordinate-major integral derivatives even for energy-only calls.
Those derivatives were not consumed by RHF, the MO provider, or the later force
owner, which contracts separately admitted derivatives with final response weights.

## Decision and invariants

Set derivative order zero explicitly at all three CC prepared-source construction
sites: the optional CUDA source and both CPU reference/source cache owners. Keep
the global Fock specification default unchanged for actual derivative consumers.
Admission and construction must request identical source capabilities.

The existing optional CUDA source fallback remains bounded and available when
device admission or source construction is genuinely unavailable. Do not increase
its budget merely to accommodate unused derivatives, and do not remove the
fallback to hide this request mismatch. Force derivative execution, its resource
admission and its scientific acceptance gates remain unchanged.

## Evidence and regression protection

A node5 Slurm probe with 28 AOs queried 40,328 device bytes. Direct order-zero
construction succeeded at exactly 40,328 bytes. Prepared construction using the
default specification failed there with `std::bad_alloc`, then succeeded with
1 MiB extra and actually retained 296,589 device bytes for the derivative source.

The host execution mock now reproduces the real derivative-order-one default;
it rejects the unadmitted request instead of modeling the specification as an
integer. The native CUDA prepared-source test uses the exact queried order-zero
allowance, checks the retained bytes and derivative order, verifies device-only
source ownership and compares the returned tile with independent CPU integrals.

CPU public CCSD/CCSD(T) and force resource tests pass (27 passed, 12 CUDA skips).
Node1 Slurm job 5357 passes all 37 public CPU/CUDA tests. Complete endpoint timing,
matched PySCF oracle comparisons and sanitizer evidence are collected separately;
this source fix does not expand the public 12-AO force qualification boundary.

## Revisit when

Revisit the explicit value-only request only if a new consumer actually borrows
prepared derivatives. Its required capabilities must then be reflected in the
complete resource admission and validated on real hardware at the exact queried
allowance. A roomy standalone source test cannot catch this mismatch.
