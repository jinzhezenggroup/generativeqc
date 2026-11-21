# Decision: Compose public CUDA Hessian capacity with prepared KS

Status: implemented
Date: 2026-11-21

## Decision

Compose the validated KS and applicable correction inventories with a Hessian
component before preparing SCF. The new component spans the prerequisite
phases: the SCF owner's full declared peak is a conservative bound on what can
still be live during response. Host terminal output and staging coexist; the
frozen nuclear direction is separate. Reserve at least the device result,
its independent native candidate, and the device direction for HVP. That floor
rejects obviously infeasible plans without claiming to size scientific response
work. Use the lesser of the endpoint cap and remaining effective device limits
as the hard response cap, then let the native owner admit shape-dependent work.

## Invariants and limits

Do not reuse Direct inventories for fitted SCF or a shared-grid inventory for
separate nonlocal correlation. Do not subtract assumed post-SCF releases from
the prerequisite peak or count unsampled driver/library memory as observed.
The batch validates its own provider requests separately from the explicitly
composed Hessian component. Response cannot consume native capacity promised
to another retained provider, even when policy allows a larger response cap.

## Evidence

Slurm `2225` on RTX 5090 passed the public Direct PBE matrix/HVP under a global
budget, alongside five unbudgeted fitted LR, independent VV10 and r2SCAN-3c
regressions. The resource test compares to the unbudgeted result and checks the
overlapping peak and retirement of temporary device charges. Host planning
tests cover distinct phases, nonzero device ordinal, pinned memory and reserves.

Source-matched Slurm `2228` passed all six again, including both rejection
stages: an infeasible public plan must not prepare SCF, and an admissible plan
whose cap cannot hold the native baseline returns an out-of-memory status
without retaining extra device charges. The latter is a declared capacity
failure, not an allocator rejection. See
`build/hessian-gpu-sm120/r2scan-orbital-f/hessian-tight-budget-regression-n5.log`.
The frontend and resource suites passed 144 host tests (11 GPU cases skipped).

## References

- [Native ledger decision](2026-11-21-cuda-hessian-device-ledger.md)
- [Fitted Fock device capacity](2026-11-21-cuda-prepared-df-fock-capacity.md)
- [Current Hessian contract](../../../../docs/developer/dft_hessian.md)
