# Decision: orthonormal Pulay metric for exact pure PBE CUDA UKS

Status: implemented
Date: 2026-11-10

## Problem

The original #2840 spherical input has a second failure mechanism before final
closure: raw-AO Pulay proposals can remain in a nonstationary cold-start cycle.
Rebuilding the current input identity exposed it on source `c342536f2`, library
`f7ff3bbc9f2b8db773a3db1156cd21ae400b5f51eae68e9c3f64b021cbd4058a`.
Slurm17085 failed spherical/mask0 E+F at the unchanged 200-cycle cap, residual
0.012109634239602205. Earlier source-matched passes are genuine observations,
not a guarantee against this sensitivity. The failure is neither a rejected
canonical-density export nor independent PySCF nonconvergence.

## Decision

Use `X^T(FDS-SDF)X` for the Pulay error history in the independently tested
all-electron, strict-FP64, exact-primary, pure PBE UKS domain. The physical AO
commutator remains untouched. Existing scratch already owns both products,
so no new device allocation or matrix export is required. The same transform
already serves the separate heavy-atom hybrid startup domain; its existing
admission and warmup remain unchanged.

Pulay fitting in nonorthogonal AO coordinates weights rotations according to
AO scaling. The orthonormal metric instead represents orbital rotations; it
changes proposals, not the physical Hamiltonian or final acceptance conditions.
Do not broaden this admission from a family label alone to hybrid, fitted,
range/nonlocal, mixed-precision, resource-budget or ECP owners.

## Controlled evidence

Slurm17090 compares the current binary with an identified experimental patch
on the same RTX5090, UUID `GPU-d05490f6-a867-d1a0-f13f-8bf4b1116387`, visibility3.
The original geometry, grid, tolerances, iteration cap and sequence are unchanged.
Experimental library `8ea05ae9e68682296397d7e19f4209001c32e7b6e884322457f8487aedeb0571`,
patch SHA256 `660c5a4836aa9caede45f182efa5d58f5c9f0ad7b621e14b29e0298a8425165c`:

- Orthonormal proposals: all eight original E/E+F x mask0/127 x Cartesian/
  spherical controls pass in 109.85s, including independent force references.
  Cold Cartesian/spherical endpoints take 32/47 iterations; moved endpoints
  take 19. These counts identify this experiment, not required implementation
  spellings or future regression limits.
- Current raw-AO control: seven pass, spherical/mask0 E-only fails at its
  unchanged cap. Keep this failure distinct from Slurm17085's E+F failure.

This is controlled numerical evidence, not performance, production, actual
Graph-launch or cross-device promotion. The experimental patch is not relabelled
as a measurement of the final guarded implementation; that needs a new receipt.

## Rejected alternatives and invariants

Do not raise iteration caps, accept a large commutator, weaken the 1e-8
canonical-density gate, replace the original quadrature, or seed from a CPU
production oracle. An acceptance shortcut cannot repair a cold trajectory.
Retain all failed observations rather than rerunning until a green sample.

Independent PySCF failures remain separate. Bounded Newton was rejected earlier.
Exact captured seed replays with `direct_scf=True/False` give identical histories
for all eight force-reference inputs; changing that flag has not demonstrated
an oracle repair and is not adopted.

## References

- `2026-11-10-original-ks-gate-completion.md` retains earlier completed and
  failed observations with their own identities; this note supplements it.
- `tests/python/test_ks_scientific_gates_cuda.py` executes original endpoint
  and independent E/D/Fock/force/canonical-state gates.
- `tests/python/test_scf_review_contracts.py` executes the metric admission and
  unchanged physical stopping boundaries without requiring a GPU.
