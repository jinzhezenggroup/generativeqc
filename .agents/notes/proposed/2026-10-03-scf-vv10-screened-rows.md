# Proposal: skip MolecularV1 screened VV10 rows in resident CUDA SCF

Status: proposed; complete endpoint timing in progress
Date: 2026-10-03

## Problem and decision

The resident WB97M-V SCF owner always applies the MolecularV1 density domain
before evaluating VV10, in both ordinary and graph replay paths. Inactive rows
have negative-zero weights, placeholder density one, and zero gradient. Partner
compaction already removes them from the inner dimension. The caller omitted
the existing row-mask layout flag, so these discarded outer rows still visited
every active partner and computed potentials that weighted AO assembly drops.

Enable that existing flag for this owner. Its constructor already requires
WB97M-V MolecularV1 for resident nonlocal execution. Active positive-zero and
signed nonzero quadrature weights retain the established semantics, including
negative-weight underflow protection. Invalid inputs still set the domain's
sticky failure flag; skipping their row does not turn a failed call into success.
Stationary forces already mask screened rows and are unchanged by this patch.
There is no new density threshold, pair cutoff, kernel, allocation or CPU work.

## Evidence and limits

The host probe executes the production pair arithmetic for VV10 and rVV10,
including the feature-only SCF specialization. A test-only counter inside the
actual partner loop verifies that one screened row among three points changes
executed pairs from six to four, while retaining the total energy. Active
positive/negative/zero/subnormal weights retain energies, potentials and work.
All 33 signed-weight, screened-row, underflow and finite-difference tests pass
with a verified ccache compiler wrapper. This is scalar arithmetic qualification,
not CUDA scheduling evidence.

Node1 Slurm 5361 passes all six independent complete RKS/UKS WB97M-V tests,
including STO-3G, def2-SVP, TZVP, TZVPD and reconverged displaced energies.
The first library SHA256 is
`18a24079d31aa77aefa6c3d79e4a09adca26c7cc63b32228e1578b01a21597a5`.
An intrusive 24-atom def2-SVP, grid 48 x 16 x 32 one-iteration profile takes
16.137 s, versus 18.207 s in the earlier unmasked allocation; these separate
profiles are only bottleneck diagnostics, not controlled endpoint speedups.
Same-device cold/warm complete endpoints are running under node1 Slurm 5359.
Maintain the all-sample 1e-8 Eh / 1e-7 Eh/Bohr gates.

The 24-atom GPU4PySCF reference converges with the repository's established
1e-11 energy / 1e-8 orbital-gradient norm controls. An earlier diagnostic's
extra-tight 1e-12 / 1e-10 controls stalled and are retained as failed reference
evidence. With the established controls and explicit shared VV10 threshold,
the unmasked native cold/priming calls agree within 3.21e-11 Eh and 2.99e-10
Eh/Bohr. This changes no physical acceptance gate and does not establish a
timing comparison across node1 RTX 5090 and node2 PRO 6000.

Source patches/archive, build/ccache receipts and raw reports are retained in
ignored `.artifacts/wb97m-vv10-mask/` and the authorized remote task directory
`/home/jzzeng/codes/wb97m-20261002/`. Molecular active-pair counts are not yet
exported; grid-square capacities must not be mislabeled as measured work.

## Revisit when

If resident SCF admits other nonlocal density domains, require an explicit
inactive-row contract before keeping this flag. Row compaction, pair reduction
reordering and changes to scalar pair algebra need separate qualification.
