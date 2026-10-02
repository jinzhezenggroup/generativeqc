# Proposal: share VV10 energy denominator with every pair derivative

Status: experimental; independent and endpoint qualification pending
Date: 2026-10-03

The previous rational candidate uses two divisions for features and three with
geometry. Inspection of GPU4PySCF v1.8.1's Apache-2.0 VV10 implementation confirms
that its pair derivatives reuse the energy denominator. The original native
closure uses three/six divisions. This candidate derives a one-division closure
inside our existing compiler IR, preserving the energy root and runtime's ordered
pair traversal. No upstream kernel code is copied into production.

For D=g_i*g_j*(g_i+g_j), phi=-3/(2D). Therefore (2/3)*phi^2=3/(2D^2).
Multiplying this common factor by g_j*(2g_i+g_j) gives dphi/dg_i; swapping i/j
gives the other partial. Feature derivatives and the radial chain reuse them.
The squared phi remains finite and normal throughout the existing bounded input
domain. Derivative rounding changes deliberately under a distinct lowering ID;
energy arithmetic, screening and FP64 precision remain fixed. All exceptional
inputs retain the original ordered closure. Independent high-precision energy
finite differences and complete molecular gates must pass before promotion.

The two/three-division rational candidate remains a separate measured binary and
source archive under `.artifacts/wb97m-vv10-rational/`; its six complete GPU tests
pass in node1 job 5410. It is not evidence for this one-division candidate.
Current candidate receipts are under `.artifacts/wb97m-vv10-denominator/`.
Reference inspected: https://github.com/pyscf/gpu4pyscf/blob/v1.8.1/gpu4pyscf/lib/gdft/vv10.cu

## Qualification and promotion boundary

Seventy host tests pass, including unchanged energy-root identity, deterministic
standalone emission, exact ordered fallback, and independent 90-digit energy
finite differences for all consumed partials in both output-demand modes.
Slurm node1 job 5415 passes six independent complete RKS/UKS molecular and
displaced-energy tests, then a same-allocation matched endpoint comparison.
The water-24 label means 24 atoms (8 waters), 192 spherical def2-SVP AOs and
589824 unpruned points (48 x 16 x 32 per atom). All five sample pairs pass:
maximum energy error 4.105e-11 Eh, force error 2.985e-10 Eh/Bohr, against gates
1e-8 Eh and 1e-7 Eh/Bohr. Shared molecular VV10 density threshold is 1e-8;
reference direct SCF tolerance is 1e-14, energy/gradient tolerances 1e-11/1e-8.

Warm complete SCF + analytic-force median is 56.725950 -> 50.955343 s (1.113x),
with one SCF iteration for every warm sample. Cold is 348.478749 -> 346.142196 s,
with 19 iterations each. GPU4PySCF warm is 27.513170 s: the candidate remains
slower. Actual molecular active-pair counts are not exported; grid squared is a
capacity only. Pair traversal/reduction and geometric tile work are unchanged.
The force endpoint drops from 39.292 to 33.448 s, whereas the integral portion
is 18.031/18.062 s; this localizes the measured benefit to the grid/pair phase.

These measurements use archived source patches and pre-#1716 binaries:
geometry `f1253b0aa722ff119f6dc99ff61731f0da1ac473e38f53a8ef2233beb24a9c47`,
candidate `55e2041452b22ceca957fc9505351886c7eac257017a2c7a9e4a5d12c65c356f`.
Receipts are under ignored `.artifacts/wb97m-vv10-denominator/` in the composed
worktree. The source was subsequently rebased to the reviewed stack including
#1716 and rebuilt with verified ccache (new library
`b018e708c974597b27eaa94ccdd164c49e80d6ca3669c11a9f6e0b98974dec89`).
Latest-stack, larger and changed-geometry qualification are pending; the older
binary results do not qualify the rebased native stack.

This supersedes the four-division radial proposal in
[the earlier note](2026-10-03-vv10-reciprocal-geometry.md). The two/three-division
rational proposal was not promoted; its
[rejection record](../rejected/2026-10-03-vv10-rational-derivatives.md) explains why.
