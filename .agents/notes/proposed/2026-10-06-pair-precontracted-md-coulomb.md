# Pair-precontracted Direct J integrated with the existing compiler and owner

Status: implemented candidate; numerical and endpoint qualification pending
Date: 2026-10-06

## Rationale

GPU4PySCF 1.8.1 routes ordinary HF/PBE0 values through independent MD J and
specialized Rys K. Its J performs a density-weighted Hermite pair transform,
contracts Coulomb interactions in pair space and projects back to AO space.
Changing only an ERI recurrence does not reproduce that schedule. The six
installed Python sources match upstream tag v1.8.1 at
`5b284c258a4260baef80e3d150b4e7a81a9dbd57`; the retained source receipt and
CUDA implementation snapshots are under the parent's ignored artifacts.

The existing measured 48/96-atom PBE0 incumbent J complete-build times are
approximately 1.18/2.43 s. Rys K alternatives regress the complete endpoint,
so the next experiment moves the density contraction outside quartet work.

## Implementation boundary

The compiler owns the pair precontraction, reciprocal Hermite contraction,
projection and accepted class inventory. Native code owns the budget, resident
FP64 storage, stream ordering and preparation-time choice. The initial inventory
is derived from value IR order <= 2; unsupported classes retain their existing
exact owner. No molecule, method or handwritten class allowlist admits math.

Reuse the shared low-order Wick expansion, Boys evaluator, Cartesian component
ordering and signed primitive-pair coefficients. Gaussian decay is already in
the cache weight and must not be applied again. Pair density consumes D_ij+D_ji
between distinct shells, but the full Cartesian block once within a shell.
The ket Hermite parity applies to both reciprocal updates; identical shell-pair
products have half weight because both directions are accumulated.

`GENERATIVEQC_DIRECT_J_FOCK_LOWERING=md` freezes the optional owner. Its two
Hermite arrays are charged before admission; insufficient capacity retains
the incumbent generated owner. Density and potential arrays are refreshed on
every build, while primitive geometry remains preparation state. The result
adds to the current Cartesian J before the existing public-basis projection.
K and stationary force execution are not consumers of this new value lowering.

## Qualification and limitations

Require independent Libcint matrices for signed/zero/non-Hermitian restricted
and unrestricted density, signed primitive contractions, Cartesian/spherical
output, screening, preparation freezing and repeated/changed geometry.
Selected masks alone do not establish actual work: retain an admitted-shell-
quartet observer in addition to trace-only complete J and clean endpoint timing.
The five initial classes are partial coverage, so expansion is contingent on
larger-size profitability and independent numerical evidence.

GPU4PySCF 1.8.1 RKS uses delta density with a retained `vhf_last.vj` even when
`direct_scf=False` (finite Slurm diagnostic 6235). Native incumbent versus MD
comparison retains the same build protocol; GPU4PySCF numerical reference is
valid, but future equal-work timing must observe full density explicitly.
