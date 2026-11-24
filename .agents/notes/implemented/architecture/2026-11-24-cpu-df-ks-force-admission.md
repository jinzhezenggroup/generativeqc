# Decision: Reserve CPU DF-KS force scratch by derivative lifetime

Status: implemented
Date: 2026-11-24

## Boundary

The existing native CPU DF energy inventory now optionally covers forces for
linked, all-electron FP64 LDA/PBE RKS/UKS with one full-range fitted J and
orbital and auxiliary shells through f. The CLI and Python ask the same
inventory. Python binds that quote to the exact native prepared method and
does not fund compatibility derivatives or upgrade a DF energy-only plan.
This does not admit fitted exchange, corrections, scalar ECPs, preliminary
guess providers, or second derivatives.

## Allocation choice

Keep the prepared raw metric and three-center tensor, fitted factors, solver
and grid reservation. The force frame retains D/F/W/C and, for UKS, must fund
the spin-summed density used by J. Metric/three-center reverse weights keep up
to three public-size three-center buffers live. That peak ends before the
Cartesian pullback; count the larger peak, not their sum. The pullback charges
both public weights and their Cartesian representations plus projections.
It accumulates generated derivatives into atomic coordinates, rather than
materializing coordinate-indexed dA/dM. The materialized diagnostic override
is consequently rejected at planning and before a budgeted force replay.

Reusing the Direct ERI-gradient allowance or interpreting the fitted provider
byte option as a CPU force cap would not account for those lifetimes. The
reservation is a modeled numeric capacity, not a process RSS limit.

## Evidence

Independent same-grid PySCF energy and analytic gradients qualify neutral
PBE RKS spherical water/def2-SVP with def2-TZVP auxiliary and distorted water
cation PBE UKS with the spherical STO-3G auxiliary default. Budgeted versus
ordinary native results and work counts match at the starting geometry;
companion checks cover one-byte-short refusal, no Python science fallback,
and moved warm replay against a fresh calculation. An exploratory OH doublet
was not used as acceptance evidence because the two solvers found different
SCF states.

On the installed CPU Release build with one BLAS/OpenMP thread, paired
endpoints included estimation, preparation, energy and forces, and close in
the bounded arm. For the neutral water case, levels 0 and 3 reserved
3,758,240 and 28,981,328 bytes in both CLI and Python. The paired median changes were +3.13% (36
pairs) and +0.33% (16 pairs); both arms used 13 SCF iterations and 15 Fock
builds and returned identical energy and force components. This qualifies
only these timed endpoints, not other hosts, grids or provider combinations.

Related: #2227 (R2/R6), `src/scf/ks_host_resources.hpp`,
`src/scf/density_fitting.cpp`, `tests/python/test_dft_df_public.py`.
