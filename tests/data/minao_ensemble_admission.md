# MINAO admission oracle fixtures

The JSON contains eight independent metric/admission fixtures: water/STO-3G,
water/def2-SVP, methane, OH-, F-, Cl-, Na+ and H2. Atomic coordinates are in
Bohr. Target bases come from `python/generativeqc/data/basis_pack.json`.
Occupied ANO contractions/occupations use PySCF blob
`a2e714f9e4eb395904b115cb1153aa3715a35fc7`, the same pinned source as the raw
native projection; no native integral or eigensolver produced these numbers.

The independent Gaussian overlap calculation normalizes contracted primitives
with the analytic same-center radial overlap
`(2 sqrt(alpha beta)/(alpha+beta))^(l+3/2)`. Cartesian overlap moments are
expanded around the Gaussian product center; even moments use
`(2k-1)!!/(2p)^k`, multiplied by `(pi/p)^(3/2) exp(-alpha beta R^2/p)`.
Real-spherical target transforms follow the repository's public AO ordering.

NumPy solves `C = solve(S_target, S_cross)` and forms `D_raw = C occ C.T`.
After trace normalization, NumPy `eigh` supplies the metric occupations and
vectors of `sqrt(S) D sqrt(S)`. An independent breakpoint/active-set solver
finds the capped-simplex occupations. It enumerates breakpoints `-lambda` and
`2-lambda`, solves the trace equation exactly for each active set, and selects
the feasible interval. AO reconstruction is `X Q diag(f) Q.T X`.

`tests/python/test_minao_ensemble_admission.py` implements that active-set
solver and recomputes every expected density from the stored S and raw D. It
compares those values to source-linked native construction followed by the
unchanged global seed validator. The same executable checks full occupancy,
invalid/singular cases, deterministic output, and heap allocation bounds.
Random rotated SPD metrics and PSD inputs provide additional independent cases.

Raw native source projection is separately pinned by the water trace and
source-inventory assertions in `test_preliminary_initial_guess.cpp`. All raw
table bytes remain unchanged by the admission repair.
