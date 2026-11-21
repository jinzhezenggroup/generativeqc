# Decision: Confirm only unresolved CUDA ECP Hessian channels

Status: implemented
Date: 2026-11-21

The real-spherical 29-AO CsH def2-SVP/ECP46 B3LYP calculation exposed the
same coarse nonlocal curvature miss as the
[CPU qualification](2026-11-04-scalar-ecp-hessian-bounded-validation.md).
The [CUDA diagnosis](../performance/2026-11-21-ecp-hessian-ao-row-pullback.md)
separated the local, nonlocal, and moving-density terms. For the unscaled
Cs-x direction the nonlocal discrepancy was `3.53e-7` Ha/Bohr², above the
unchanged `2e-8` derivative threshold. The fixed fine and independent denser
quadratures agreed at the original threshold. Replaying both channels would
take over `2^34` affine terms on this case.

Keep both baseline grids mandatory, and publish only their original fine
candidate. Only a finite coarse/fine miss permits one 320 × 60 replay of the
affected channel mask, with the same absolute gates and no cross-channel
cancellation. Nonfinite arithmetic (including overflowed differences), CUDA
errors, asymmetry and invalid ownership must not turn into retries. When the
replay cannot confirm the candidate, publication never begins.

Compute separate value and contraction admissions from AO, projection,
affine and reduction work and aligned storage. Deduct stationary staging and
the physical spin sum from the appropriate budgets first. Charge the ECP pair
before exact J/K, but allocate optional quadrature capacity only after the
required consumers have been prepared. An allocation failure can fall back
to the original pair, not to a host derivative. CUDA reuses the spent coarse
candidate slot and collocates each replayed layer once, skipping unnecessary
maps. Receipts expose attempts per channel, confirmed actions, attempted
radial layers, AO/projection/affine work, and a conservative reduction bound.

This is validation of a fixed model, not an adaptive production Hamiltonian,
a global error estimator, or a promise to admit arbitrary ECP families. Do
not loosen the gates or use a scaled direction to qualify a unit column.

## Production evidence

An SM120 SDK build ran without interposition on the RTX 5090 (Slurm 2112).
The explicit 16 × 6 × 12 and default Treutler level-3 grids produced all six
columns in 48.09 s and 49.75 s, and the original unit Cs-x HVP in 11.25 s and
11.58 s. The explicit matrix differs by at most `7.95e-8` Ha/Bohr² from the
retained 0.0003-Bohr independently reconverged Libcint/Libxc gradient stencil
in `benchmarks/results/scalar-ecp-hessian-2993/qualification.json.xz`. All six
rows of each of the default grid's three Cs-axis columns agree with the
independent reconverged energy Richardson stencil to `9.50e-8` Ha/Bohr².
Maximum raw asymmetry is `5.61e-13`, translation residual `1.78e-15`, and
matrix/HVP difference `1.31e-15` across both grids, with no symmetrization.

Both dense actions confirmed four nonlocal channels and no local channels.
Each standalone action attempted one nonlocal confirmation: 320 radial
layers, 66,816,000 AO evaluations, 10,690,560,000 projection terms,
27,572,160 affine terms and 10,693,295,360 bounded reduction terms. The source
and physical-endpoint regressions (Slurm 1696) passed 18 tests, including
two-grid-only capacity, invalid/stale inputs, and a finite but unconfirmed
quadrature that leaves every output unchanged. Eight separate native-KS
source cases (Slurm 1697) passed with direct/fitted, RKS/UKS and both AO
representations. CPU ECP, compiler and aggregate budget regressions passed
32 tests; the host-only admission probe passed separately.

The rebuilt SM120 wheel passed 21 isolated installed-artifact regressions
(Slurm 1698), including this physical CsH CLI matrix and the original unit
HVP through both CLI and public Python, matched to the independent reference.
The [wheel link note](../architecture/2026-11-21-hessian-aot-wheel-link.md)
records deployment constraints and the artifact identity.
