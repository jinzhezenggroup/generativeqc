# Decision: align the molecular VV10 density cutoff with GPU4PySCF

Status: implemented and qualified on the recorded endpoint domain
Date: 2026-10-21

Supersedes the density cutoff in the
[original WB97M-V composition decision](../architecture/2026-09-22-wb97mv-self-consistent-composition.md#explicit-nonlocal-density-domain).
Also supersedes the 1e-8 reference-mask choice in the
[October 1 matched-domain decision](2026-10-01-matched-vv10-comparator-domain.md).

## Problem

Treutler/Lebedev/NWChem grid alignment left an active-domain mismatch: native
used CPU PySCF's rho >= 1e-8, while GPU4PySCF 1.8.1 uses rho >= 1e-10 in both
SCF and nuclear gradients. This was incorrectly easy to interpret as complete
VV10 alignment. Default-grid WB97M-V/48 diagnostic reference runs changing only
that density cutoff differed by about 1.6% in warm time, so this does not explain
the observed several-fold native advantage, but the numerical contract matters.

## Decision

Version the molecular density policy and use rho >= 1e-10 in every native
consumer: compiled CPU SCF, resident CUDA SCF, complete force response, and
fixed-density first/second derivatives. Existing code generation imports this
single constant; its changed identity must invalidate affected artifacts.
Raw fixed-grid kernels retain their caller-selected positive-density contract.

GPU4PySCF also screens abs(weight) <= 1e-14. Native retains all nonzero signed
weights and their derivatives. This remaining distinction must stay explicit;
this change aligns the density cutoff, not every screening operation. Discarding
weight derivatives is a separate scientific-policy change and is not needed to
correct the density mismatch. No benchmark reference default is tightened.

Live matched-domain GPU qualification follows the new native threshold and
binds it in the protocol, rejecting historical 1e-8 receipts. Other historical
diagnostic helpers retain their explicit 1e-8 defaults. Archived evidence is
unchanged. CPU PySCF remains an independent 1e-8 oracle, with its domain
difference declared when used for molecular comparisons.

## Qualification

133 host tests cover boundary, inactive-row, signed-weight and policy identity
contracts; compiler dependency checks and applicable pre-commit hooks pass.
Slurm job 11407 passes all 39 resident six-channel force seed tests at 1e-10.
Source `bb25db89e` completes cold plus three warm WB97M-V Direct populations:
Standard3/12/48 and Fast12. Every sample passes energy/atom 1e-5 Eh, maximum
force 1e-4 and RMS force 3e-5 Eh/Bohr against independent XC5/6, VV10 4/5
references. Reference stability and same-grid CPU/GPU spots meet quarter gates;
the CPU oracle retains its independently declared density cutoff. Native and
reference producer revisions are preserved separately.

Standard48 warm is 7.035 s, versus 32.342 s in the separately recorded default
GPU4PySCF cutoff diagnostic. This limited comparison is consistent with the
earlier warm advantage; it is not a new all-method README ranking. Native
Fast12 warm remains slower than Standard12 on this branch, which deliberately
does not include the separate Fast warm-revalidation fix. No universal Fast
performance claim follows. Active-set crossing derivatives remain outside the
local derivative contract.

## Matched-domain CPU regression oracles

The existing strict CPU regression pins and complete-force finite differences
must compare the same density domain. Independently reconverging pinned
PySCF 2.14.0/Libxc 7.0.0 first reproduces the old 1e-8 constants, then changing
only its VV10 density threshold to 1e-10 supplies the new reference constants.
`tests/reference_data/molecular_vv10_cpu_oracles.json` retains both results and
separate downstream native comparisons; native answers are not oracle inputs.
The optional verification tools keep their upstream 1e-8 default and expose an
explicit matched-domain flag. Complete-force reference calculations select
1e-10 for the undisplaced geometry and every finite-difference point, restoring
the upstream function afterward. Numerical acceptance gates are unchanged.

Inactive-row fixtures now use a density below 1e-10, while explicit boundary
and formerly excluded tail cases verify both VV10 pair legs and resource caps.
These fixture changes do not relax screening, memory limits, or scientific
tolerances.

## Larger-basis CUDA acceptance follow-up, 2026-10-26

PR #2795 exposed one missed live-test migration: the complete CUDA force test
still forced GPU4PySCF to 1e-8 for cold and displaced geometries. Native already
used this note's 1e-10 domain. Water/def2-TZVP, local def2-TZVPD water and UKS
NH2/def2-TZVP therefore failed the unchanged 1e-8 Eh energy gate by approximately
2.50e-8, 2.56e-8 and 8.05e-8 Eh. Reproducing these on the earlier optimization
baseline excluded a new AOT regression but did not establish reference parity.

The test now imports the declared molecular threshold for both independently
computed reference endpoints and records that configuration in its receipts.
Slurm 14542 reproduces all three failures with the stale reference mask, then
passes all three full tests with the matched mask using the same frozen native
binary. No native arithmetic, tolerance or reference kernel changed. See the
[paired replay and complete receipts](../../../../benchmarks/results/cold-four-defaults-20261026/README.md#vv10-reference-domain-follow-up).
Keep explicitly historical 1e-8 diagnostics distinct from current-domain
acceptance tests; the historical data remains valid for its original domain.
