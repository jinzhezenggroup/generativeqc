# Decision: align the molecular VV10 density cutoff with GPU4PySCF

Status: implemented, device qualification pending
Date: 2026-10-21

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

Boundary, inactive-row, signed-weight and policy identity tests cover the
contract. Device gates must include complete default-grid WB97M-V E/F and the
resident six-channel force seeds at the new threshold. Timings alone cannot
qualify the larger domain or mixed precision. Active-set crossing derivatives
remain outside the local derivative contract.
