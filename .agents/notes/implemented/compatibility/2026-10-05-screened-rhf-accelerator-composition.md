# Decision: compose screened RHF response with optional accelerators

Status: implemented
Date: 2026-10-05

## Problem

The landed response controls and optional preconditioning/recycling branch
independently extended the same solve and private endpoint arguments. A textual
choice of either side would remove physical acceptance or cache fallback behavior.

## Decision

Apply the optional inverse and recycled guess to the provisional operator. Keep
the zero-screening scalar audit and use the exact diagonal operator for a needed
correction. An unsuccessful unscreened optional solve likewise receives one cold
diagonal retry. Preserve attempted action, iteration, and preconditioner counts.
The screened warm vector remains covered by the existing extra-vector admission.
Publish only the exact audited image after all derivative gates pass.

Retain the established endpoint controls through argument 13, append new response
accelerator controls, and retain response options before the derived Boolean in
the native signature. Resource-only outer endpoint retry still happens once after
all failed phase owners unwind; incomplete abandoned work is reported as null.

## Rejected alternatives

Dropping screening discards the landed execution contract. Accepting screened
convergence without a physical audit changes the equation. A second accelerated
retry would not provide the bounded exact fallback. Guessing arguments from
numeric values makes existing DIIS and screening commands ambiguous.

## Evidence

`tests/python/test_rhf_response_accelerator_composition.py` runs the actual policy
with real GMRES against independent two-dimensional SPD operators, including
refused inverses, recycled guesses, failed exact audits, and physical errors.
The force selector and RCCSD forwarding tests compile extracted real signatures.
The cache admission and complete-phase retry tests execute production control.
These are host/source checks, not GPU qualification or endpoint timing claims.
