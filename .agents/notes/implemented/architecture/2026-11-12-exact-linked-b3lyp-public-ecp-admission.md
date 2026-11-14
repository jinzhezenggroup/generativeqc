# Decision: exact-linked B3LYP public scalar-ECP admission

Status: implemented
Date: 2026-11-12

## Problem

The linked native CPU closure already evaluates the same scalar-ECP electronic
sources in retained-batch curvature controls. Python and the standalone CLI
still excluded all public full-range ECP Hessians. Removing the general ECP
restriction would incorrectly grant PBE0 and coefficient-variable compositions
the qualification of the exact linked B3LYP graph.

## Decision

Determine the exact canonical electronic B3LYP identity before Python composite
admission. Permit its scalar-ECP exception only on CPU with the existing direct
FP64, AO, dispersion and automatic-Libxc restrictions. Reuse that identity for
native dispatch. Missing linked electronic or total correction programs still
fail before public SCF, without a Python/JIT substitute. The native CLI already
authenticates the same linked total program before preparing its system, so it
no longer adds a blanket ECP refusal.

This is a Hessian/HVP interface decision, not ECP saddle admission. Both molecular
saddle consumers retain their independent ECP refusals. No force capability,
operator support, scientific tolerance or work allowance changes here.

## Evidence and limits

`test_native_cpu_ecp_admission.py` tests actual public admission/dispatch for
canonical external I28core/Cs46core inputs, RKS/UKS and optional two-body BJ.
It rejects PBE0, changed exact-exchange coefficients and DF before SCF, and
checks that Hessian admission does not promote ECP saddles. These are boundary
controls with a mocked native execution boundary, not numerical evidence.

The opt-in `test_native_canonical_heavy_hessian.py` can independently execute
retained-batch or fresh public endpoints. Both retain raw matrices, two HVPs,
cold central states and three displaced independent-gradient stencils with
electron-count and occupied-root gates. Complete public or installed acceptance
must be attributed to an actual run of that endpoint, not inferred from the
retained-batch results or these dispatch controls. Ni, complete 25--40-atom
electronic execution and requested transition states remain separate gates.

## Rejected alternative

Do not widen `full_range_stationary_second_order_eligible` into a blanket ECP
qualification. That function recognizes mathematics; exact linked ownership
and molecular/operator/resource qualification remain additional requirements.
