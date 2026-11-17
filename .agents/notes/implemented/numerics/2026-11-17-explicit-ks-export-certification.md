# Decision: separate KS retention from physical export certification

Status: implemented
Date: 2026-11-17

The subsequent [energy-only consumer decision](2026-11-17-energy-only-ks-export-intent.md)
narrows which owners request certification; the export checks and measurements
recorded here remain unchanged.

## Problem

`ScfOptions::retain_ks_state` promises a read of the already evaluated physical
Fock, not a different occupation root or additional canonicalization. Using it
to request a stronger final-projector proof conflates two distinct consumers.
The existing retained LDA/PBE OH native fixture legitimately converges using
occupation stabilization, but an arbitrary unshifted frontier ordering can
replace that physically stationary projector.

Conversely, exempting every stabilized curated state from export certification
is incorrect. Commit `199084bb3` passes all 149 native CTests (130.82 seconds)
but fails the independent CsH(-) B3LYP and spherical open-shell mixed RSH/DF
force regressions: 43 scientific cases pass and two fail physical-state export.
Occupation stabilization can also occur before a nondegenerate exportable
solution. Its use is not evidence that the export projector may be skipped.

## Decision

Introduce the explicit internal `certify_ks_export` request. The production
CPU DFT method owner sets it whenever retaining a publishable physical KS
state. RKS and UKS apply the existing bounded unshifted final-projector closure
and unchanged RMS/max-entry physical tolerance for that request, whether or
not a prior SCF proposal used occupation stabilization. Automatic point-program
policy remains unchanged.

Retention-only SCF consumers preserve their ordinary stabilized-occupation
behavior and retain the unshifted Fock without claiming an export certificate.
The native fixture keeps all original retained/unretained OH and exhausted-
budget controls and adds successful certified H3 cases for both LDA and PBE.
It independently rebuilds the physical Fock/projector before checking its
maximum-entry density agreement. The independent public force regressions
remain necessary; the all-native pass alone did not catch this distinction.

## Invariants

- Do not weaken the final-state exporter, physical commutator/density/trace/
  idempotency/energy gates or independent oracle tolerances.
- Do not infer degeneracy or an export exemption from an occupation-stabilizer
  flag, and do not guess a frontier-gap threshold to bypass certification.
- Reuse the bounded correction/resumed-DIIS machinery and original iteration
  budget; no reference/PySCF production fallback is introduced.
- The two failed `199084bb3` force controls remain failures. Fresh build/source
  receipts and their reruns are stored separately under
  `/data/jzzeng/pr3083-ready-20261117/`.

## References

The source-matched cached rebuild passes the native UKS state group, including
the retained OH controls and both new certified H3 controls (3.01 seconds).
Both previously failing independent public force regressions then pass together
(45.19 seconds), without changing their tests, physical thresholds or oracles.
Broader final-tree qualification is recorded separately from these focused
controls and from the failed `199084bb3` campaign.

Supersedes the consumer-selection and initial degeneracy guard in
[the initial projector note](2026-11-17-curated-cpu-ks-final-projector.md), not
its source-matched historical measurements. Practical-scale Hessian/HVP,
Ni reaction saddle and raw continuum-rotation gates remain open.
