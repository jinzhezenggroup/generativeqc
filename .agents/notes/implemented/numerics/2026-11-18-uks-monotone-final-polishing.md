# Decision: retain beneficial UKS physical polishing without destabilizing DIIS

Status: implemented
Date: 2026-11-18

## Problem

Hosted CI run `44218958302` on `6682b45b7` passes core-b and compiler-heavy, but
core-c exposes two real VV10/rVV10 force-state regressions. Blanket retention of
the DIIS-certified density stops at physical residual RMS about `2.12e-11`;
the existing independently reconverged force/energy tests require `<1e-12`.
Both failures reproduce locally with the same values. Their tolerances are not
changed, and the intermediate run is not an accepted scientific result.

## Decision

Retain the [primary DIIS certificate](2026-11-18-uks-certify-current-diis-state.md).
During the existing four-step bounded physical closure, evaluate each projected
trial before replacing the certified current density. Ordinary exported CPU
states accept a trial only when its maximum per-spin physical residual RMS does
not worsen. Otherwise keep the current density and its matching physical Fock,
then apply every existing final-state gate to that retained pair.

This keeps useful final polishing for contractive VV10 force states, while
rejecting the destabilizing unaccelerated LTA projection. Generic non-export
curated occupation policy and non-ordinary correction behavior remain unchanged.
Both accepted and rejected physical evaluations contribute to the real Fock and
Coulomb/exchange work counters. Observe the simultaneous current/trial storage
before releasing either state. Do not change iteration budgets, the four-step
closure bound, numerical thresholds, independent references or export admission.

## Evidence

The initial adaptive correction passes all 38 combined automatic CPU KS and
VV10 self-consistency tests in 41.63 s. This includes LTA cold/warm/moved-state
oracle comparisons, the exact two formerly failing VV10 residual gates, and
independently recomputed displaced-energy force checks. Final-source cache,
build, CTest, Python and hosted receipts are retained separately under
`/data/jzzeng/pr3083-ci-20261118/remaining/`; an in-progress run is not a pass.

The final source-matched CPU build passes all 149 native CTests in 87.99 s.
The combined scientific regression passes 140 tests with 14 explicit skips in
151.10 s, including independent UKS/ECP forces, total-method UKS D3 second
order, native Hessians, automatic-program lifecycles and VV10 force/energy
checks. All pre-commit hooks pass. Its native source identity is
`c3f5c42f88206ca48ce26adcb0569487f483f560313410f00bce9a6251f70640`.

## Rejected alternatives

- Blanket suppression of final projection loses already-required force-state
  accuracy even though ordinary SCF tolerances are met.
- Unconditional projection can amplify a tiny residual and exhaust LTA's
  original SCF budget. Raising the cap does not fix the noncontractive map.
- Loosening the VV10 test or reporting an invented smaller residual would hide
  a real regression rather than preserve the existing endpoint contract.

## Revisit when

The shared physical-state certificate owns an explicitly safeguarded correction
policy and its storage/work accounting for both RKS and UKS.
