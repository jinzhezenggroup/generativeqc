# Decision: preserve unrestricted endpoint capacity ownership

Status: implemented
Date: 2026-11-04

## Problem

Composition review of the open-shell CBS and scalar-ECP branches found that
fitted UMP2 charged the retained fitted source both outside correlation and
inside the MO provider. CUDA UMP2 also admitted a complete molecular MO output
and sixteen such host planes, although it executes one occupied/virtual spin
channel at a time. Grouped CC diagnostics could omit a larger MP2 companion
from the planned endpoint peak. Review also questioned whether the shared
fitting selectors were intended to forbid explicit CPU-reference contexts.

## Decision

- Keep fitted preparation scratch in the external reservation, but credit its
  retained source to the provider that already charges the complete borrowed
  reference and fitted payload. Refuse accounting underflow; do not drop source
  or preparation coverage.
- Plan CUDA host/provider storage over active occupied/virtual spin channels.
  Charge the transformed provider output there and separately count direct,
  same-spin exchange, orbital-energy feeds and the scalar return. Retain the
  executor's independent numeric-capacity admission and checked arithmetic.
  This does not remove the fitted source's retained AO-rank-four tensor or
  change its algorithmic work; that source payload remains charged.
- Report the maximum complete grouped endpoint, including an MP2 companion.
- Apply the same fitted-source credit to CUDA UCCSD solve/response admission,
  while keeping CPU reported storage unchanged. Plan solver provider peaks over
  the actual spin-allowed direct/exchange request shapes, not a full molecular
  MO block; response still charges its genuinely executed full-AO requests.
- Reserve the still-live fitted source before a separately evaluated CPU UHF
  force component, in both UMP2 and grouped CC consumers. Reject exhausted
  budgets before allocating derivative work; sequential execution does not make
  the retained source disappear.
- Preserve the explicitly requested native CPU-reference fitted domains.
  Unlike restricted DF owners, unrestricted owners intentionally interpret
  the shared fitting selectors as operator policy and choose execution from
  the requested context. The Python frontend explicitly admits these CPU
  domains, independently qualified by the existing raw-Coulomb/PySCF energy
  and finite-difference force tests. This is not fallback from a CUDA request.
- Document that analytic CC forces require retained amplitudes. Unrestricted
  CBS remains strict FP64 even when its request records mixed preconvergence;
  tests distinguish requested policy from actual selected execution.
- Keep the sequential one-electron/ERI derivative stage reservation shared:
  the one-electron bound already includes the full 8 MiB source scratch, while
  the ERI batch is capped at that scratch bound. The admitted s/p/d shell
  fallback has smaller fixed expansion/result requirements. Do not add a
  second simultaneous reservation for non-overlapping consumers or pass
  arbitrary leftover endpoint memory to hide a missing bound.

## Evidence and invariants

Synthetic native shape tests exercise unequal/empty spin channels without AO
reads or a CUDA device. Existing independent UMP2 energies, gradients, orbital
response, source lifetime, exact-budget and one-byte-short gates remain in place.
No molecular endpoint, GPU timing or new-source performance qualification is
inferred from these host-only admission tests. Required CUDA 12.9 compilation,
runtime/scientific gates and the protected merge queue remain mandatory.

## Rejected alternative

Do not reserve an unexecuted dense molecular MO-output tensor merely to obtain a simple
upper bound, or omit an existing retained source to compensate for its duplicate
charge. Both obscure the true owner and make later budget composition unsafe.
