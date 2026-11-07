# Decision: contract the exact stationary RHF density products

Status: implemented
Date: 2026-11-08

## Problem

After correcting the MP2 energy source-scan schedule, complete #3034 replays
reached another independent amplification boundary: the stationary HF companion
force used the general factorized MP2 MO pullback and ordered shell quartets.
For the 97-AO case, that derivative phase remained active for more than thirty
minutes before the MP2 primal-cache phase could even start. Retained traces
bind this incomplete observation to library `08041f1c2047d555725861c0cbaaf9bdedb1609ef5f68446b3a8052972d7573f`.
It is a phase-duration lower bound, not a completed endpoint latency or speedup.

## Decision

Recognize only the exact stationary two-electron algebra already constructed by
the HF companion: the Fock cotangent is the unit occupied diagonal and every
correlation cotangent is exactly zero. Compute the one-spin occupied projector
`P = C_occ C_occ^T` in that same canonical frame. The eightfold ERI projection
of its AO cotangent is

```text
2 P_uv P_wx - (P_uw P_vx + P_ux P_vw) / 2.
```

Generate that bounded local cotangent directly for each canonical shell
quartet, multiplying by its distinct shell orbit rather than an unconditional
factor of eight. Repeated shells retain all local AO components; repeated atoms
still receive all separately differentiated center slots through the existing
consumer. The one-electron, overlap, ECP and nuclear consumers are unchanged.

This removes both unnecessary rank-four MO staging and redundant derivative
source evaluations. The extra projector is rank two, and projector plus local
shell storage fits inside the already admitted generic staging bound. No new
molecular AO-rank-four tensor, dense reference oracle or CUDA arithmetic kernel
is introduced. The host projector is production linear algebra, not a CPU ERI
or PySCF fallback.

## Invariants and fallback

- Recognition uses exact algebraic values, not a numerical cutoff on small
  correlation terms. General MP2/CC cotangents keep their existing path.
- The existing explicit staged layout bypasses this optimization and remains
  available for independent mathematical and resource validation.
- A resource failure retains the endpoint's existing bounded memory fallback;
  no new iteration limit, tolerance, screening or memory-budget relaxation is
  permitted.
- Timings and work counts must cover complete endpoints before speedup claims.
  The trace distinguishes stationary density products and canonical quartets.

## Validation

Native executable tests compare the density-product derivative with both the
ordered factorized MO pullback and an independently materialized dense MO
cotangent. S and S/P frames exercise repeated shells/atoms and multiple occupied
orbitals without relying on the optimized formula as their oracle. Existing
independent PySCF ECP/CBS component and composite gates and the complete pinned
#3034 two-step finite differences remain required; selected small-domain tests
must not be relabelled as complete iodine-CBS qualification.

## Rejected alternatives

Increasing Slurm time or device memory does not remove redundant derivative
work. A fresh ordinary SCF force calculation would duplicate the already
published physical reference and weaken endpoint reuse. Materializing molecular
AO rank-four weights would discard the bounded-storage contract. Approximating
small correlation cotangents by stationary weights is scientifically incorrect.
