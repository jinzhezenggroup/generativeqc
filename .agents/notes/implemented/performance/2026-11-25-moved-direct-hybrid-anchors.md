# Incremental Direct hybrid Focks after a geometry update

Status: implemented in the candidate branch
Date: 2026-11-25

## Problem

The water96 timing in #3169 and #3176 used eight full-density Focks per
moved PBE0 Direct call. The default delta controller was already available
for the same prepared models during cold solves. Its solve-time check treated
a same-batch, last-good density as an imported seed after the geometry-specific
plan was rebuilt.

## Decision

The batch owner identifies a moved seed while it still holds the last
successful CUDA resident state. The newly constructed owner receives the
density and its provenance. The existing qualified automatic Direct hybrid
domain can start its delta controller from this seed.

The first Fock makes a full-density J/K anchor against the new geometry.
The updates stay within the existing bounded proposal policy and finish with
the consecutive full-density energy comparison and canonical state proof.
The previous geometry contributes the seed only. A rejected seeded solve
can take the existing one-attempt cold recovery. Frozen-density replay uses
its ordinary warm proof and imported seeds use their external admission.

## Evidence

The lower-overhead H100 progress journal on `22f72d069` records 4.04 s of
J/K and 2.24 s of XC across eight Focks on PBE0 Direct's first moved step.
Full tracing activates a census that bypasses MD-J and measures 13.53 s
for that J/K region. Keep those profiles separate. Pure-Coulomb PBE spends
0.32 s in J across the same number of Focks, so this change scopes moved
delta admission to hybrids.

On one H100 80 GB with driver 595.58.03, the same-image explicit-controller
precheck reduced the ten PBE0 Direct moved calls from 88.83 s to 78.47 s.
Each moved call used three delta builds. Both arms passed the independent
#3176 reference gates at steps 1, 5 and 10 with exact geometry identity.
These values qualify the mechanism in the explicit old-image control.
The automatic candidate has its own source-matched qualification.

For ωB97M-V Direct, the first moved solve spent 9.98 s preparing its
1.87-billion-value coefficient-bound resident source. Four VV10 pair
sweeps took 6.19 s inside the XC region. Neither of these owners is in the
full-range hybrid delta domain. For PBE/PBE0 DF, repeated XC and setup
dominated the parts of the call targeted by this investigation.

The short occupied-XC rollback kept the Fock counts unchanged in all six
semilocal/global-hybrid cases. Disabling the relative force responses
made both sampled PBE Direct moves slower. The diagnostic launch and
selector failures were retained with their receipts.

## Revisit when

Changed integral or geometry ownership could make an anchor from another
geometry available. Any such extension needs exact source identity, the
same physical convergence proof and independently qualified complete
endpoints. For pure J and fitted/RSH paths, profile the changed operator
and measure their own moved and stationary controls before changing defaults.

## References

- Follow-up and qualification scope [#3177](https://github.com/jinzhezenggroup/generativeqc/issues/3177)
- Original moving-geometry dataset [#3176](https://github.com/jinzhezenggroup/generativeqc/pull/3176)
- [Previous cold provenance policy](2026-10-23-cold-default-seed-provenance.md)
