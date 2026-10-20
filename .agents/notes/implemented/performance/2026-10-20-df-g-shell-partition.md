# Decision: retain shell reuse around an auxiliary-g remainder

Status: implemented, qualification pending
Date: 2026-10-20

## Problem and evidence

The default GPU4PySCF auxiliary contains oxygen g shells. PR #2647 admitted
their complete native values and derivatives, but routed every three-center
derivative through the generic component consumer when any g shell was present.
Same-binary n1 PBE-DF/48 attribution at `02337e0cf` found 19.083 s warm, of which
16.602 s was three-center response and only 0.237 s was SCF. Substituting the
historical cc-pVDZ-JKFIT auxiliary for diagnosis gives 2.571 s warm and 0.134 s
three-center response. The old fit has more auxiliary AOs (1856 versus 1808),
but no g class and therefore retains shell reuse. This substitution is not a
README configuration or a proposed scientific default.

## Decision and invariants

Partition three-center response by auxiliary angular momentum. Existing shell
consumers own s/p/d/f; the unchanged generated generic derivative owns g. Public
AO offsets and dense response weights remain intact. Both metadata views carry
their actual expansion stride: six-term packing cannot be interpreted as three
merely because a particular shell has lower angular momentum.

The generic g remainder excludes all lower auxiliary classes, while metric
response always includes every auxiliary pair. The source already bounds orbital
angular momentum by f and auxiliary by g. Packed requests keep the dense weight
fallback required by the generic remainder. Partial s/p shell selection retains
its original generic complement. Mixed-gradient qualification is not broadened.
Resource admission, source identities, stream lifetime, independent physical
force gates and the explicit generic fallback remain mandatory.

## Qualification

Pending: independent CPU complete E/F and translation gates for the full,
partial, packet and generic routes; larger default-auxiliary cold/warm endpoints
with source/response work accounting. No default performance claim follows
until those checks pass. The generic g remainder can still dominate large cases;
full g shell sharing is a distinct future optimization.
