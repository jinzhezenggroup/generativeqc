# Authenticate spatial force reuse for pure Coulomb sources

Status: implemented
Date: 2026-10-13

## Problem

Forwarding pure-J task demand fixes unnecessary K-only screening work, but the
96-atom direct r2SCAN-3c population still times out in its last moved replay.
The existing spatial derivative recurrence was qualified automatically only
for restricted global hybrids. Pure-J consumers used the older schedule even
though their full-range Coulomb source has the same spatial recurrence.

## Decision

Allow the method-owned hint for authenticated pure-J full-range exact Direct
RKS in strict FP64, at the existing 384-AO crossover. Keep all backend, source,
spin, approximation, resource, geometry-token, no-ECP and no-range/nonlocal
composition guards. An absent K source is distinct from a malformed present
K coefficient. D4/gCP owners may accompany pure J because they add independent
geometry terms; the existing hybrid-with-D4 boundary stays unchanged.

The generic integral layer still has no method policy. It resolves only the
captured automatic selector plus the authenticated hint. Explicit zero/unknown
selectors, missing spatial topology and capacity refusal retain the existing
bounded fallback. No screening threshold, quadrature, precision, SCF stopping
criterion or correction model changes.

## Qualification boundary

The existing opt-in `spatial-reuse` route with pure-J task demand is measured
separately at frozen integrated source `9e21b8cb6`. Automatic-default results
must use a newly built source/binary and original complete 12-call population
within 1800 seconds. Keep failed prior runs visible. Qualify 48/96-atom 3c,
smaller controls and semilocal pure-J controls before issue closure.

## Complete qualification

Frozen integration `aaec57399ef29f96981da368c3a8c6a91af74ae8`, native library
`766617414dad28c4a475d7c9fbf6caf69418ddc3ad1aaefba244c111fc6cc321`,
completed the original 96-atom 12-call population in Slurm job 8182 under
the original 1800 s limit. Cold / warm median / moved / moved-warm median
were 538.312 / 90.092 / 268.003 / 90.114 s. All observations passed the original
independent E/F gates (max energy 3.28e-11 Eh, force 2.19e-10 Eh/Bohr).

The 3/12/48-atom controls and PBE/r2SCAN 48/96 controls each completed all
12 calls. A diagnostic retry exposed a separate resource-dependent issue:
with only 8 GiB free, optional shell-owner admission can fail and leave no
stationary shell derivative lease. The same resource refusal was retained in
the initial control attempts; normal-resource retries passed. It is independent
of this method hint and requires a canonical Direct derivative fallback.
Neither these refused attempts nor the preceding 96-atom timeouts may be
removed from follow-up evidence. Control timing publication uses the ordinary
driver; error-detail wrappers are retained as diagnostics only.

## References

Issue #2416; PR #2446; the implemented pure-J task-demand note; existing
`tests/python/test_pbe0_spatial_force_policy.py` executable composition gates.
