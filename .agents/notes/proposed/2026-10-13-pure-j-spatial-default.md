# Authenticate spatial force reuse for pure Coulomb sources

Status: proposed (complete automatic-default endpoint qualification pending)
Date: 2026-10-13

## Problem

Forwarding pure-J task demand fixes unnecessary K-only screening work, but the
96-atom direct r2SCAN-3c population still times out in its last moved replay.
The existing spatial derivative recurrence was qualified automatically only
for restricted global hybrids. Pure-J consumers used the older schedule even
though their full-range Coulomb source has the same spatial recurrence.

## Candidate

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

## References

Issue #2416; PR #2446; the implemented pure-J task-demand note; existing
`tests/python/test_pbe0_spatial_force_policy.py` executable composition gates.
