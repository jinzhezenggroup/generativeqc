# Decision candidate: admit the independent strict range cache with fitted J/K

Status: implemented in candidate; default qualification pending
Date: 2026-10-23

## Problem and decision

The [held-factor repair](2026-10-23-held-density-factor-and-fitted-range-cache.md)
admitted only explicit fitted-primary range caching. A fresh compiled
[16 KiB stack candidate](2026-10-23-through-d-stack-headroom.md) now completes
WB97M-V DF96 cold/warm in 341.910032/46.329076 seconds. Both pass the unchanged
dense-reference energy and analytic-force gates. The concurrent fresh GPU4 DF
cold observation is 392.384761 seconds. These are single-process pilots, not
repeated default speedup evidence.

The same compiled predecessor passes 24 through-d/high-angular Direct GPU cases
and native restricted/unrestricted fitted WB, cold-factor and warm-factor
checks. Compute Sanitizer reports zero errors for the 24 cases and fitted WB
composition. Its Direct96 cold/warm pilot completes in 280.243997/30.068404
seconds. These controls remain bound to source `35f67156ebfd30028a58018bdb5ee2b6852c8ebc`;
they do not qualify the subsequently changed automatic fitted admission.

Extend the existing automatic strict range policy to the independent Direct
correction when its full-range primary is fitted. Keep the existing automatic
reuse, restricted VV10, grid-size, angular-domain and cold-phase gates. Retain
the 16 GiB optional ceiling, actual free-memory admission, both providers' host
accounting and exact streaming of omitted buckets. No primary fitted values,
metric cutoff, precision, screening, grid or convergence setting changes.

## Invariants and qualification

Explicit zero disables the optional source. Warm calls never build a quartic
source; policy changes retire incompatible leases. Public device ledgers and
unsupported mixed sources keep their existing fallback. Fitted final U remains
authenticated independently of optional range storage and can survive pressure
retirement at force handoff.

Extend the complete energy/force, changed-geometry and policy-invalidation
tests to fitted primaries. Independent CPU composition remains a separate
scientific oracle. Before promotion, qualify the compiled candidate itself on
clean repeated cold/warm/moved endpoints and Fast compatibility. Do not relabel
explicit-cache pilot measurements as unset-default results. No hand-selected
bucket mask or enlarged resource allowance enters this candidate.
