# Decision: do not narrow DF response panels to recycle projection storage

Status: implemented
Date: 2026-10-31

## Problem

The projection-storage policy treated any failure to fit the preferred panel
as sufficient reason to recycle a dead projection. That can replace a viable
499-column owned panel with a 25-column borrowed panel, repeating occupied
response work despite sufficient ordinary headroom.

## Decision

Keep the authenticated projection ownership and irreducible-scratch checks.
Under pressure, recycle only when the borrowed interval is at least as large
as the available owned scratch after retained factors. Equal capacity remains
eligible; a smaller borrowed interval retains the ordinary owned fallback.
The comparison uses subtraction after validating the retained size, so the
capacity arithmetic remains valid at the size_t limit.

## Evidence and limits

The host-executed planner regression covers the 499-versus-25 case, its exact
capacity boundary, mandatory storage refusal and size_t limits. This correction
does not alter scientific equations, resource limits or numerical tolerances.
Earlier device measurements remain bound to their original sources; no new
GPU endpoint timing or performance qualification is claimed by this repair.

This refines the pressure choice in
[the final-projection decision](2026-10-30-final-df-projection-and-pressure-scratch.md).
