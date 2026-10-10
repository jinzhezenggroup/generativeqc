# Decision: Default exact RHF values to bounded automatic routing

Status: implemented
Date: 2026-10-10

## Problem

The [phase-local source](2026-10-10-rhf-phase-canonical-values.md) removed
repeated ERI evaluation in the qualified ethane cold endpoint, but requiring
an environment opt-in indefinitely would hide that benefit from normal use.
Resource legality alone does not establish payoff for warm references or
already fast generated shell schedules.

## Decision

Unset and explicit `auto` now select a conservative cold/domain heuristic.
It requires a fresh bucket, an f-containing domain with uncovered generic
Fock work, at least 128 Cartesian AOs and an iteration limit of at least
eight. The Cartesian threshold implies at least 256 MiB of canonical
values and keeps smaller low-setup references out of automatic construction.
The iteration limit excludes short requests, not a prediction that SCF
will perform eight iterations. A new supplied guess is not a proof of a
converged reference; known warm/reused buckets are rejected independently.

`0` forces the original schedule. `1` retains the original forced-resident
request, including smaller/warm eligible references, without bypassing
memory or scientific gates. Other values are rejected on eligible calls.
Fully generated or lower-angular schedules do not pay a phase setup merely
because they have sufficient memory. No molecule name or geometry key is
used for automatic dispatch.

Inactive owners use ordinary reusable bucket graphs. Only admitted values
use local graphs; lazy ordinary capture handles a bucket whose earlier
forced/cold execution had only resident graphs. Source lifetime, FP64
compensation, final physical Fock, census, finite audits and failure
propagation remain unchanged.

## Rejected alternatives

- Default every eligible call to the old `1`: memory-fit and a large iteration
  limit do not establish payoff, especially on warm or generated routes.
- Build on every warm call: the source is intentionally not retained across
  executions, so a quickly converged replay would pay a fresh construction.
- Change SCF convergence or skip final audits to predict reuse: scheduling
  must not alter the scientific acceptance contract.

## Evidence and revisit

Host boundary/injection tests protect parsing, cold/domain/size/limit refusal
and the existing resource/numerical contracts. Independent allocated-device
tests must distinguish automatic cold admission from warm, smaller,
lower-angular and tight-budget refusal. Complete default-auto endpoint
qualification uses new source/binary identities; historical opt-in samples
are not relabelled as measurements of this policy.

The heuristic deliberately leaves potential wins unselected. Broaden its
domain only with source-matched full endpoints and short-convergence cases;
it is not a universally optimal or per-iteration online cost model.
