# Decision: share full-range J/K derivative source work

Status: implemented; device numerical and endpoint qualification in progress
Date: 2026-10-02

## Problem

The PBE0 stationary shell derivative provider previously traversed the same
screened shell domain twice, once for J and once for K. At 24 atoms the full
native integral-source callback takes about 3.02 s even after independent grid
scheduling improvements. Both density contractions need the same full-range
integral derivative, but their observable source arrays must remain distinct.

## Decision

`execute_generated_full_range_energy_derivatives` clears two coordinate-sized
channels, submits one bounded shell traversal and downloads both together. It
borrows the existing three-channel force scratch already charged for the RSH
provider; no larger derivative tensor or new persistent allocation is required.

The bounded scheduler has a full-range separate-source variant. Screening,
quartet ownership, translation recovery, coefficient signs and output ordering
remain unchanged. Low-order shell workers contract weights into their recurrence
early; they still execute the two qualified contractions, but share screening
and queue enumeration. Generic order-four-and-higher workers evaluate the shared
derivative once, then scatter each separately computed density coefficient to
its own channel. The skip predicate requires both coefficients to be zero, so
cancellation between J and K cannot remove separately observable work.

Existing total-force callers keep the original single-output template
specialization. RSH radial operators retain their separate implementation;
zero-range or zero-frequency tricks are not used to emulate a full-range
source. Failure and exception paths drain the stream before host result storage
can be destroyed. Native force-to-energy-derivative sign conversion still runs
exactly once after successful download.

## Work and acceptance

The auditable provider counts change from two to one shell-domain traversals,
two to one downloads, and two to one generic derivative evaluations per
surviving AO quartet requested by both sources. Low-order recurrence work is
not claimed to halve. Complete endpoint time, not those counts alone, decides
whether the change is beneficial.

Host fault-injection tests cover every submission/download/fence return and
exception and preserve transactional output. Source ownership and coefficient
routing tests passed (81 tests before adding the explicit fused-work guard).
An independent libcint analytic derivative test compares both channels at the
same native final density for RKS, UKS, zero exchange and Cartesian/spherical
def2-SVP, including d-shell work. Its real-device gate and large complete
PBE0 energy/force endpoints must pass before this branch is qualified to merge.

## Rejected alternatives and revisit conditions

- Publishing J+K in one channel and zeroing the other violates the stationary
  source contract even if their final sum happens to be right.
- Reimplementing low-order recurrence algebra solely for this optimization
  would create another scientific owner. A future compiler-generated
  multi-output contraction may fuse it safely with independent qualification.
- Retaining all derivative integrals would remove the current bounded-memory
  guarantee. This change consumes each derivative before eviction instead.

Exact raw profiles, baseline library hash, compiler-cache statistics and build
logs are local under `.artifacts/pbe0-large-20261002/`. GPU work is scheduler
owned on the `main` partition; no device visibility overrides are permitted.
