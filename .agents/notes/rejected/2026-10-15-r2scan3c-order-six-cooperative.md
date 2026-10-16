# Rejected: reuse the cooperative recurrence for order-six 3c forces

Status: rejected; experimental implementation removed, no default promotion
Date: 2026-10-15

## Problem

The README's direct and density-fitted r2SCAN-3c curves both use the defining
spherical def2-mTZVPP basis, not def2-SVP. Substituting the latter would change
the method rather than optimize it. The retained 2026-10-13 direct 96-atom
warm sample takes 90.089 s, including 62.196 s of stationary integral
derivatives. Its fitted counterpart spends 2.915 s in integral derivatives;
it is not evidence that the same direct-force bottleneck dominates DF.

For the 96-atom water topology, def2-SVP has 224/128/32 s/p/d shells and
def2-mTZVPP has 288/160/64. The unscreened dd-by-dp shell rectangle grows from
2,162,688 to 21,299,200 quartets. These are inventory bounds, not executed
integral counts. More d shells amplify high-order work much more than the
768-to-1088 public-AO ratio suggests. Neither bundled snapshot has zero
coefficients for these elements, so removing padded primitives is not a fix.

The existing bounded schedule uses a component-local warp consumer for angular
order six and a shared cooperative consumer for order seven. The compiler's
cooperative recurrence already admits both six and seven, including their
independent center responses and exact Cartesian component screening.

## Rejected experiment

The experimental `GENERATIVEQC_DIRECT_SPD_FORCE_DOMAINS=cooperative-six`
captured a distinct prepared schedule. It kept the order-zero-through-five sources and
uses the existing complete 256-lane cooperative recurrence for order six.
Order-seven and order-eight acquisition, recurrence and source layouts stay
unchanged. The candidate queue still admits at most 128 descriptors per packet;
the wider drain must not change its candidate coverage. Shared recurrence
storage replaces repeated component-local Coulomb preparations, without a new
derivative formula or a retained shell/primitive tensor.

The experiment retained the qualified spatial-reuse automatic schedule and
the cache, topology, angular-domain and recurrence-availability guards.
Explicit disabled/unknown settings and resource refusals retained the incumbent.
The existing schedule-setting provenance captured the experimental selector;
the intrusive census additionally identified actual shared recurrence work.
The rejected patch and independent tests are archived in
`benchmarks/results/r2scan3c-order6-20261015/`, not retained in production.

## Required acceptance before promotion

- Compile and execute the selector and dispatch/error-propagation contracts.
- Independently check separate and combined J'/K' against Libcint for Cartesian
  and spherical layouts and both spin domains; verify executed order-six work.
- Run real-device qualification only inside finite Slurm allocations, preserving
  their assigned device visibility. Use the same FP64 Release image for both
  schedule arms and retain actual source and binary identities.
- Compare complete cold, five fixed-density warm, moved, and five moved-warm
  endpoints against the unchanged reference gates, 1e-8 Eh and 1e-7 Eh/Bohr.
- Require a repeatable >=5% 96-atom warm and moved-warm benefit, no >5%
  48-atom/cold/moved regression, and small controls before any default or README
  timing promotion. Different SCF work stays explicit, never normalized away.

## Decision and evidence

The six independent real-GPU derivative cases pass (Slurm n5/1606): separate
and combined J'/K', both AO representations, restricted/unrestricted spin,
translation and executed order-six work. Complete canonical 12-atom endpoints
also pass the unchanged independent energy/force gates, but their warm median
rises from 1.625 to 1.870 seconds. At 48 atoms, five warm observations rise
from 22.476 to 34.911 seconds, about a 55% regression. These are same-binary
diagnostic observations, not source-qualified publication timings.

The intermediate image advertises an earlier build source identity than the
frozen experimental inventory, despite matching consumer hashes. Its numerical
checks and actual observations are retained, but it cannot update the README.
The final identity-refresh rebuild was stopped once the substantial regression
made promotion inappropriate. Do not repair the receipt by editing a manifest.
No 96-atom native candidate, DF speedup or default improvement is claimed.

Earlier full-endpoint pilots correctly rejected old stationary AOT plugins with
a `contract_identity` mismatch. Freshly built matching plugins allowed the
subsequent numerical checks; admission was not relaxed. A one-repeat pilot also
correctly rejected five-repeat reference protocols. All failed attempts remain
in ignored raw storage and the compact campaign evidence.

Sharing the Coulomb ladder is not sufficient evidence of an endpoint win.
This candidate widens the mixed low/order-six consumer to 256 lanes and adds
shared workspace and CTA barriers. Those are possible costs, not measured
occupancy/stall conclusions. Revisit with a separately bounded order-six drain
or a generated recurrence/liveness change only after independently measuring
its full endpoint, exact component screening, work and fallback behavior.

## Rejected shortcuts

Do not shrink the canonical basis, replace direct Coulomb by DF, relax
screening or convergence, reduce the cooperative block below all 256 lanes,
exclude D4/gCP from the total endpoint, or call diagnostic intervals complete
endpoint speedups. The earlier page-width and dddd-acquisition experiments
remain distinct negative evidence in
`benchmarks/results/bounded-dft-schedules-2475-2479/`.
