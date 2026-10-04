# Qualification: compose PBE0 SCF and force local AO on current master

Status: implemented qualification harness; combined GPU evidence pending
Date: 2026-10-04

## Problem and ownership

The earlier [force-policy composition](2026-10-04-pbe0-integrated-baseline.md)
does not contain SCF local AO admission. Its frozen endpoint measurements must
not be presented as results for this source or added to a separate SCF saving.

This checkout starts at master `8c1233a5c6f5dd0ff7d528a8c0772b9da06bf60f`.
It composes #1833 `4c5aa3eb6`, #1830 `aae46734b` through the qualified force
integration `c87d698d4`, and #1847 production `c99e524b3`. Existing owners keep
their scientific changes; this is integration qualification, not a new force
algorithm. The current master's 2048-AO resource cap and SCF attempt/work
reporting are retained. No compact-force #1841, incremental #1803, approximate
force-product #1798 or DF substitution is included.

## Decision

Keep the production defaults unchanged. In `benchmarks.readme_pbe0_integrated`,
select SCF admission explicitly with `--scf-active-ao`, separately from the
existing force policy. Its absence explicitly selects dense SCF even when an
ambient environment variable requests local AO. Restore the environment and
shared caller bindings after success or failure. Reject a reference-engine
SCF opt-in before the endpoint. Report the request as
`scf_active_ao_requested`; native SCF diagnostics, not that Boolean, establish
what executed.

The force observers still require the native complete integral route and time,
full grid coverage, local AO work or an explicit budget fallback, and actual
phased Becke batch counts. No missing work is replaced with zero. Indexed force
is still a request without an endpoint-native page counter.

## Minimal qualification

Use one frozen binary and one n1 RTX 5090 for default, force-local/indexed/phased
with dense SCF, and the same force policy with local SCF. First qualify small
molecular/component cases, then retain cold, five warm, moved geometry and five
moved-warm complete endpoints. Compare every native result to every independent
same-geometry GPU4PySCF result with energy below 1e-8 Eh and force below
1e-7 Eh/Bohr. Record actual SCF AO maps and discovery, all reported SCF attempts,
force work, source/library identity and real reference XC backend. Separate
process/JIT cache order from any causal cold-speedup claim.

All GPU work requires finite Slurm allocations on n1, never node3. The n5 build
uses verified ccache and CPU compilation only. Optional allocation/budget
declines retain the existing dense fallback; a sampled AO cutoff is not a
certified error bound. Host tests alone are not chemistry/GPU qualification.

## Evidence boundary and rejected alternatives

The combined focused host suite passes 614 tests, including policy restoration,
independent admission, AO-map fallback, phased Becke math/resources, actual
cooperative host kernels and capacity mutation gates. Compiler structure checks
430 modules and SCF structure checks 206 modules with no dependency errors;
CUDA ownership checks 314 files. All staged repository hooks pass. These checks
do not establish a combined GPU endpoint result.

The older `c87d698d4` complete96 warm medians (75.887 s default, 48.553 s
force-local/indexed, 39.856 s additionally phased) belong only to that source
and campaign. Its combined cold 541.683 s and moved 250.802 s remain negative
results, not removed outliers. #1847 has separate component and complete24
qualification; its complete96 run is still in progress at note creation.

#1841 already tested one-screening compact class queues. It reduced static
stack but retained 255 registers/thread and lost whole-source performance,
including at larger page sizes. Do not import it on the strength of resource
counts. #1830 already exposes point-by-atom and point-by-pair phase dimensions;
do not describe this integration as introducing those for the first time.

Revisit defaults only after same-source complete endpoint gains, numerical
gates, memory/fallback coverage and cold/moved tradeoffs are established.
