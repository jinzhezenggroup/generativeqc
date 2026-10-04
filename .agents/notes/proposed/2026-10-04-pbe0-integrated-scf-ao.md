# Qualification: compose PBE0 SCF and force local AO on current master

Status: implemented qualification harness; complete96 composition qualified
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

## Completed qualification and requested pause

Frozen implementation `d40aeafee` completed n1 RTX 5090 Slurm5757 at
2026-10-04 14:25:59 UTC. Source identity
`60f730f179cef1c1649435f39915f85774f76ecc16dd0b5308d0cd11e42a60c8`, library SHA256
`948c0d6892c8ad3ab05020b4e19b76f221a89d14b1e26ad83af715193015802d`.
Both coarse48 combined/fallback preflights and component/native-KS/memcheck/
synccheck gates pass. Each full96 policy passes72 same-geometry independent
reference pairings, max E/F1.074e-10 Eh /2.478e-11 Eh/Bohr over all three.

| Phase | Default (s) | Force-local/indexed/phased (s) | Plus SCF-local (s) | Reference (s) |
| --- | ---: | ---: | ---: | ---: |
| Cold | 751.783607 | 708.875197 | 479.354171 | 88.744315 |
| Warm, five-repeat median | 77.366631 | 40.953721 | 32.512149 | 17.167006 |
| Moved | 271.620869 | 259.362597 | 137.329585 | 85.542494 |
| Moved-warm, five-repeat median | 77.317810 | 41.108577 | 32.546870 | 10.080260 |

Native cold iterations29/28/28, moved13/14/13, warm all1. Reference warm
iterations4/5/5/4/5 and moved-warm4/1/3/1/1; all XC components use CUDA LibXC.
The composed candidate remains1.89x/3.23x slower for warm/moved-warm. Cold and
moved negatives, missing Fock counts, all slow samples and cache-order caveats
remain in the receipts. Do not replace this reference with Slurm5752's faster
original warm result from another physical GPU.

The first launch5756 failed before GPU gates because the binary SHA receipt was
still being transferred. Retry5757 validates remote binary/head/wrapper transfer
before Slurm launch; source was not changed. Retain `.artifacts/endpoint-5756/`
and its driver failure. Completed raw data are `.artifacts/endpoint-5757/`;
`.artifacts/summarize-integrated.py` produces the compact qualified receipt.
PR1847 retains that receipt as
`benchmarks/results/pbe0-scf-local-ao-20261004/integrated-summary.json` and the
full interpretation under its existing performance note. This evidence branch
does not need a duplicate scientific-optimization PR.

The current optimization is now qualified without enabling defaults or merging.
Per the user's request, publish the closing PR evidence and pause further work.
