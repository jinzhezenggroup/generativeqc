# Experiment: exact resident VV10 SCF pair reuse

Status: default-off prototype, awaiting GPU and endpoint qualification
Date: 2026-10-14

The separately qualified force reuse only removes the geometry pair sweep.
Frozen SCF density replays still rebuild the quadratic VV10 feature component.
`GENERATIVEQC_VV10_SCF_COMPONENT_REUSE=1` enables an independent prepared-owner
experiment. Its immutable device/grid/parameters and bitwise ingredient key
reuse the same generated scales and failure-aware key comparison as force
reuse. A pair-schedule change invalidates the key. There is no tolerance,
approximate density, pointer-equality shortcut or final-endpoint memoization.

Semilocal XC, all J/K operators, AO potential assembly, diagonalization, physical
residual and convergence checks still execute. The scalar nonlocal energy is
retained outside pair scratch; seven ingredient panels and prior-error status
reuse dead scratch. Forty optional bytes cover that scalar, the predicate and
three device work counters. This lease is charged before allocation and must
fit the existing prepared nonlocal bound. Tiny grids and unsupported variants
retain the existing unconditional route.

Counters are copied with the existing completed-iteration scalar packet and
share its synchronization; their 24 bytes are charged to transfer diagnostics.
The private batch receipt reports provisional/target Focks and actual pair
probes, hits and sweeps per solve. Missing older APIs and unsupported paths
remain unavailable. The bounded device-chunk path may reuse exact components
but does not claim these host-loop work counters. Every reused target Fock
still counts as a fully evaluated physical Fock of the current density.

Required gates include exact frozen replay, geometry rebuild, changed density,
schedule change, failed-generation recovery, independent complete molecular
E/F, memory/synchronization sanitizers and source-bound 12-call populations.
This experiment is separate from the provisional native density seed, whose
first 12-atom diagnostic still required about 16 expensive target sweeps and
did not establish a useful large-system cold improvement. Neither experiment
is enabled by default or closes #2426.
