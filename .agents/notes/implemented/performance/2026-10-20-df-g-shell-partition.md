# Decision: retain shell reuse around an auxiliary-g remainder

Status: implemented, qualified on the recorded endpoint domain
Date: 2026-10-20

## Problem and evidence

The default GPU4PySCF auxiliary contains oxygen g shells. PR #2647 admitted
their complete native values and derivatives, but routed every three-center
derivative through the generic component consumer when any g shell was present.
Same-binary n1 PBE-DF/48 attribution at `02337e0cf` found 19.083 s warm, of which
16.602 s was three-center response and only 0.237 s was SCF. Substituting the
historical cc-pVDZ-JKFIT auxiliary for diagnosis gives 2.571 s warm and 0.134 s
three-center response. The old fit has more auxiliary AOs (1856 versus 1808),
but no g class and therefore retains shell reuse. This substitution is not a
README configuration or a proposed scientific default.

## Decision and invariants

Partition three-center response by auxiliary angular momentum. Existing shell
consumers own s/p/d/f; the unchanged generated generic derivative owns g. Public
AO offsets and dense response weights remain intact. Both metadata views carry
their actual expansion stride: six-term packing cannot be interpreted as three
merely because a particular shell has lower angular momentum.

The generic g remainder excludes all lower auxiliary classes, while metric
response always includes every auxiliary pair. The source already bounds orbital
angular momentum by f and auxiliary by g. Packed requests keep the dense weight
fallback required by the generic remainder. Partial s/p shell selection retains
its original generic complement. Mixed-gradient qualification is not broadened.
Resource admission, source identities, stream lifetime, independent physical
force gates and the explicit generic fallback remain mandatory.

## Qualification

Source `33461b49e` passes all 34 GPU tests in Slurm job 11404: independent
complete E/F and translation gates for full, partial, packet and generic routes,
plus existing default-auxiliary HF/KS Standard/Fast and open-shell endpoints.
The first attempt passed 19 tests before finding a missing UKS AOT package;
the separately retained rerun adds that package without changing the core hash.

All four samples of PBE/PBE0 DF48/96 and r2SCAN-3c DF48 pass the unchanged
energy/atom 1e-5, maximum force 1e-4 and RMS force 3e-5 gates against independent
stable direct references. Grid stability and CPU/GPU spots use one quarter of
those gates. Each candidate also agrees with the earlier native default-fit
population. Oracle and native source identities remain distinct and unchanged.

| Method / atoms | Previous cold / warm median (s) | Candidate (s) |
| --- | ---: | ---: |
| pbe / 48 | 30.577 / 19.133 | 17.129 / 5.555 |
| pbe / 96 | 167.194 / 129.270 | 68.180 / 29.708 |
| pbe0 / 48 | 50.469 / 38.167 | 23.380 / 10.897 |
| pbe0 / 96 | 304.197 / 255.253 | 104.991 / 56.309 |
| r2scan-3c / 48 | 101.397 / 70.632 | 57.032 / 26.899 |

Exact timings and numerical errors are retained in the evidence bundle. These
separate complete-endpoint populations are diagnostic comparisons, not an
interleaved statistical comparison or proof of superiority to GPU4PySCF.
The separately traced PBE48 warm three-center response decreases from 16.602 s
to 3.140 s, with positive shell and auxiliary-g remainder work. The generic
g remainder can still dominate large cases; full g shell sharing is a distinct
future optimization. r2SCAN-3c DF96 and WB97M-V DF96 resource/work limits remain
unresolved and are outside the completed endpoint qualification above.

The clean r2SCAN-3c DF96 attempt hits its 45-minute Slurm limit without a
completed sample. A separate 180-second trace shows the first RI-J evaluation
alone taking 135.361 s, versus 1.797 s for metric factorization and 6.558 s for
preparation. It repeatedly regenerates raw three-center tiles under the value
budget. That value-path work cliff requires a separate optimization; speeding
the force consumer cannot fix it. The untouched partial measurement, timeout
receipt and compact trace counters are retained in
`benchmarks/results/df-g-shell-20261021/large-case-diagnostic.json`.
