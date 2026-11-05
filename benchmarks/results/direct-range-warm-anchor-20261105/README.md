# Full-Direct dense warm-anchor trial

Historical opt-in qualification. The later
[guarded-default decision](../../../.agents/notes/implemented/performance/2026-11-05-default-direct-range-warm-anchor.md)
supersedes the selection policy below, not these measured source/binary identities.

One matched, progress-instrumented cold/three-warm population per policy on
actual A100 and RTX 5090 allocations. This is an opt-in result, not statistical
default promotion. Both policies run full Direct WB97M-V/def2-SVP energy and
analytic forces for water96: 96 atoms, 768 spherical AOs. Cold includes
construction, preparation, SCF, forces and host export; each warm call freezes
only that owner's accepted density. Native imports of PySCF/GPU4PySCF are
prohibited by the endpoint runner.

| Device | Eager cold (s) | Opt-in cold (s) | Cold reduction | Lifecycle reduction |
| --- | ---: | ---: | ---: | ---: |
| A100-PCIE-40GB | 131.754409 | 119.604375 | 9.222% | 6.152% |
| RTX 5090 | 218.958945 | 202.866657 | 7.349% | 5.700% |

The lifecycle is one cold plus three warm calls. A100 warm ratios, opt-in/eager,
are 1.000116, 0.999896 and 1.002389; RTX 5090 ratios are 1.008376, 1.007512 and
1.002172. All twelve native warm observations use one iteration/one Fock.
These observations pass the declared per-call 1.05 timing guard but do not
establish zero regression or a general device ranking. Unset Direct defaults
remain eager. Explicit `GENERATIVEQC_KS_RANGE_LAZY_WARM_ENERGY=1` enables the
existing guarded source reuse for qualified dense-Direct owners.

Both cold policies take 18 iterations; Focks decrease 19 to 18, J applications
19 to 18 and K applications 38 to 36. VV10 work is unchanged. No convergence
threshold, force tile, precision or Hamiltonian changes. The independent fresh
A100 GPU4PySCF reference takes 88.380248/28.426633 seconds cold/warm; the native
cold gap is reduced, not eliminated. Reference effective-potential calls are
not interchangeable with native J/K counters.

## Replay and provenance

Run `python benchmarks/results/direct-range-warm-anchor-20261105/replay.py`.
The CPU-only reader rechecks all paired policy gates, complete work ledgers,
and 32 native/reference observable pairs from `publication.json.xz`. Shared
cohort inputs are stored once; every sample's E/F and work payload is retained.
It does not run GPU or electronic-structure work.

Native source identity:
`a5e1f58ffe3b605a2396e334a5c3206995f4cdb5a3cce71166592ab154d71cca`.
Core ELF SHA256:
`4fafc4ac3f0ea533673f371c2bd4ff116243a022a6cd592a2bb01a3bd0445433`.
The same SM80/SM120 Release bytes run in ECNU LSF 225523 and RTX 5090 Slurm
15954, with scheduler visibility preserved and fast-compile off. Publication
metadata retains archive/trace/log checksums and negative attempts. The proposed
code is based on master `82d672dd51e2e371d3f6167ae03fc29ce501f9f7`, not the
historical repair image previously measured in issue #2976.

See the [decision and failure history](../../../.agents/notes/implemented/performance/2026-11-05-direct-range-warm-anchor.md)
for independent small-system qualification, resource/fallback invariants,
limits and why defaults are not promoted. Engineering memory: #2976; PR #3006.
