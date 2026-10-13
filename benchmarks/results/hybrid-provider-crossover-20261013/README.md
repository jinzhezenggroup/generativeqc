# Installed RTX 5090 hybrid-provider evidence, 2026-10-13

This campaign records complete native 48-atom endpoints and independently
accepted non-water coverage. It does not close #2054: both Direct 48 independent
references failed to converge, both 96-atom Direct force attempts ran out of
memory, both DF-JK 96 attempts were rejected by the frozen 1 GiB workspace limit,
and DF-J/exact-K remains unsupported. No crossover or default-provider change is claimed.

## Frozen scientific and build identity

- Source: `e0ad22036fe83d9bcaed277d0be9a1dbcb453e1c`, clean for this campaign.
- CMake-installed core SHA-256:
  `a8ca3ecabc6834f66b3928d7890cb786c87e1996041e15137c9adc9dbfc3b7dc`.
- CUDA 12.9.86, Release, portable AOT, sm_120, PBE0 RKS stationary AOT assets;
  verified ccache 4.5.1 wraps CXX and CUDA compilation. CPU linear algebra uses
  the configured scalar fallback. Selected CMake values, actual launcher examples, asset hashes and cache
  statistics are retained; the compact build receipt authenticates full local logs.
- One Slurm allocation, job **7968**, node1 RTX 5090 UUID
  `9af5c15a-a02d-fa3e-fe5c-ceb8a0a80c68`, finite 90-minute limit. The runner
  preserved scheduler-assigned device visibility. The final accounting query
  could not recover the completed job because accounting storage is disabled;
  that limitation is retained rather than replaced with a fabricated receipt.
- Unchanged [runner](../../../tools/benchmark_hybrid_provider_crossover.py):
  PBE0, spherical def2-SVP orbital basis, FP64, 48×16×32 quadrature per atom,
  energy tolerance `1e-12`, density tolerance `1e-10`, screening `1e-12`.
- Full DF-JK explicitly uses **def2-SVP as the auxiliary basis**, a 1 GiB
  workspace hint and occupied exchange reuse. It has its own Hamiltonian and
  matching PySCF oracle. DF-J plus exact K is recorded as `UNSUPPORTED`.
- Separate Direct/DF-JK/DF-JK/Direct processes for each case; every supported
  run includes preparation plus cold E+force, unchanged warm, changed geometry
  and moved-warm E+force, including host return. Diagnostic traces run afterward
  and their times do not enter the clean endpoint populations.

The earlier job 7966 was a smoke probe, excluded from this accepted population:
its source receipt marked an untracked `build-cuda/` output as dirty. Only that
build-output directory was added to the task repository's local Git exclude;
no tracked source or installed binary changed. Job 7968 reran the complete
population with clean source receipts. Earlier failed and smoke files remain
in ignored task storage.

## Recorded endpoints

Values below are medians of the two independent ABBA processes, in seconds.
They compare the explicitly labeled approximations, not equal-accuracy methods.

| Case / provider | NAO / Naux / Nocc | Cold E+F | Warm E+F | Changed E+F | Moved-warm E+F |
| --- | --- | ---: | ---: | ---: | ---: |
| 48-atom water / exact Direct | 384 / — / 80 | 27.040 | 6.764 | 47.242 | 6.900 |
| 48-atom water / full DF-JK | 384 / 384 / 80 | 17.555 | 4.229 | 10.895 | 4.330 |
| Formaldehyde / exact Direct | 38 / — / 8 | 3.638 | 0.293 | 1.518 | 0.294 |
| Formaldehyde / full DF-JK | 38 / 38 / 8 | 2.674 | 0.153 | 1.006 | 0.154 |
| 96-atom water / exact Direct | 768 / — / 160 | force OOM twice | unavailable | unavailable | unavailable |
| 96-atom water / full DF-JK | 768 / 768 / 160 | preparation rejected twice | unavailable | unavailable | unavailable |

The auxiliary choice has a substantial scientific approximation effect: the
first cold DF-JK energy differs from exact Direct by **−10.9430976359 Eh** for
48-atom water and **−1.2335227581 Eh** for formaldehyde. Agreement with each
arm's matching oracle does not establish DF agreement with exact Direct. These
timings must not be used to justify an automatic approximation switch.

Each completed native receipt received a separate independent PySCF 2.14.0
attempt on the exact exported quadrature. Six references completed both original
and moved geometries: both DF-JK 48 references and all four formaldehyde
references. They pass the unchanged energy `1e-8` Eh, force maximum `3e-7`
Eh/Bohr and physical-residual `1e-8` gates.

Both Direct 48 references returned **`converged: false`** for the original
geometry under the frozen 180-cycle, energy `1e-13` and gradient `1e-10` oracle
settings, after 1,207.7 and 1,186.2 seconds respectively. Once that mandatory
gate failed, their later phases were terminated. Their raw oracle JSON remains
unchanged with its partial `RUNNING` status and failed original sample; separate
termination receipts bind the exact raw hash, reason, signal and exit status.
They are not accepted references, even if individual energy/force differences
look small. No tolerance was relaxed and no convergence flag was overwritten.

The offline verifier recomputes both formaldehyde pairs as `PILOT_ACCEPTED`,
both 48-atom pairs as `INCOMPLETE` because the Direct references are invalid,
and both 96-atom pairs as `INCOMPLETE` because their native endpoints failed.
None authorizes a crossover claim. Raw values and independent errors remain
in the retained records and summaries.

## Work, memory and remaining limits

Native receipts retain actual Fock-build counts, iterations, transport counts,
NAO/Naux/Nocc and physical-state diagnostics for every completed phase. Separate
traces retain DF metric/three-center preparation, RI-J/occupied RI-K dimensions,
executed contraction counts and force-response reuse. Both successful DF
profiles prove reuse of the final fitted occupied projection in force response.

For 48-atom DF-JK, metric diagnostics report 774,372,577 device-resident bytes and
a planned 781,978,493-byte peak. These are allocation-plan records, not a sampled
or instrumented whole-process peak. Direct traces expose compact force and
transfer counts, but do not provide complete admitted/rejected quartet-work
accounting. Absent counters and `resource_diagnostics: null` remain absent;
Fock-build counts are not substituted for quartet counts.

The 96-atom Direct cold SCFs converged, but no complete force output exists.
DF-JK failed before a cold solve because the metric and one contraction tile
could not fit the explicit budget. Their separate diagnostic attempts also
failed. A future larger-memory study must state its budget before measurement,
retain these losing records, and rerun matched populations. Node2's PRO 6000
cannot directly load this core: the core requires GLIBC 2.38 while node2
provides 2.35, so reusing this installed binary there is not an available shortcut.

Before closing #2054, obtain converged independent Direct 48 references, finish
the 96-atom supported endpoint/reference population, complete memory/work
decomposition, and review the approximation contract and
all source-matched comparisons. No performance or default claim is inferred
from missing results.

## Reproduction and integrity

The finite [campaign script](run-campaign.sh) uses the same installed prefix as
the [build script](build-cuda.sh). After a fresh build, verify tracked source with
`git diff --exit-code` and exclude only the generated `/build-cuda/` directory
through the task repository's `.git/info/exclude` before the clean-source check. The output root was
`/data/jzzeng/qc-closeout-20261013/.artifacts/closeout-campaign-7968`.
Grid exports and full build output remain in ignored task storage; small raw
JSON, diagnostic traces and build/installation identity records are retained
as deterministic gzip members with exact original byte counts and SHA-256.
The manifest also authenticates the retained scheduler-query failure.

```bash
PYTHONPATH=python:. python benchmarks/results/hybrid-provider-crossover-20261013/verify_evidence.py
```

The verifier authenticates each member, binds each oracle to its original native
receipt, recomputes all six pair summaries without a GPU, and rejects any claim
that the incomplete water populations established a crossover.
