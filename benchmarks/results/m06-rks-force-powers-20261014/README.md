# FP64 M06-2X RKS force integer powers

Campaign: October 14–15, 2026. Control: master `8c770b34c` (including #2486).
Scientific candidate: `8021e2d92`. This does not depend on or re-credit #2464's
unmerged equal-spin branch CSE.

Lower powers up to four **after independent polarized AD**, only in FP64
M06-2X RKS stationary forces. SCF point source, UKS, float scalar graphs, MN15
and r2SCAN defaults remain unchanged. Strict FP64 force fallbacks use the
qualified graph. Generated-source/artifact identities deliberately change.

## Complete endpoint timing

RTX 5090/n5, finite Slurm jobs 1576/1578, CUDA 12.9.1, sm_120, Release, FP64,
spherical def2-SVP, matched moving quadrature and DF model. Measure native energy
**plus host-returned analytic forces**, with fixed own-engine seeds, no foreign
density. Each population has cold + five warm + moved + five moved-warm calls.
Use `complete_seconds`, including cold preparation; warm entries are medians
of **all five** calls. Cold/moved are single samples, not statistical assurances.

| First population | Cold, baseline → candidate (s) | Warm (s) | Moved (s) | Moved-warm (s) |
| --- | --- | --- | --- | --- |
| DF48 | 45.952664 → 45.384089 | 8.899711 → 8.077829 | 34.786684 → 34.070293 | 8.891267 → 8.076551 |
| DF96 | 152.750312 → 151.213377 | 37.364340 → 35.669518 | 112.652525 → 110.930658 | 37.354787 → 35.647791 |
| Direct96 | 163.911763 → 163.142795 | 20.089646 → 19.214328 | 113.078680 → 112.308974 | 20.064995 → 19.228165 |

Complete warm E+F reductions: **9.23%, 4.54%, 4.36%**; moved-warm: **9.16%,
4.57%, 4.17%**. First-population iterations/Fock builds match in both arms:
DF48 18/1/13/1, DF96 18/3/12/3, direct96 18/1/12/1, in phase order. This is
neither a point-only speed claim nor a new GPU4PySCF comparison.

| Second direct96 population | Baseline → candidate (s) | Builds |
| --- | --- | --- |
| Cold | 171.732486 → 163.300249 | 19 → 18 |
| Warm | 20.093679 → 19.228675 | 1 → 1, all five |
| Moved | 120.919491 → 112.209232 | 13 → 12 |
| Moved-warm | 20.089118 → 19.224459 | 1 → 1, all five |

Warm/moved-warm reproduce **4.30%/4.30%** fixed-work gains. The unchanged
baseline image itself varies from 18/12 to 19/13 cold/moved builds. Do **not**
credit the second population's larger cold/moved changes as a solver improvement,
subtract SCF work or normalize it away. Unknown integral counters remain null.

## Independent acceptance

All **96 master calls** pass unchanged **E 1e-8 Eh / F 1e-7 Eh/Bohr** gates
against independent references at both geometries. Maxima: 2.0464e-11 Eh /
1.5725e-10 Eh/Bohr. Host suite: 124 passed/43 explicitly optional skipped;
final policy suite: 17 passed. Production force tests independently check Libxc
Cartesian partials, displaced energies, tau/2 ABI and polarized derivatives.

Clean-source GPU scalar acceptance n2/3021 explicitly probes the exact production
force body, passing Libxc 7, finite-difference, restricted-embedding and
authenticated 113-bit Maple boundary gates. It does not patch production SCF.
Memcheck n2/3020 has zero errors; its earlier source attribution is retained,
with byte-identical final force scalar body. Both endpoint arms use the same
authenticated #2486 glibc loader/runtime, not a production CPU fallback.

Keep the original compiler prerequisite failure, two PATH failures, stricter
M06 tail failures and historical direct moved regression. Extra same-image
controls contain all 48 calls; they do not erase the original 108-call negative
experiment. No #2486 AO/tile/dddd/page knob is retried. Details and numerical
limits: `.agents/notes/implemented/performance/2026-10-15-m06-rks-force-powers.md`.

## Retention and reproduction

`manifest.json` binds the current `summary.json.xz` **endpoint ledger** by bytes
and SHA256. It keeps all 96 master and 48 historical-control timing samples,
phase ranges, E/error/gate/convergence values, semantic work, model/source/library
identities, diagnostic reports and failures without rounding or filtering.
Full E/F vectors, independent references, point arrays, intermediate SCF history,
component profiles, build/cache/scheduler logs and scripts are **not** in this
compact ledger; they remain byte-exact in the original Git capsules.

`full-evidence.json` pins six original files at ancestor
`3ca3fa0038fbea5350a8bb799cca26d678522678`, with individual SHA256/lengths and
snapshot totals. Restore both full capsules and their original guides/manifests
offline, with no Release, upload or automatic fetch:

```bash
python tools/restore_retained_evidence.py --all \
  --manifest benchmarks/results/m06-rks-force-powers-20261014/full-evidence.json \
  --output .artifacts/m06-force-full-evidence
```

Use a new output directory; the reader verifies every member before publishing
and refuses overwrite. Clones must contain that ancestor; if absent, explicitly
run `git fetch origin refs/pull/2505/head`, the recorded PR ref that contains it.
Do not rewrite/drop the bound history. `python tools/compact_m06_force_evidence.py` regenerates both ledgers
from these authenticated originals. Recovery does not rerun GPU work.

For reproduction, restore first and extract `reproduction` from the full current
capsule. Use its `build_master.sh`, `repeat_endpoint.sh`, `python.sh` and
`validate_force_schedule.py` at the pinned scientific revisions. Keep caches,
run every GPU command through finite compatible Slurm, preserve device visibility
and retain all planned phases/failures. Do not rewrite master README historical
timings using differently sourced populations.
