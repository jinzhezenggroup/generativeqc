# Bounded default-auto Global-Hybrid phase-source reuse

Actual unset-default PBE0/def2-SVP water-96 complete E+F qualification.
Every ABBA sample, force vector, precision event sequence and gate is retained.
This is not an atom-count rule or universal/kernel profitability claim.
Earlier opt-in evidence remains local and in historical notes, not pooled here.

## Measured source and scientific contract

- Base master: `637466c624c4c4c4dd4eb7c511a2c88675744325`.
- Clean measured revision: `1cef1189224c6ddcb29d684f9ef7eaa0aa0e34f8`.
- Production source identity:
  `d920301e312f258b4b941fb331b90ca9b2b22e561acac2320c56c71c38f0d4db`.
- Linked library SHA256:
  `1824909e08553210409b1a5078f34c58e038ca6414c23c6cca2188f52ef58ec9`.
- Original independent reference SHA256 before compact selection:
  `84c567be1a6e4702e41d3530f8fea31cb121c127389982b4438d0899dc469ccf`.
- Offline basis: `benchmarks/results/pbe0-def2-svp-20261003/def2-svp-ho.json`,
  SHA256 `fad2c7433fe5fac19525d4c80fbd995a7ebed7816b7ea4bb88547e4f6a58a3d8`.
- 32 waters / 96 atoms, 768 spherical / 800 Cartesian AOs, Bohr coordinates;
  grid 48/16/32, 2,359,296 points; screening/energy/density tolerances
  1e-12 / 1e-12 / 1e-10, 100 permitted iterations, one repeated fixed-density
  reference per geometry. Both selectors retain strict FP64.
- Energy/maximum-force gates: 1e-8 Eh / 1e-7 Eh/Bohr. Oracles provide assertions
  only; no oracle density, J/K, ERI or force enters the production solver.

Release sm120, portable CUDA AOT and `GENERATIVEQC_CUDA_FAST_COMPILE=OFF` are
pinned in the retained build settings. Compilation cost is not a timing claim.

## Routing boundaries

Admission/lifetime follows `docs/developer/scf_module_boundaries.md`:
bounded unset/auto, explicit `0` rollback and exact resource fallback.

Every admitted cold phase builds 1,873,221,716 values / 14,985,773,728 bytes.
One counted incumbent fixed-D handoff rebases only warm energy metadata with
retained XC, without modifying the returned state or convergence gates.

## Complete endpoint observations

Jobs 7338 and 7340 run two ABBA cohorts on separate RTX 5090 UUIDs, with four
fresh processes each: `0 → unset → unset → 0`. Each process completes cold
preparation+E+F, warm E+F, moved E+F and moved-warm E+F.

| Median complete endpoint | Disabled | Unset default | Disabled/default |
| --- | ---: | ---: | ---: |
| Cold including preparation and handoff | 97.8693 s | 76.2461 s | 1.2836 |
| Warm | 16.3450 s | 15.0689 s | 1.0847 |
| Moved | 62.8232 s | 62.7861 s | 1.0006 |
| Moved-warm | 14.9794 s | 14.9855 s | 0.9996 |

**Do not attribute the entire cold ratio to faster ERI replay.** Baseline cold
physical Fock counts are 17, 19, 21 and 22; every default cold has 17 resident
SCF actions plus one charged incumbent handoff (18 total). The SCF trajectories
are not work-matched. Every warm/moved/moved-warm arm has 1/12/1 physical Focks;
those phases construct no source. The warm timing difference is not evidence
of warm source reuse. Two cohorts are descriptive bounded endpoint evidence,
not a broad hardware/crossover qualification or formal performance promotion.

All 32 large and 16 genuine three-atom E+F oracle gates pass. Maximum large
energy error is 9.5497e-12 Eh; maximum force error is 3.8277e-11 Eh/Bohr.
Default warm solves pass unchanged gates with one fresh Fock; audits and
source/handoff counts are explicitly checked.

Job 7345 passes four signed raw-J/K and same-grid PBE0/B3LYP energy probes
(water f / 24-atom SPD, chunks 1/2), including default/seed/budget/frozen-cache
cases, plus three independent RHF original/displaced tests. Job 7347's f-shell
memcheck reports zero errors. It copies its input into the new job directory
before writing snapshots; historical snapshots and mapped binaries stay intact.

Host suites pass 994/336 cases, with 89 skips and four known offline GFN1-D3
cases deselected. Fixture failures 7336/7337 (missing LibXC assets) and 7341
(missing SONAME alias) remain in `fixture-setup-failures.json.gz`, excluded
from acceptance. Fixing them does not change production source/binary bytes.

## Reproduce and inspect

Restore the measured commit under `<fixture-root>/source`, including tracked
`upstream/` assets and benchmark/build dependencies. Run
`bash build.sh <fixture-root>` with verified ccache and CUDA 12.9; keep fresh
binary identities. Adjust fixture/dependency/Python paths in `run-default.sh`,
then run two finite Slurm ABBA allocations preserving assigned visibility:

   ```bash
   srun --partition=main --nodelist=node1 --gres=gpu:5090:1 \
     --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=48G --time=00:30:00 \
     bash <fixture-root>/run-default.sh abba
   ```

Use n1/n2/n4/n5, never n3; allocate `probes`/`memcheck` for 20 minutes. Preserve
visibility and active fixtures. Require zero exits, completion markers and
passing observer gates.

`publication.json` verifies selected member identities; `validation.json.gz`
holds quantitative acceptance. `samples.json.gz` keeps every timing, full force
vector, gate, work count, strict operator inventory and Fock event sequence.
`reference.json.gz` keeps the exact cold/moved oracles used by the comparisons.
`receipts.json.gz` and `qualification.json.gz` retain device/source/binary/build
identities, native/oracle outputs, test logs and sanitizer results. Full raw
traces, dense probe inputs/snapshots, unused oracle repeats and historical
campaigns stay in ignored local artifacts. Regenerate inputs with the committed
tests; compact selection respects repository retention budgets.
