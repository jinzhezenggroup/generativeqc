# Stationary-density energy confirmation: repaired endpoint qualification

Date: 2026-10-16. This is a new source-bound qualification, not a relabeling of
the [original failed cohort](../r2scan3c-final-validation-20261015/README.md).
The historical `none/96/moved` complete-endpoint regression remains rejected.

## Scope and acceptance

The repaired ordinary strict-FP64, all-electron, restricted exact-direct pure-J
path holds an already stationary density for another complete physical F[D]
when only the original energy-change gate is missing. It does not replay an
energy, accept an orbital proposal, relax a residual tolerance, or declare
convergence. Both actual residual norms must already pass. All existing stopping
gates, iteration limits, cache-publication guards and bounded resident fallbacks
remain in force. See the
[decision note](../../../.agents/notes/implemented/numerics/2026-10-16-stationary-pure-j-energy-confirmation.md).

Each comparison runs selector `GENERATIVEQC_CUDA_KS_DEVICE_FINAL_VALIDATION=0`
against a **genuinely unset** selector on the same repaired library. The force
selector `GENERATIVEQC_DIRECT_SPD_FORCE_DOMAINS` is absent in both arms. All
complete cold, warm, moved and moved-warm endpoints must pass the existing
nonregression gate, even if SCF work changes; warm work must also match.
The existing 2%/MAD benefit label is descriptive, **not** an enablement threshold.
Positive Direct/48 benefits below 2% are not grounds to disable that default.

The scientific protocol and independent GPU4PySCF references are unchanged:
r2SCAN-3c, canonical spherical def2-mTZVPP, original grids/screening and D4/gCP,
five warm repetitions per geometry, and independent gates of **1e-8 Eh** in
energy and **1e-7 Eh/Bohr** in maximum-component force error. The strict SCF
energy tolerance remains **1e-12 Eh**. The repair introduces no CPU/PySCF or
reference-oracle calculation into the production endpoint.

## Complete endpoints

Seconds include the complete energy-and-force endpoint; warm rows are medians
of five calls. Benefit is `(control - default) / control`. Work is
`SCF iterations / Fock builds` per call and is equal in both arms in every row.

| Direct atoms | Phase | Selector 0 (s) | Unset default (s) | Benefit | Work, both arms |
|---:|---|---:|---:|---:|---:|
| 48 | cold | 76.255838 | 75.996318 | +0.340% | 19 / 19 |
| 48 | warm | 21.795101 | 21.428271 | +1.683% | 1 / 1, each of 5 |
| 48 | moved | 57.352647 | 57.057831 | +0.514% | 13 / 13 |
| 48 | moved-warm | 21.745778 | 21.438734 | +1.412% | 1 / 1, each of 5 |
| 96 | cold | 399.700329 | 396.484187 | +0.805% | 24 / 24 |
| 96 | warm | 90.681922 | 86.932478 | +4.135% | 1 / 1, each of 5 |
| 96 | moved | 267.157748 | 263.365821 | +1.419% | 14 / 14 |
| 96 | moved-warm | 90.759765 | 87.184101 | +3.940% | 1 / 1, each of 5 |

All eight complete-phase gates pass. All 48 comparison calls pass independent
scientific gates; their largest errors are **4.047252e-11 Eh** and
**1.627374e-10 Eh/Bohr**. A further 24 candidate accuracy calls at 3/12 atoms
pass, with maxima **2.899015e-12 Eh** and **2.505197e-11 Eh/Bohr**: **72 complete
scientific calls** in total. Small-system calls do not expand resident-default
admission below 544 AOs or make an unpaired speed claim.

In both new Direct/96 moved histories, iterations 13 and 14 have identical
one-electron traces within each arm. The freshly computed compensated energy
changes on iteration 14 are 5.676562e-14 Eh (control) and 2.011674e-13 Eh
(default), passing the unchanged energy gate. These histories are not a claimed
per-iteration record of the true maximum residual; the executable predicate
tests cover the actual maximum-norm gate independently.

Intrusive cold/first-warm resident-route diagnostics at both sizes execute
valid `resident_final_state_validation` proofs with no CUDA error. They are
separate runs and **excluded** from every timing population in the table.

## Image and execution provenance

- Source base: `fb37f6d1c66cdb53ed83b3b68a284024b6060bab`, plus `source.patch.xz`.
- Native source identity:
  `c300965dbc535596b2310a23eecb741eabb51dc1dd158f468292c9147f554867`.
- Core SHA256:
  `21526c6c1a1ae5258282a92f942c37ddf9536e1094b81a85e4e7697942460e2f`.
- Native input inventory: 1744 files, retained as a 46-entry delta against the
  pinned historical inventory. Upstream inputs: 610 unchanged files, identified
  by the base Git tree and canonical inventory digest.
- Both `r2scan_rks` and `r2scan_rks_spd` force AOT contract identities and
  binary SHA256 values are retained in the qualified job logs inside receipts.
- Slurm jobs: **8847 / node1** (48 atoms, 3/12-atom accuracy and 48-atom route
  diagnostic), **1635 / node5** (96 atoms and its route diagnostic). Both use
  RTX 5090, partition `main`, `--gres=gpu:5090:1`, and preserve Slurm visibility.
  Runs use finite `srun` allocations; the retained scheduler record confirms
  the 96-atom allocation's 01:30:00 limit.
- CUDA 12.9.86, GCC 11.4, verified `ccache 4.5.1`; build logs and before/after
  cache statistics are retained. Cache contents were not cleared. GPU predicate
  job **1626** passes all 30 cases, including 22 device executions. New-image
  resident proof/library-lease fixtures also pass before endpoint measurement.

The 46 inventory differences include changes already in the base, not only this
repair. **Old-to-new absolute timings are not a controlled patch-only gain.**
Qualification comes from the new same-image selector-0/unset comparisons.
Earlier failed setup attempts (missing `upstream/libxc/7.0.0`, stale force AOT
contracts, and an old-core/repetition-mismatch pilot) remain in the receipts as
failures, not qualification. Force AOT sources, binaries and manifests were
regenerated normally, not repaired by editing identity fields. Final execution
checks all frozen native/upstream inputs, recomputes the full native inventory,
checks the loaded core identity, and uses the actual force AOT loader to verify
both contracts and binary hashes before starting SCF.

## Retained artifacts and reproduction

- `qualification.json.xz`: observations and receipts. Observations contain full
  per-call diagnostics, work/timing populations and comparison summaries as
  lossless dictionary deltas against the pinned historical comparisons.
  Receipts retain actual comparison force arrays, small-system gates, separate
  resident proofs, build/cache/test logs, launch/freeze/retention scripts and
  failed-attempt receipts. Force arrays are lossless little-endian binary64 XOR
  deltas against the previous call at that geometry (the independent reference
  for the first call), hex-encoded before compression. No force is rounded;
  the verifier decodes every value before recomputing independent errors.
- `source.patch.xz`: qualified native/test/checker patch against the named base;
  later documentation and this evidence packet are not native compiler inputs.
- `manifest.json`: byte lengths and SHA256 digests of the compressed artifacts.
- `verify.py`: **opt-in evidence-integrity check**, not a permanent CI fingerprint
  test of production implementation spelling.

From a repository checkout containing the pinned base commit and historical
packet:

```bash
PYTHONPATH=.:python python \
  benchmarks/results/r2scan3c-stationary-confirmation-20261016/verify.py
```

The verifier reconstructs source/upstream inventories, checks artifact hashes,
scientific protocol identity, actual selector absence, complete populations,
independent energies and force arrays, all existing nonregression/work gates,
small-system gates and resident proofs. It also requires that the original
Direct/96 moved comparison **still fails**. It neither modifies nor regenerates
the historical archive or its plots. Native identity checks here describe this
immutable qualification image, not a requirement that future refactors preserve
its bytes.

For a new qualification, rebuild the base plus patch, regenerate force AOT,
and adapt the retained `run.sh` deployment paths. Use finite `srun`, unchanged
references and five repeats; preserve Slurm visibility and avoid node n3.
Collect both arms and separate diagnostics before reduction/retention. Preserve
all new failures without overwriting either historical cohort.
