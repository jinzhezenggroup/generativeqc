# README default-configuration endpoints (2026-10-20)

[Direct figure](../../../assets/readme-direct-defaults.svg) ·
[DF figure](../../../assets/readme-df-defaults.svg) · [All timings](timings.md)

## Results

The campaign attempted **256 populations**: 8 methods × 4 water-cluster sizes ×
Direct/DF × 4 engine profiles. It completed **765 energy-plus-analytic-force
calls**. CBS and historical measurements were not rerun or relabelled.

| Profile | Direct | DF |
|---|---|---|
| GenerativeQC Standard | 32 complete | 32 unsupported |
| GenerativeQC Fast | 32 complete | 32 unsupported |
| GPU4PySCF Default | 32 complete | 32 complete |
| GPU4PySCF Fast | 31 complete; HF/96 cold only | 32 complete |

**Native DF cannot use the stock reference auxiliary basis on this master.**
GPU4PySCF automatically chooses def2-SVP-JKFIT for def2-SVP; canonical
r²SCAN-3c's explicit orbital tables select an even-tempered auxiliary. Both
contain oxygen g shells (`l=4`). Native CUDA `df_metric` supports only `l≤3`
and rejects preparation. All 64 negative records remain in the bundle. Neither
engine was switched to the historical cc-pVDZ-JKFIT basis to bypass this limit.

GPU4PySCF Fast HF/Direct/96 completed cold but did not converge the first warm
call within its unchanged 50-cycle limit. Its cold point remains; no warm
median is invented. Native Fast uses its real public controller: **120 calls
ran one solve, and 8 ran two solves**, including FP64 audits. Its statuses are
112 `qualified_budget_fock`, 8 `qualified_mixed_gradient`, and 8 `observed_met`.
In particular, HF/96 Fast warm calls include the audit cost and are slower than
Standard; no sample or fallback was selected away.

These figures show **absolute default-configuration latency**. No common
independent grid-converged accuracy qualification was run, no equal-force-work
speedup is claimed, and no fastest-qualified production configuration is
selected. The publication decision remains **inconclusive** for that separate
performance qualification claim.

## Protocol

- Methods: HF, PBE, PBE0, B3LYP, r²SCAN, r²SCAN-3c, M06-2X, and ωB97M-V.
  Water clusters contain 3/12/48/96 atoms, using the retained
  `readme_hf_scaling.scaling_cases` geometries. Spherical def2-SVP is used except
  canonical r²SCAN-3c/def2-mTZVPP. Charge and spin are zero.
- Every population has one fresh-owner **cold** call and three **warm** calls.
  Cold includes construction, preparation, SCF, complete forces and host export.
  Imports and CUDA context initialization precede timing. Runtime compilation,
  if entered during an endpoint, is included. Existing compiler/artifact caches
  are retained; cold is not a cache-purged installation measurement.
- Warm calls reuse the same owner's **frozen cold density**, independently for
  each engine. Fast auditors never seed native candidates. Reference warm
  density copies are timed. No moved-geometry calls were made.
- GPU4PySCF Default has no numerical/grid/force-response overrides. Native
  Standard uses ordinary `Calculator`; Fast uses
  `PrecisionCalculator(policy="fast")`, with no numerical overrides. Native
  implicit XC/VV10 grids and reference defaults use level-3
  Treutler/Ahlrichs radial, Lebedev angular, NWChem pruning and Treutler-adjusted
  Becke partitioning. Native omits reference alignment padding; density-screened
  domains can also differ. For water-96 ωB97M-V, native reports 1,078,336 XC
  points; GPU4PySCF reports 1,078,528 XC and 1,078,528 VV10 points.
- GPU4PySCF Fast copies the actual native Fast candidate's energy and density
  thresholds to `conv_tol` and `conv_tol_grad`. **Density RMS and orbital-gradient
  norm are different residual definitions.** Direct DFT candidates use
  `1e-6`/`1e-4`; HF/DF candidates retain `1e-10`/`1e-8`. The other reference
  controls remain stock. These thresholds are not application force-error gates.
- GPU4PySCF's stock force `grid_response=False` is preserved. Native includes
  moving-grid response. Consequently, the returned force definitions differ.
  r²SCAN-3c includes canonical native D4/gCP; the reference uses GPU4PySCF r²SCAN
  plus independent CPU D4/gCP, entirely inside the timed endpoint.
- Native DF receives an offline snapshot of the reference's automatically
  selected auxiliary tables. Reference DF independently chooses its defaults
  and checks the executed tables against that snapshot after timing. No PySCF
  calculation or reference density enters a native timed endpoint.
- Reference effective-potential build counts and native SCF/Fock/precision
  counts are retained separately; they are not asserted to measure identical
  work. Native VV10 receipts include target/provisional Focks, pair sweeps and
  cache probes, not an invented exact interacting-pair count.

## Provenance

Production source is master **`e5fa4d2f0cd13775327261a0b6ced1eab6006c62`**, including
#2626/#2633. The measured driver is from `39e2ebc05`, adapting #2623's default
comparison approach; #2623 was not merged into the production baseline.
`measurement-driver.patch` reconstructs that sole measurement-source addition
on the frozen master. The receipts bind its full SHA-256 independently.

All GPU work used finite Slurm allocations on n1 / `node1`, partition `main`,
with one RTX 5090 per job, two CPUs and 48 GiB host reservation. Each launcher kept up to eight jobs submitted; up to seven GPU jobs ran
concurrently while another user's job occupied one card. CUDA visibility was preserved. Individual deadlines
were 4 hours for 96 atoms and 90 minutes otherwise; no job exhausted its limit.
The untimed HF/3 native and PBE/3 reference smoke tests are excluded.

GPU4PySCF 1.8.1, PySCF 2.14.0, CuPy 13.6.0 and NumPy 2.4.6 were used on n1.
The source-matched Release/sm_120 build used CUDA 12.9.1, verified sccache 0.16.0,
normal CUDA optimization, native stationary forces and ten RKS force AOT
plugins. The loaded native library SHA-256 is
`b08d0089948e5595a08c707e36696df23dada58cfa45f832d947dc8d3811709b`.
[Build settings](build-settings.json) retain all plugin hashes and CMake
settings; [cache receipts](compiler-cache.json) retain before/after statistics.
These cache statistics are shared-server observations, not exclusive campaign
hit counts. Clock/temperature metadata was collected at process startup; the
shared helper's legacy `sampling_point` string must not be read as an end-time
clock observation.

`receipts.json.xz` preserves all 256 receipts, including every completed energy,
force vector, setting, work counter, precision report and failure.
`job-outcomes.json.xz` preserves every scheduler command, process exit code and
raw log/receipt hash. Large build logs and compiler listings remain in ignored
storage on n1 at `/data/jzzeng/readme-defaults-20261020`; their identities are
recorded in [provenance](provenance.json). No external archive or Release is
required for replay. Historical source inventories were losslessly consolidated
to fit the unchanged repository evidence budget; their original bytes and Git
identities remain recoverable.

## Reproduction

Build the frozen production source with the settings above. The retained
`build-n1.sh` and `build-aot-n1.py` record the exact local build commands;
`run-n1.sh` and `launch-n1.py` record the finite Slurm campaign launcher.
Paths in these receipts are the actual n1 deployment paths, to be adapted for a
new run. The portable per-population entry point is:

```bash
PYTHONPATH=python:. python -m benchmarks.readme_defaults inputs \
  --inputs .artifacts/readme-defaults-inputs
srun --partition=main --nodelist=node1 --gres=gpu:5090:1 \
  --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=48G --time=01:30:00 \
  python -m benchmarks.readme_defaults fast --method pbe0 --mode direct --atoms 48 \
  --warm-repeats 3 --inputs .artifacts/readme-defaults-inputs \
  --source-revision e5fa4d2f0cd13775327261a0b6ced1eab6006c62 \
  --output .artifacts/pbe0-fast-direct-48.json
```

Set `PYTHONPATH`, `GENERATIVEQC_LIBRARY` and toolchain paths as in `run-n1.sh`.
The frozen auxiliary payloads are also retained in `auxiliary-inputs.json.xz`.
Offline replay needs Python, NumPy and Matplotlib 3.10.9, with no GPU or oracle.
The current three-profile v2 renderer does not accept these historical v1
receipts. Replay their original four-profile renderer from the already-published,
reviewed source below; do not relabel the receipts as v2. Run from the repository
root with that Git object available locally (no automatic download):

```bash
git worktree add --detach .artifacts/readme-defaults-v1 \
  9f3fa27a2017600b8ef986ac2cfc3545e85dfd29
(
  cd .artifacts/readme-defaults-v1
  PYTHONPATH=python:. python -m benchmarks.readme_defaults_report \
    --receipts benchmarks/results/readme-defaults-20261020/receipts.json.xz \
    --output .artifacts/readme-defaults-replay
)
```

Use a fresh worktree destination. Replayed files are
under `.artifacts/readme-defaults-v1/.artifacts/readme-defaults-replay`;
this does not modify the current v2 protocol or retained measurements.

The renderer retains median/range variability, rejects missing warm repeats and
never substitutes old measurements. `publication.json` authenticates the
selected evidence and explicitly records its unqualified scope.
