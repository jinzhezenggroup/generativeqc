# Original data for the two three-process HF/DFT figures

This dataset supplies the original measurements underlying the two SVG assets
merged in PR #3093. It contains the approved H100 acquisition and the final
RTX 5090 acquisition, without adding the earlier configuration experiments.

| GPU | Original timing table | Plot summary | Formal processes | E+F calls | Cold / warm median wins |
|---|---|---|---:|---:|---|
| H100 | [timings.csv](h100/timings.csv) | [summary.csv](h100/summary.csv) | 384 | 1536 | 63/64, 62/64 |
| RTX 5090 | [timings.csv](rtx5090/timings.csv) | [summary.csv](rtx5090/summary.csv) | 384 | 1536 | 64/64, 64/64 |

Each timing-table row is one fresh process: engine, method, Direct/DF mode,
atom count, process-repeat index, one cold time, all three warm times, and the
cold SCF/Fock or effective-potential build counts. No times are rounded. The
summary tables contain every plotted center and min/max endpoint. There are
128 separate prewarm processes per GPU; their times do not enter these tables.

## Full original records, including energies and force vectors

Each GPU directory contains:

- `samples.json.xz`: all fields of all 384 formal process records and their
  successful command census, including full-precision energies, full force
  vectors, convergence/settings, precision diagnostics, correction energies,
  semantic work, software/device/allocation identities and original raw SHA256s.
- `inputs.json.xz`: the actual orbital/reference and auxiliary basis inputs,
  with exact original-byte reconstruction and hashes.
- `provenance.json.xz`: measured source/native-library/recorder identities,
  actual build options and outcomes, shared reference-reuse contract and
  acquisition configuration. These are acquisition records, not a new run.
- `measured-source.json.xz`: a durable public Git base, the recorded per-file
  text-export spelling and a literal source patch for the actual native inputs
  and timing producers. Recovery does not depend on an unpublished local commit.
- `publication.json` and `evidence.json`: checked storage bindings and explicit
  endpoint, numerical and promotion outcomes.

The original serializer is `json.dumps(record, indent=2, allow_nan=False)` plus
one LF. Removing only the collector's `process_repeat`, `raw_file` and
`raw_sha256` additions reproduces **every original formal JSON file byte for
byte**, including all fields not used by the plot. The loader verifies each
original hash and binds each repeat one-to-one to a distinct successful
command/output. Equal measured values are allowed; duplicate file ownership,
missing processes, relabelled repeats and edited records fail closed.

## Offline reproduction

From the repository root, with Python 3.12+ and `numpy`, `defusedxml`,
`array-api-compat` and `typing-extensions` available:

```bash
python -m benchmarks.paper_hfdft_replay --verify-only
```

For an isolated environment without building the native library:

```bash
uv run --isolated --no-project --no-env-file \
  --with numpy --with defusedxml --with array-api-compat --with typing-extensions \
  python -m benchmarks.paper_hfdft_replay --verify-only
```

Verification authenticates the publications, restores original raw-byte hashes,
checks all 768 processes and 3072 complete E+F calls, recomputes every statistic
and independent energy/full-force error, and reads the plotted SVG coordinates
back independently. It checks 256 medians and 512 range endpoints in **each**
SVG. The SVG coordinate rounding tolerance (2e-5 seconds) applies only to this
graphics comparison, not to chemical or timing acceptance gates.

Restore the actual original JSON receipts to ignored run storage:

```bash
python -m benchmarks.paper_hfdft_replay --restore-raw .artifacts/paper-raw
```

Recreate timing/summary tables, full per-call numerical errors and the combined
PNG/PDF/SVG figures (also requires `matplotlib`):

```bash
python -m benchmarks.paper_hfdft_replay --output .artifacts/paper-replay
```

The plotted values/ranges reproduce the merged figures. SVG identifiers,
metadata and raster font rendering can differ across Matplotlib versions;
byte-identical newly rendered PNGs are not an acceptance criterion.
Output paths inside `benchmarks/results` are rejected before replay begins.

## Measured source and new GPU acquisition

| Cohort | Measured revision | Native source identity |
|---|---|---|
| H100 | `efeaa3c7107a759e95c088d366e990d66c551c27` plus measured recorder overlay | `20d337f7ffdb1c75eb89a3ba6b0c307518e21388a4d4fda7df6b54248b4337e7` |
| RTX 5090 | experimental `4b8b103a57fbd4fd794e17dab405b6940b3adcea` plus measured recorder overlay | `e7335217cc4d94a4b450e1455d5164330441a02b1a6da98b822fe5c78e0726f9` |

Recover the measured native and recorder **bytes** without compiling or running
a GPU (the public EFE and `17f9317a2495c142941d7390d465fcf79887d643`
base objects must already exist locally; recovery never fetches implicitly):

```bash
python -m benchmarks.paper_hfdft_replay \
  --reconstruct-source .artifacts/paper-measured-source
```

The command verifies both native identities with the source-owned identity
generator and verifies the actual recorder hashes. It also reconstructs the
basis inputs under each destination's `measurement-inputs/inputs`. This recovers
the measured native/recorder closure, not historical Git commit metadata.
The source patch is retained as experimental reproduction data; it does not
install or promote the experimental implementation in the current source tree.

For a fresh GPU acquisition, use those reconstructed sources, the exact compiler,
provider/package versions and CMake flags in each provenance record, verified
compiler-cache launchers, and a real finite scheduler allocation. RTX uses
automatic SM120 AOT selection; H100 uses its recorded portable schedule.
Adapt filesystem paths to that allocated environment without changing scientific
controls. Feed the retained acquisition configuration to the reconstructed
`benchmarks.readme_cold_repeats` producer: it uses separate prewarms and three
fresh formal processes per engine/pair, balanced engine order, reversed middle
traversal, and finite parent/process deadlines. Offline replay does not request
compute resources or imply that a current-master binary has been remeasured.

## Statistical and scientific scope

Cold center/range is the median/min/max of three independent process cold calls.
Warm center/range is computed from three within-process warm medians; the nine
warm calls are not nine independent process replicates. There is no smoothing,
timing/accuracy filtering, sample replacement or inferential confidence claim.
H100 and RTX use their own measured sources; this is not a same-commit hardware
comparison or a universal speedup guarantee.

The unchanged independent reference archive is
`benchmarks/results/readme-refresh-20261026/oracles.json.xz`, SHA256
`d61a3e8ba0a8f8a8c08ee4c4b30c8e3e397e67863b8d75e7ba974e90cf0f85a4`.
Each GPU still has 348 qualified and 36 failed process qualifications: both
engines' DF-HF energy approximation and three-atom M06-2X grid/force failures
remain in the original data. Performance rankings do not turn these into
numerical passes. Full errors are regenerated in `qualification.json` with
`--output`; the retained error gates are unchanged.

## Storage

All original scientific values are retained in bounded XZ JSON records; no
transient build/profiler logs, compiled images or prior experiments are bundled.
Every file remains below the repository's hard 1 MiB limit. The exact incoming
CSV review exceptions and bounded aggregate addition are reviewed explicitly
with this dataset; they do not waive file-size or scientific checks.
