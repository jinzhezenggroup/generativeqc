# Scalar DF exchange layout — 2026-10-16

This is a source-matched CPU follow-up to
[`../range-separated-df-stable-20261015/`](../range-separated-df-stable-20261015/).
It optimizes the fitted Hamiltonian rather than replacing it with Direct LR.
Earlier negative receipts remain intact. The fitted-LR selector stays explicit;
these measurements do not promote a different default Hamiltonian.

## Identity and controls

- Base source: `f1cabc4bb3f553f7b5da471e378290993fd6f6a6`. The baseline already
  includes the explicit-opt-in selection fix. `baseline-production.patch`,
  `v3-source.patch` and the two rejected layout patches reconstruct the measured
  production changes. `v3-identity.json` binds source, binary and compiler hashes.
- Native CPU Release/O3, scalar linear algebra without OpenBLAS, AMD EPYC 7K62,
  one CPU/BLAS thread on a shared host. No real-GPU execution or new CUDA
  qualification. Stationary CPU force AOT is disabled in this build; no native
  non-DF force or new force-speed qualification is implied. Build commands use
  the verified ccache launcher and preserve
  its shared cache; global cache statistics are not per-build hit counts.
- WB97M-V RKS, spherical def2-SVP/def2-TZVP, retained cc-pVDZ-JKFIT,
  FP64, metric cutoff 1e-10, energy/density tolerances 1e-10/1e-8,
  screening 1e-14 and Hcore initial guess. The unpruned 24 x 6 x 12 grid is
  deliberately the historical controlled grid, not the production default.
- Separate prepared owners execute complete public energy endpoints in
  same-process ABBA order. Water has seven interleaved pairs; the dimer has five.
  Clean timings have no `LD_PRELOAD`. Diagnostic interposer runs are separate.
  Actual loaded-library paths are asserted and hashes retained.
- Cold is one observation per arm, including preparation in the table below;
  it is not a statistically qualified cold-speedup campaign. Warm measurements
  reuse converged densities. Changed geometry is measured for water/SVP only.

## Same-Hamiltonian optimization

Both arms explicitly select fitted LR. Seconds, medians for warm replay:

| Case | Cold baseline → optimized | Warm baseline → optimized | Relative reduction |
| --- | ---: | ---: | ---: |
| Water/SVP, 24 AOs | 3.496893 → 3.441972 | 0.838583 → 0.830309 | 0.99% |
| Water/TZVP, 43 AOs | 5.010584 → 4.738689 | 1.173277 → 1.111625 | 5.25% |
| Water dimer/TZVP, 86 AOs | 38.893122 → 29.569751 | 7.952534 → 5.901382 | 25.79% |

Water/TZVP and the dimer exceed the unchanged comparison gate:
max(2%, twice the summed relative MAD). The SVP difference is below that gate;
no significant SVP improvement is claimed. Both arms retain two warm SCF
iterations, four physical Fock builds, four Coulomb applications and eight
exchange applications across the full-range and LR sources. Cold iterations
are 14 for water and 16 for the dimer. The grid populations remain 5,184 and
10,368, respectively; this optimization does not reduce VV10 grid-pair work.

Every cold and warm energy is checked against independently reconverged
PySCF/libcint energies for the fitted Hamiltonian at the unchanged 2e-8 Eh gate.
The dimer is an energy-only size qualification, not an admitted force workload:
its grid-pair domain still exceeds the existing 100,000,000 force-work bound.
No large-system, batch or whole-process memory bound is established here.

## Remaining selector tradeoff

A separate same-optimized-binary comparison measures the scientifically distinct
Direct and fitted LR models:

| Case | Cold Direct → fitted | Warm Direct → fitted | Fitted warm slowdown |
| --- | ---: | ---: | ---: |
| Water/SVP | 4.373644 → 3.442056 | 0.826038 → 0.829790 | 0.45% |
| Water/TZVP | 26.323994 → 4.733820 | 1.107172 → 1.115258 | 0.73% |

The historical approximately 1.02%/3.89% replay slowdowns are substantially
reduced, not proven zero. Neither arm improves by 2% in the warm selector
comparison; its assessment remains `not-run`, not a default-promotion pass.
SVP changed-geometry medians are 3.722061 s Direct and 2.825803 s fitted across
five interleaved pairs, with 11 SCF iterations and 13 Fock builds in both arms.
All original and moved SVP energies have independent oracle receipts. This
family does not remeasure changed-geometry TZVP or claim a new force speedup.

## Rejected attempts and correctness

The first cached layout leaves a 1.74% TZVP replay slowdown against Direct.
An exact-symmetry variant leaves 2.35% and was stopped before completing
geometry/oracle qualification. Preserve their partial receipts as rejected
diagnostics, not successful final-source measurements. The final transpose
schedule handles nonsymmetric B as well as physical symmetric tensors.

The retained validation command checks public energy/force replay and selector
freezing, independent fitted energy and force oracles, native RHF/UHF four-center
oracles, stale-cache rejection, resource/owner boundaries and documentation.
Native cache tests straddle both the small-size admission and 16 MiB scalar
retention boundary. The temporary transpose/row and uncached routes remain
explicit bounded-resource fallbacks. See the
[decision note](../../../.agents/notes/implemented/performance/2026-10-16-range-df-scalar-layout.md).

The final Python run reports **121 passed, 17 CUDA skipped and 5 failed**.
All five failures are non-DF native-force eligibility expectations in
`test_dft_mixed_native_forces.py`; the same five fail against the preserved
pre-optimization binary because stationary CPU force AOT is disabled. They are
not changed or hidden by this patch. RSH-DF endpoint/force oracles, full-range DF
force cases and the 89 snapshot/resource host contracts pass. Related native
CTest passes **11/11**; clang-format, Ruff, native API/ownership checks and strict
Sphinx also pass. The allocation-refusal branches are retained but are not
claimed as fault-injection qualification. No CUDA skip becomes a GPU pass.

## Retained bytes and reproduction

The Git-retained `evidence.json.xz` losslessly preserves raw JSON, timing/work
records, source patches, harness versions, failed attempts and validation logs.
`storage.json` authenticates stored and decoded bytes and member count. Binaries,
compiler objects and compiler caches remain ignored local artifacts.
The initial retention attempt exceeded the aggregate byte budget. Full verbose
compiler logs consequently stay ignored locally; the archive keeps their hashes,
sizes and exact 20-line completion excerpts in `build-receipts.json`. No timing,
oracle, rejected optimization or test receipt is removed to satisfy the budget.

First restore both bundles with the command in
[`../range-separated-df-stable-20261015/`](../range-separated-df-stable-20261015/).
The pinned existing Git objects retain every byte within the combined master/PR
evidence budget. Then inspect without executing historical scripts:

```bash
.venv/bin/python - <<'PY'
import hashlib, json, lzma
from pathlib import Path, PurePosixPath
root = Path("benchmarks/results/range-df-scalar-layout-20261016")
manifest = json.loads((root / "storage.json").read_text())
stored = (Path(".artifacts/range-df-history") / root / manifest["path"]).read_bytes()
assert hashlib.sha256(stored).hexdigest() == manifest["stored_sha256"]
decoded = lzma.decompress(stored)
assert hashlib.sha256(decoded).hexdigest() == manifest["decoded_sha256"]
files = json.loads(decoded)["files"]
assert len(files) == manifest["file_count"]
target = Path("build/validation/range-df-layout-restored")
target.mkdir(parents=True, exist_ok=False)
for name, text in files.items():
    relative = PurePosixPath(name)
    assert not relative.is_absolute() and ".." not in relative.parts
    path = target / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
print(f"Verified and restored {len(files)} members")
PY
```

The restored `warm_endpoints.py` takes explicit baseline/candidate library paths,
`--baseline-selector df` for same-Hamiltonian comparisons, `--basis`, `--case`,
`--repeats`, `--geometry-repeats` and `--oracle`. Build libraries from the retained
source patches with the recorded configuration, set `PYTHONPATH` to include
`python`, `tests/python` and the checkout root, and use new output paths; the
harness reserves receipts exclusively. Historical profiler runs are not clean
benchmark entrypoints. `validate.sh` records the exact final validation commands.
