# Resident KS final-state products: scoped PBE0 evidence

This is an **algorithm/dataflow comparison**, not another GPU4PySCF timing
contest. The inspected GPU4PySCF commit is
`c1a6e371d1a46afc9932c618108a4bc70e895edf`. Its gradient owner consumes
density/orbital data without executing our live native snapshot proof. We keep
that proof and every scientific acceptance gate, but reuse its resident inputs
and intermediate products. For canonical nonempty spins, two redundant full-AO
products disappear (11 becomes 9, plus the unchanged occupied-rank density
reconstruction). The asymptotic matrix work remains cubic. Legacy dense exports
and host orbital-energy W construction remain; this is not a zero-transfer path.

Both direct J owners already use density-precontracted MD/Hermite work.
GPU4PySCF's Rys K versus our generated exact K merits separate recurrence/work
analysis, not a wholesale replacement by name. Its two-pass SCF-XC integration
aggregates device-grid density/XC before revisiting AO blocks. Our retained AO
panels trade producer reuse against small/ragged contractions; larger AO unions
can increase quadratic work. Occupied projection costs roughly O(B*m*o), versus
local-density O(B*m*m), so it is not cheaper when active m is below occupied o.
Full moving-grid response, full-density Focks and direct/DF method contracts
must remain matched. No omitted response or relaxed screening is a speedup here.

## Source-matched results

Five interleaved pairs per original/moved geometry and 48/96 atoms are retained
in **each** cohort. Times are complete synchronized E+F host returns in seconds.
The unchanged descriptive gate is a reduction greater than
`max(2%, 2*(relative_MAD_baseline + relative_MAD_candidate))`.

| Source cohort | Atoms / AOs | Geometry | Baseline | Candidate | Reduction | Gate |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Prototype V4 / observer V5 | 48 / 384 | warm | 4.881321423 | 4.822977915 | 1.195% | below gate |
| Prototype V4 / observer V5 | 48 / 384 | moved | 4.920020342 | 4.820484616 | 2.023% | pass |
| Prototype V4 / observer V5 | 96 / 768 | warm | 14.691069994 | 13.755493056 | 6.368% | pass |
| Prototype V4 / observer V5 | 96 / 768 | moved | 14.699058909 | 13.759123527 | 6.395% | pass |
| Submission PR V1 | 48 / 384 | warm | 4.876250710 | 4.740550406 | 2.783% | pass |
| Submission PR V1 | 48 / 384 | moved | 4.890742093 | 4.791956812 | 2.020% | pass |
| Submission PR V1 | 96 / 768 | warm | 14.595477730 | 13.687022794 | 6.224% | pass |
| Submission PR V1 | 96 / 768 | moved | 14.526292738 | 13.641281232 | 6.092% | pass |

Do not pool cohorts, erase slow samples, or attribute their difference solely
to the shared-provider wrapper. The submission reconciles master
`c820ad08de1ceb33bc19c79115e0af2d52b01784` and moves identical product arguments
under existing tensor vendor ownership. Its reconstruction patch SHA256 is
`d44b766d753414cf7c2fdc1895b6dbac999901ebf1ada7069971c0c1eb5ee5ad`.
Later master is **not** the measured source.

All 80 timed records pass unchanged independent E/F gates (1e-8 Eh / 1e-7
Eh/Bohr), one-iteration/one-Fock replay, scoped semantic-work parity, resource
bounds and frozen-seed identity checks. Submission maximum E/F errors are
6.366463e-12 / 2.467937e-11 at 48 atoms and 9.094947e-12 / 2.961187e-11 at
96 atoms. Available work counters are not executed screened J/K quartet counts.

## Qualification boundaries

- Default remains `GENERATIVEQC_CUDA_KS_DEVICE_FINAL_VALIDATION=0`.
- Finite CPU-build job 7218 and GPU job 7222 qualify the submission; the GPU
  job preserves assigned visibility 3 on node1, `main`, `gpu:5090:1`.
- Host/native independent product and shared DF contracts pass. Memcheck reports
  zero errors; racecheck reports zero hazards/errors/warnings.
- The broader KS suite fails `PBE0 density provider admission route` with both
  selectors in the same candidate binary. It has seven AOs, below the AO>=17
  device-proof threshold. This is **not** a pristine-master control or all-green
  KS qualification, and the unrelated assertion is not repaired here.
- Admitted proof exports add 112 bytes and two drains. Four repeated stationary
  exports per case retain legacy bytes, one read/drain and the same identity;
  the published W/total-D lease blocks scratch reuse.
- Earlier analytic-fixture failure, observer-construction failures and broader
  KS assertion/control logs are retained independently, not rewritten as passes.

## Storage and reproduction

`publication.json` authenticates the standard `validation.json.gz` envelope,
lossless `raw-receipts.json.xz`, both reconstruction patches and this summary.
Raw receipts include exact UTF-8, SHA256 and byte sizes for selected original
logs, samples, references, configuration, compiler/cache receipts and recipes.
Large full-build logs and generated command/source listings stay ignored; their
source-inventory digests and selected executed compiler commands are retained.
Actual NPZ checkpoint files stay ignored locally. The independent analyzer
verified their file hashes and numeric blobs before packaging; offline tests
only authenticate those receipts, not reconstruct NPZ contents from hashes.

To fit the unchanged aggregate budget, the existing
`tools/compact_evidence_publications.py` is also applied to the density-candidate
GPU and psss-fock publications with a local 70-KiB selection threshold. This
only losslessly gzips their sample members and rebinds storage hashes/references;
their decoded numerical records, sample ordering and decisions are unchanged.

Offline validation from the repository root:

```bash
PYTHONPATH=python:. python -m pytest -q tests/python/test_ks_final_validation_evidence.py
python tools/compact_evidence_publications.py --check
```

For GPU reproduction, first decode `raw-receipts.json.xz` and authenticate every
`files` member against its stored size/SHA256 before extracting it beneath an
ignored working directory ROOT. Build a fresh checkout at the submission base
plus `submission-source.patch.gz`, using a verified sccache/ccache launcher,
CUDA 12.9, sm_120, Release/AOT and the recorded `configure.log`/`build.sh` options.
Do not reuse the measured library SHA as a new-build claim. Make ROOT/baseline
point to that checkout: the retained factory loads its paths and stationary
generator/header/AOT files. Place `recipes/retained-owner.py` at
ROOT/evidence/bulk-point-v7/paired-ao-endpoint.py; extract both
`reference-*-retained-perf5.json` files there, and the submission
`paired-ks-validation-endpoint.py` recipe into ROOT/evidence/reproduction/.

Set `PYTHONPATH` to the checkout's `python` and root, `GENERATIVEQC_LIBRARY`
to its rebuilt native library, `GENERATIVEQC_BENCHMARK_SOURCE` to its actual
base/patch identity, and `GENERATIVEQC_BOUNDED_ANGULAR_FORCE=1`. Preserve the
remaining recorded runner settings, including force mapping/materialization
and packaged stationary admission. Adapt the retained runner's site-specific
`env-next.sh` paths; do not treat them as portable prerequisites. Run each atom
count through finite Slurm, never overriding its assigned device visibility:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:30:00 \
  python "$ROOT/evidence/reproduction/paired-ks-validation-endpoint.py" \
  --root "$ROOT" --atoms 48 --repeats 5 \
  --basis-file "$CHECKOUT/benchmarks/results/pbe0-def2-svp-20261003/def2-svp-ho.json" \
  --reference "$ROOT/evidence/bulk-point-v7/reference-48-retained-perf5.json" \
  --output "$ROOT/evidence/reproduction/paired-48.json"
```

Repeat for 96 atoms, then run the extracted `recipes/analyze-endpoints.py` on
both new records and actual checkpoint files. New runs retain their own source,
binary, scheduler and timing identities; no existing sample is replaced.
