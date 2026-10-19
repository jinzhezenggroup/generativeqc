# PR #2609: actual per-Fock cost and qualified improvements

The user's concern was correct: similar cold energy-plus-force totals concealed
24 native Focks versus 59 GPU4PySCF SCF iterations / 60 potential builds.
This follow-up measures actual per-round work and qualifies a tighter Hermite
source bound plus explicit FP64 XC fused multiply-add.

RTX 5090, CUDA 12.9.1, 96 atoms, spherical basis, FP64/direct/noDF, unpruned
48x16x32 moving grid. These are matched-grid implementation diagnostics, not
independently grid-converged production rankings. Cold/moved have one sample;
warm has five retained samples and reports the median.

| Measure | Previous default | Candidate |
| --- | ---: | ---: |
| r2SCAN-3c per-Fock J+XC elapsed median | 4.587006 s | 3.257288 s |
| r2SCAN-3c complete cold E/F (24 Focks) | 158.655704 s | 125.702913 s |
| r2SCAN-3c complete warm E/F (1 Fock) | 25.963102 s | 24.697389 s |
| r2SCAN-3c moved E/F (13 Focks) | 97.468332 s | 80.255866 s |
| r2SCAN-3c moved-warm E/F (1 Fock) | 25.944059 s | 24.708059 s |
| PBE0 complete cold E/F (17 Focks) | 62.452559 s | 57.834300 s |
| PBE0 complete warm E/F (1 Fock) | 10.373085 s | 10.112376 s |

All 18 E/F gates pass, with maximum errors 9.10e-12 Eh / 4.51e-11 Eh/bohr
against gates 1e-8 / 1e-7. Counts and per-Fock AO work are unchanged. The
per-Fock intervals exclude initialization, solver and forces, and come from
separate profiled runs; clean endpoint samples do not include counters/profiling.

GPU4PySCF still has lower normal-configuration per-round cost: 2.173046 s
median `get_veff` over 55 calls in a separate 54-iteration profiled trajectory.
Its clean cold is 153.314540 s over 59 iterations / 60 calls. Do not divide
one population's times by another's counts. Native AO cutoff is 1e-16 versus
upstream default 1e-10, J omission contracts differ, and solver initialization
and convergence metrics differ. Same-density/same-cutoff XC is 1.797135 s
native versus 1.918083 s upstream occupied-factor / 2.444948 s upstream dense.
J remains the principal residual gap; per-round parity is not claimed.

After rebasing onto the updated PR head, n4:1191 repeats all 18 E/F gates with
the same work counts. Integrated r2SCAN-3c cold/warm/moved/moved-warm are
125.866900 / 24.619059 / 80.190273 / 24.625029 s; PBE0 cold/warm are
57.833701 / 10.109733 s. Source identities and every sample are retained in the
record's `integrated` section; the table above remains the isolated comparison.

## Retained records

- `evidence.json.gz` (deterministic gzip JSON): full-precision comparison samples, component event samples,
  semantic work counts, numerical errors, source/binary identities and resolved
  qualification setup failures. Complete force vectors and CUPTI streams remain
  in ignored run storage; this record is a performance report, not a permanent
  scientific reference fixture.
- `measured-source.patch.gz`: decompress and apply to reconstruct the measured production sources on
  `992c744575b2953390fb4b54b41096dc12666988`. Source identity is
  `cb3cee83afc8d8f066b89ceb1548b5605a17ee7e329dbb10436427b4e69f3888`;
  library SHA256 is
  `f886f18ad70c6330cb83ab589139f83ab55976cdc5b940fd204d674e2022bbdc`.
- [Decision and invariants](../../../.agents/notes/implemented/performance/2026-10-19-per-fock-source-norm-xc-fma.md):
  screening proof, rounding policy, fallbacks, qualification scope and remaining
  differences. The prior MD radial FMA experiment remains rejected.

Raw run root: `.artifacts/pr2609/per-iteration/`. n4 jobs 1186 (baseline
profile), 1187 (source-norm-only fixed J), 1188 (combined fixed XC), 1189 (clean
combined endpoints), 1190 (combined profile and separate work census). n2:3486
runs device MD/native XC and four sanitizer tools; n2:3488 runs 133 additional
arithmetic/cancellation tests. Source-matched remote roots are
`/data/jzzeng/qc-pr2609-{defaults,source-norm,xc-fma}-20261019`.

## Reproduce

Build the measured production source with the `cuda-release-sm120` preset,
verified sccache/ccache launchers and the six stationary-force manifest targets.
Keep `--fmad=false`; the candidate explicitly fuses only the qualified XC
expressions. Use matching generated libraries/manifests. The linked decision
and existing [MD guide](../../../docs/developer/md_j_engine.md) define the
qualification and resource contracts.

With the built library, Python runtime, GPU4PySCF 1.8.1, PySCF 2.14.0, CuPy
13.6.0 and correction dependencies available, run on an allowed RTX 5090 node:

```bash
export GENERATIVEQC_LIBRARY="$PWD/build/cuda-release-sm120/libgenerativeqc.so"
export PYTHONPATH="$PWD/python:$PWD"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
unset GENERATIVEQC_MD_J_ENGINE GENERATIVEQC_CUDA_XC_SPATIAL_ORDER
unset GENERATIVEQC_MD_J_COUNTS GENERATIVEQC_MD_J_WORK_COUNTS
mkdir -p .artifacts/pr2609-reproduce
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:40:00 python -m benchmarks.readme_dft --method r2scan-3c \
  reference --atoms 96 \
  --basis-file python/generativeqc/data/r2scan3c/def2-mtzvpp-h-ar.json \
  --density-fitting none --grid 48 16 32 --grid-pruning none --repeats 5 \
  --output .artifacts/pr2609-reproduce/reference.json
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:40:00 python -m benchmarks.readme_dft --method r2scan-3c \
  native --atoms 96 \
  --basis-file python/generativeqc/data/r2scan3c/def2-mtzvpp-h-ar.json \
  --density-fitting none --grid 48 16 32 --grid-pruning none --repeats 5 \
  --reference .artifacts/pr2609-reproduce/reference.json \
  --output .artifacts/pr2609-reproduce/native.json
```

Preserve Slurm's assigned device visibility. Use the same settings with method
`pbe0` and `benchmarks/results/pbe0-def2-svp-20261001/def2-svp-ho.json` for PBE0;
the retained PBE0 campaign measured cold plus five fixed post-cold replays only.
Run independent MD suites with `GENERATIVEQC_TEST_MD_J_ENGINE=1` and
`GENERATIVEQC_TEST_MD_J_NORMAL=1` inside finite `srun`, and native
`generativeqc_dft_cuda_tests` with default, `--spatial-grid`, `--point-batches`,
`--matrix-schedule` and `--mixed-matrix-contractions`. Work-counter and profiler
runs must remain separate from clean timing samples.
