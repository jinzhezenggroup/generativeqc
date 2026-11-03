# Current-source occupied-XC certificate candidate

This capsule retains **candidate feature/device evidence**, not a completed
#2751 repair or a performance/default qualification. Parent scientific source:
`b735793be54caed18fb2e2cacfa9235ecb935f7c`; `receipt.json` binds each changed
scientific/test input and the actually emitted/compiled products. The proposed
[decision note](../../../.agents/notes/proposed/2026-11-04-occupied-xc-stored-density-certificate.md)
contains the derivation, fallback boundaries and remaining gates.

## Observations

- Focused host/compiled-probe/portable tests: **156 passed**. These include
  independent 800-digit Decimal bounds, original-D cancellation, ordinary
  admission, RKS/UKS/empty spins and execution of actual occupied compact batch
  bodies with nonzero map/scratch offsets and scratch-owner sentinels.
- Final emitted CUDA suite: **108 passed** normally and **108 passed each**
  under memcheck, synccheck and racecheck; zero reported errors or hazards.
  Slurm job **15794**, node1, `main`, `gpu:5090:1`, `00:10:00`; assigned
  visibility `0` preserved. Device: RTX 5090, NVIDIA driver 580.95.05,
  nvcc V12.9.86, strict FP64 `--fmad=false`, `sm_120`.
- The unchanged finite-probe harness from evidence commit `33f2c1648` is
  re-emitted against the candidate, not relabelled from its old output. Slurm
  job **15796**, node1, `main`, `gpu:5090:1`, `00:02:00`, visibility `0`:

| Finite input | Stored D00 | Dense rho | Candidate rho | Device error |
| --- | ---: | ---: | ---: | ---: |
| Ordinary identity-frame control | 2 | 0.125 | 0.125 | 0 |
| Orthonormal finite underflow boundary | 0 | 0 | 0 | 0 |

All other four features are zero in that probe; all returned values are finite
and the structural witness still matches. The retained earlier baseline in
PR #2969 returned occupied rho=2 for the same boundary. Density formation in
this probe is **host FP64 with explicit FMA**, not a GPU density-producer test.
Mapped/all-jet versions are also exercised by the actual emitted CUDA suite.

## Resource and provenance limits

`resources.txt` is offline inspection of the normal standalone probe image,
with trailing whitespace removed and numeric fields unchanged.
Its occupied feature kernel uses 72 registers, 40 stack bytes and zero declared
LOCAL bytes; its occupied product uses 34 registers, 56 stack bytes and 5376
shared bytes. These are not full-library compiled-batch/native-owner envelopes.
Stack, warp votes, upward arithmetic and possible original-D recomputation are
real costs. Suite wall times are **not** molecular endpoint timings.

The existing verified sccache 0.16.0 cache is reused. Retained before/after
statistics describe shared servers, not attributed per-compile cache hits.
Normal and three sanitizer source hashes agree; their library hashes differ
and are all recorded rather than substituted. Generated sources/binaries are
retained on node1 at the receipt's local paths, not uploaded as Release assets.

The internal per-feature roundoff rule is **not an XC E/F sensitivity proof**.
Native full-plan/batch GPU integration, point/domain E/Vxc/force acceptance,
independent molecular/default-grid E+F, whole-runtime resource envelopes and
complete admission-inclusive cold/warm cost remain unqualified. The existing
automatic selector is unchanged. Keep the implementation PR draft and #2751
open until those gates are satisfied; old force receipts are not new evidence.

## Reproduction

Host checks run without a GPU; the fixture's `CCACHE` executable can be pinned
to the verified sccache launcher:

```bash
sccache --version
CCACHE="$(command -v sccache)" PYTHONPATH=.:python python -m pytest -q \
  tests/python/test_xc_occupied_certificate.py \
  tests/python/test_xc_occupied_features.py tests/python/test_xc_compact_batches.py \
  tests/python/test_xc_occupied_resources.py tests/python/test_dft_xc_compiled_resources.py \
  tests/python/test_compiler_structure.py tests/python/test_dft_xc_schedule.py
```

On a compatible node with the exact candidate checkout, compiler and test
dependencies, run every actual device test through finite Slurm allocations:

```bash
srun --partition=main --nodelist=node1 --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:10:00 bash -lc 'GENERATIVEQC_TEST_OCCUPIED_XC=1 \
    PYTHONPATH=python:tests/python python -m pytest -q \
    tests/python/test_xc_occupied_features_cuda.py'
```

The same command is prefixed inside the allocation with `compute-sanitizer
--tool memcheck|synccheck|racecheck --error-exitcode 86 --target-processes all`
for the three sanitizer cohorts. Do not unset or override the visibility
assigned by Slurm and do not use node3 for device execution.
