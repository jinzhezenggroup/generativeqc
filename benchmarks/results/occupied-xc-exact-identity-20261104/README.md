# Occupied XC: exact original-D restriction

This capsule measures source `0b9adb1a6dfcb7bae88f922235d247d52bf09bb9`,
not the earlier roundoff-only candidate or historical force arrays. Scientific
and test inputs were dirty during measurement and committed unchanged at that
revision; `receipt.json` records their verified hashes and artifact identities.

## Actual qualification

- 162 focused portable/compiled host cases passed; 168 additional consumer cases
  passed. Another 510 opt-in mixed CUDA cases were skipped, not qualified.
- Two new bitwise regressions both fail against the preceding occupied template
  at `9b17638b2b8cc49268eae37159fd1fe30c71cb16` and pass with the restriction.
- Slurm job **15811**, node1, main `gpu:5090:1`, finite ten-minute allocation:
  **138 passed normally and 138 each under memcheck, synccheck and racecheck**,
  zero reported errors/hazards/warnings. Assigned visibility `0` was preserved.
- The real emitted GPU SCF Gram producer supplies 24 orthonormal-frame cases.
  Six more cases execute actual compact batch specializations with ragged AO
  maps, offsets, empty spins, overflow errors, capacity fallback and guards.
- Comparisons require bitwise equality with the retained original-D CUDA
  consumer, not merely an allclose threshold or unrounded factor identity.
  Standalone resources and exact source/binary identities are in the receipt.

The contract now requires exact original-D features, including signed zero,
before selecting a candidate. A small forward-error budget is only a prefilter.
Because exact verification repeats dense work, KS defaults to ordinary matrix
products; only explicit `GENERATIVEQC_CUDA_KS_OCCUPIED_XC=1` requests the verified
experimental route. No tolerance or acceptance gate is relaxed.

## Reproduction

From the measured checkout, portable checks:

```bash
CCACHE="$(command -v sccache)" PYTHONPATH=.:python python -m pytest -q \
  tests/python/test_xc_occupied_certificate.py \
  tests/python/test_xc_occupied_features.py tests/python/test_xc_compact_batches.py \
  tests/python/test_xc_occupied_resources.py tests/python/test_dft_xc_compiled_resources.py \
  tests/python/test_compiler_structure.py tests/python/test_dft_xc_schedule.py
```

On node1, verify `sccache --version`, then execute every real-device run through
Slurm (do not override its visibility). The normal device command is:

```bash
srun --input=none --partition=main --nodelist=node1 --gres=gpu:5090:1 \
  --nodes=1 --ntasks=1 --time=00:10:00 bash -lc '
    GENERATIVEQC_TEST_OCCUPIED_XC=1 PYTHONPATH=.:python \
      python -m pytest -q tests/python/test_xc_occupied_features_cuda.py'
```

Within the same finite allocation, prefix that Python command with
`compute-sanitizer --tool TOOL --target-processes all --error-exitcode 99`
for each of `memcheck`, `synccheck`, and `racecheck`. Diagnostic suite durations
in the receipt are not molecular endpoint timings. Raw stdout/cache/resource
streams remain ignored local artifacts with compact recorded byte/hash identity;
they are not copied into tracked evidence as redundant test logs.

## Acceptance boundary

This is an exact-feature correctness restriction and compiled-batch device
qualification, **not full native owner integration or molecular E+F acceptance**.
The full native library's resource envelope, independent molecular/default-grid
E+F and complete cold/warm endpoint cost remain required before merging/closing
#2751. No speedup, default promotion or link to the cause of #2915 is claimed.
See the [decision note](../../../.agents/notes/implemented/numerics/2026-11-04-occupied-xc-exact-identity.md).
