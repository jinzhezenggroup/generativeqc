# Default-control HF48 force stationarity

This is a source-bound RTX5090 numerical repair, not a speedup, A100 result,
all-method qualification, or acceptance of an inaccurate performance control.
Both separate native processes block PySCF/GPU4PySCF imports. Their complete
cold endpoints include constructor, preparation, SCF, analytic forces and host
export, followed by three same-owner frozen-own-density warm calls.

RHF, spherical def2-SVP, Standard FP64 retain untouched public controls:
energy 1e-9, density 1e-5, screening 1e-12. The independent pinned CPU full-Fock
reference and original absolute-energy <1e-8 / maximum-force <1e-7 gates are
unchanged. The reference's orbital-gradient norm is 4.44794e-11.

| Actual n4 allocation | Cold seconds | Cold iterations/Focks | Maximum force error across all four calls |
| --- | ---: | ---: | ---: |
| Baseline, Slurm1532 | 6.366749 | 10 /10 | 1.818375e-7: all four fail |
| Strict force owner, Slurm1533 | 8.002465 | 14 /14 | 1.651619e-9: all four pass |

Candidate maximum energy error is 1.045919e-11. All warm calls use one
iteration/Fock; the original individual times, full arrays and work counters
remain in the capsule. The stricter force owner costs four extra cold Focks.
Do not present the failed baseline as an accepted performance population or
these two diagnostic processes as an ABBA comparison.

`cohort.json` binds original per-file UTF-8 texts, native source and ELF hashes,
source trees, scheduler assignments and the existing independent reference.
`observations.json.xz` retains both complete endpoint populations, the executed
oracle-isolated runner, native source patch, device/toolchain/cache receipts and
the first candidate's missing-include compilation failure. The corrected
candidate is a new source identity, not a discarded or relabelled sample.

The first configuration failed because of a Ninja path; the first finite
30-minute compilation expired. The resumed 90-minute build completed normally
with verified sccache0.16.0, Release, SM120-real and fast compilation disabled.
Both failures and full compiler output remain checksum-indexed in the ignored
task receipt archive named by the cohort. No evidence budget is increased.

Replay requires no GPU, reference installation, network or `.git`:

```bash
python benchmarks/results/hf-force-stationarity-20261109/replay.py
python -O benchmarks/results/hf-force-stationarity-20261109/replay.py
```

For scheduled reexecution, recover `runner.py` and `native.patch` through
`replay.py`'s authenticated `load`, build the corresponding source with the
recorded normal Release CUDA options and a verified compiler cache, and run the
driver through a finite `srun --partition=main --gres=gpu:5090:1` allocation.
Supply the exact retained reference geometry using `--geometry`, the source
tree with `--tree`, and an ignored destination with `--output`. Score outside
the native process against the pinned reference. Never override Slurm device
visibility. See the implemented HF force-stationarity Agent Note for rationale.
