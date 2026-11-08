# Default-control HF48 force stationarity

This is a source-bound RTX5090 numerical repair, not a speedup, A100 result,
all-method qualification, or acceptance of an inaccurate performance control.
All three separate native processes block PySCF/GPU4PySCF imports. Their complete
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
| Formatting-only requalification, Slurm1534 | 7.948143 | 14 /14 | 1.651475e-9: all four pass |

Maximum energy error is 1.045919e-11 before formatting and 1.318767e-11
after formatting. All warm calls use one
iteration/Fock; the original individual times, full arrays and work counters
remain in the capsule. The stricter force owner costs four extra cold Focks.
Do not present the failed baseline as an accepted performance population or
these diagnostic processes as an ABBA comparison. The unchanged baseline was
not rerun; the formatted candidate is a separate source-bound population.

`cohort.json` binds original per-file UTF-8 texts, native source and ELF hashes,
source trees, scheduler assignments and the existing independent reference.
`receipts.json.xz` and `fp64.xz` use the existing exact-FP64 v3 codec, with both
streams authenticated by `sha256.txt`. They retain all three complete endpoint
populations, the executed oracle-isolated runner, both native source patches,
device/toolchain/cache receipts and the missing-include compilation failure.
Every original text and its checksum survives under `before-format/`; byte-exact
decoding was verified during migration. No observation is discarded or relabelled.

The final candidate has native source identity
`99f7dfb96e87f420abfd75b230e8dc16f105336c868d821d8a91b01f86ab1d6a`
and ELF SHA256 `2284e04f2e324089cf323eb2d19cad460d0bdd6ecdb58aa7e484f16456476070`.
The earlier candidate's distinct identities remain in the cohort and original
receipts. Formatting changed the native source identity, so the final image was
rebuilt and independently rescored rather than borrowing the earlier image's pass.

The first configuration failed because of a Ninja path; the first finite
30-minute compilation expired. The resumed 90-minute build completed normally
with verified sccache0.16.0, Release, SM120-real and fast compilation disabled.
Both failures and full compiler output remain checksum-indexed in the ignored
task receipt archives named by the cohort; the original archive is unchanged.
No evidence budget is increased.

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
