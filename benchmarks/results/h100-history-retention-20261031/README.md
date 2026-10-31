# Historical H100 payload recovery

Seven historical storage files (1,218,292 bytes) are retained unchanged at the
already merged Git revision `5bf1d00a9f4de2f47e381d2c9fba66fd43bf51ba`.
`snapshot.manifest.json` pins each original path, size and SHA256. The normal
checkout retains numerical summaries, model/threshold declarations, original
source identities and test fixtures. No scientific outcome is changed: the old
integrated baseline remains a single-process 19/30 exploratory result, the PBE0
matrix retains all eight slower native groups, and small CBS remains incomplete
with 16 numerical returns, two refusals and 14 NOT_RUN slots.

From the repository root, restore to a **new** ignored directory using the
unchanged standard recovery tool:

```bash
python tools/restore_retained_evidence.py --all \
  --manifest benchmarks/results/h100-history-retention-20261031/snapshot.manifest.json \
  --output .artifacts/h100-history-20261031
```

Recovery verifies every original Git object before creating the destination. It
uses no network, solver, GPU, archived script execution or external archive. If
the exact Git object is missing, the tool reports the recorded revision to fetch;
it does not fetch automatically. Existing outputs are refused.

Replay the recovered data explicitly:

```bash
uv run --no-env-file --no-project --python 3.12 --with numpy==2.2.6 \
  python benchmarks/results/h100-integrated-baseline-20261030/replay.py \
  --payload-root .artifacts/h100-history-20261031/benchmarks/results/h100-integrated-baseline-20261030
python -I -S -B benchmarks/results/h100-pbe0-direct-df-20261029/replay.py \
  --payload-root .artifacts/h100-history-20261031/benchmarks/results/h100-pbe0-direct-df-20261029
python benchmarks/results/h100-small-cbs-correctness-20261024/verify_actual.py \
  --capsule .artifacts/h100-history-20261031/benchmarks/results/h100-small-cbs-correctness-20261024/actual-terminal-originals-v2.json.xz \
  --output .artifacts/h100-cbs-restored-originals-20261031 \
  --report .artifacts/h100-cbs-offline-replay-20261031.json
```

The original CBS `publication.json` is recovered beside its capsule. It describes
the original self-contained publication; the current checkout is a compact
historical summary plus verified Git recovery, rather than an incomplete copy of
that publication. Its dirty-source patch, numerical records and fixtures remain
tracked. All historical readers retain their original numerical and provenance
gates; the new arguments select only where already verified payloads are read.

The 80 MiB aggregate, 1 MiB per-file and 2 MiB change-review limits are unchanged.
Current production DF provenance and the WB97MV archive used by ordinary CI remain
in the checkout. This storage change does not accept source492 or any later
H100 implementation and does not authorize compute or distribution publishing.
