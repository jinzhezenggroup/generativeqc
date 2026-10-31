# Retained DF projection evidence

The original three-file bundle is preserved byte-for-byte at Git revision
`ba424c457cff8bfd92ca0df62b42f03593a9197c`, pinned by
`historical.manifest.json`. Its complete populations, noisy cohorts and paired
repeats remain recoverable offline. Restore into a new directory with:

```bash
python tools/restore_retained_evidence.py --manifest benchmarks/results/ecnu-df-projection-20261030/historical.manifest.json --all --output .artifacts/df-history
python .artifacts/df-history/benchmarks/results/ecnu-df-projection-20261030/report.py
```

The restore and original report were verified before migration. Current portable
scheduling evidence is [retained separately](../ecnu-portable-force-20261030/).
