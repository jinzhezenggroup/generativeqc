# Historical default-grid README comparison (2026-10-21)

The [repaired comparison](../readme-repaired-20261021/README.md) supersedes this
campaign. Its original source was `0fe06f31c6c96182973405b58f5047f3f177fff4`:
192 attempted populations, 188 complete, 170 accuracy qualified, 18 accuracy
failures and four resource/time-limit failures. Direct ωB97M-V/96 warm speedups
were 2.61× Standard and 2.93× Fast under that source. They are historical
observations, not timings for the repaired implementation.

[Original complete report and receipts](https://github.com/jinzhezenggroup/generativeqc/tree/0e691d4000414ecb5fbf563232819178511e1642/benchmarks/results/readme-unified-20261021) ·
[Original timing table](timings.md) · [Original work](work.md) ·
[Original conditions](conditions.md)

The [verified Git snapshot](snapshot.manifest.json) binds all 19 original files,
989,391 bytes, to published master `0e691d4000414ecb5fbf563232819178511e1642`. Byte-for-byte recovery
was verified before retiring superseded timing payloads from the current tree.
The independent oracle receipts, auxiliary inputs and reuse contracts remain
locally available; the new campaign keeps its own complete live evidence.
No history was rewritten and no Release or external archive was created.

To restore the exact original report, publication manifest and replay script:

```bash
python tools/restore_retained_evidence.py --all \
  --manifest benchmarks/results/readme-unified-20261021/snapshot.manifest.json \
  --output .artifacts/readme-unified-history
```

Run historical reproduction from that restored snapshot with the original
measurement-driver revision. Existing destinations are never overwritten;
missing Git history is reported without an implicit fetch.
