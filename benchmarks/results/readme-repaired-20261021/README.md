# Historical repaired default-grid comparison (2026-10-21)

The [2026-10-22 comparison](../readme-final-20261022/README.md) supersedes these
measurements. The original source was `cbd4d8074de35105717c8aac7cd1f86babbe90ec`:
192 complete populations / 768 complete E/F calls, 174 accuracy-qualified
populations and 18 retained accuracy failures. Standard and Fast each won
43/58 qualified warm comparisons. The large r2SCAN-3c DF regeneration cost
belongs to this historical implementation.

[Original full report and receipts](https://github.com/jinzhezenggroup/generativeqc/tree/7522994e2e2e477e11c41cb2c2dbe74a2ce600e9/benchmarks/results/readme-repaired-20261021) ·
[Original timings](timings.md) · [Original work](work.md) ·
[Original conditions](conditions.md)

The [verified snapshot](snapshot.manifest.json) binds every original file to
already-published master `7522994e2e2e477e11c41cb2c2dbe74a2ce600e9`. Exact local
Git recovery was checked before retiring superseded timing payloads. Independent
oracle receipts, auxiliary inputs and reuse contracts remain locally available.
No Release, tag, external archive or history rewrite is involved.

```bash
python tools/restore_retained_evidence.py --all \
  --manifest benchmarks/results/readme-repaired-20261021/snapshot.manifest.json \
  --output .artifacts/readme-repaired-history
```

Restore the original publication and replay script before reproducing this
historical campaign. Existing destinations are not overwritten; missing Git
history is reported without an implicit fetch. Current timings are retained
separately and never replace values inside this historical snapshot.
