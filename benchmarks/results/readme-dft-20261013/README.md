# Historical HF/DFT README baseline, 2026-10-13

The live `summary.json.xz` and `df-summary.json.xz` remain **byte-for-byte
unchanged**. They retain the complete endpoint samples, failures, reference
bindings and measured source/build identities consumed by the current
[WB97M-V composition](../readme-wb97mv-20261014/README.md) and
[r²SCAN-3c composition](../r2scan3c-final-validation-20261015/README.md).
These are historical measurements, not qualification of a newer native image.
`manifest.json` authenticates these two retained numerical inputs.

The original campaign compared complete cold/warm/moved/moved-warm energy plus
analytic-force endpoints for HF, PBE, PBE0, B3LYP, r2SCAN, r2SCAN-3c, M06-2X
and ωB97M-V in Direct and DF. Its task-start source was master `e0ad22036`,
with individually identified follow-up builds. Missing references, partial
populations and unfavorable attempts were not converted into passing results.
Current plots use the later composition reports, not the original figures.

## Exact historical report recovery

To keep the unchanged 64 MiB results budget while retaining the new complete
r²SCAN-3c comparison, superseded charts/tables, prior-publication copies,
historical qualification receipts and reconstruction/audit reports now use
the existing Git-history restoration contract. They are not current numerical
consumers, permanent scientific fixtures or new externally published assets.

`snapshot.manifest.json` pins **all 15 original campaign files**, including
the original README, manifest, verifier, source reconstruction and both live
summaries, at the already merged commit
`449cd63ccfbdb298c97b78f3c2d85edff0041310`. Every original path, byte count
and SHA-256 was verified by a complete restoration before retiring copies.
No numerical values, gates or observations were rewritten. The original
manifest is recovered with its members rather than left as an active manifest
pointing to absent files.

From a checkout containing that historical commit, restore into a **new**
directory and run the original opt-in verifier:

```bash
python tools/restore_retained_evidence.py --all \
  --manifest benchmarks/results/readme-dft-20261013/snapshot.manifest.json \
  --output /tmp/readme-dft-history
PYTHONPATH=python:. python \
  /tmp/readme-dft-history/benchmarks/results/readme-dft-20261013/verify.py
```

The original verifier also supports `--rerender` in its documented rendering
environment. Restoration never fetches implicitly; a shallow clone may need
an explicit fetch of the pinned commit before this optional historical audit.
Normal current-panel replay and scientific tests still use the two local
summaries and do not require a history fetch or artifact service.
