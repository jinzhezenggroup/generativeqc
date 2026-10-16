# Historical README endpoint investigation, 2026-09-22

This was an **unqualified diagnostic draft**, not a current README performance
claim. Measurements used master `572fbd6d4cfc3dd0db03fb0d1abaac4add0b9f12`:
matched GPU HF energy/forces, partial DFT SCF energies and an internal composed
CCSD(T) endpoint. Unsupported public paths were explicit rejections, not zero
timings. Correctness, completeness and capability blockers stopped publication;
no later fix is inferred to qualify these old observations. See the preserved
[diagnosis](../../../.agents/notes/proposed/2026-09-22-gpu-benchmark-blockers.md).

The historical inputs used spherical def2-SVP and 3–96 atom water prefixes,
explicit cc-pVDZ-JKFIT for fitted paths, matched reference grids, and independent
HF gates of `1e-8 Eh` and `1e-7 Eh/Bohr`. The original report records all measured
latencies, failed/stopped and unavailable branches, convergence gates, binary
identities, raw-file checksums and its distinct internal CCSD(T) contract.
Those historical limitations remain part of the evidence, not current support
or issue-state claims.

## Exact recovery

No current scientific test, reference fixture or README composition consumes
these draft records. To retain new qualified r²SCAN-3c endpoints without
raising the 64 MiB results budget, the superseded samples/charts now use the
existing Git-history restoration contract. No external archive is published.

`snapshot.manifest.json` pins all **10 original files**, including the full
report, source/input provenance, paired samples, checksummed reduction and
figures, at already merged commit
`0cce0708e551621838cb84c79aecba664eebf1cf`. Every original file was restored
and checked against its exact byte count and SHA-256 before retiring copies.

```bash
python tools/restore_retained_evidence.py --all \
  --manifest benchmarks/results/readme-20260922/snapshot.manifest.json \
  --output /tmp/readme-draft-history
```

Read the restored `benchmarks/results/readme-20260922/README.md` for original
commands, full outcomes and limitations. Recovery never fetches implicitly;
a shallow clone needs an explicit fetch of the pinned commit for this optional
historical audit. Normal offline tests do not need that history.
