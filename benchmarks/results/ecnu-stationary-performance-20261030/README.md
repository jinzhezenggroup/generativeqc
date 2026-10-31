# Historical ECNU A100 stationary force qualification

The exact five-file bundle from merged PR #2886 is retained in Git at
`bc77e0b62d5fe67c2502d6b8f5440ac27463c0f0`. The
[recovery manifest](historical.manifest.json) pins every original byte length
and SHA256, including the independent force arrays, frozen basis, observations
and reporting script. Recovery was verified byte-for-byte before removing the
duplicate checkout payloads; no numerical data or historical claim is relabeled.

These are matched-grid diagnostics on A100-PCIE-40GB with eight math threads,
GPU4PySCF 1.7.6 without cuTENSOR, cold + three warm + moved + three moved-warm
calls. They are separate from the later one-thread README-default campaign.
All 13 original cohorts pass their 1e-8 Eh / 1e-7 Eh/Bohr gates. WB97M-V/96
lacked a complete reference and was not ranked. Original warm medians include
r²SCAN-3c/96: 29.116 → 16.226 s (reference 15.269 s), and M06-2X/96:
11.672 → 9.835 s (reference 7.419 s); these results do not establish universal
GPU4PySCF parity.

Restore into a new ignored directory and rerun the original report:

```bash
python tools/restore_retained_evidence.py --all \
  --manifest benchmarks/results/ecnu-stationary-performance-20261030/historical.manifest.json \
  --output .artifacts/ecnu-stationary-history
python .artifacts/ecnu-stationary-history/benchmarks/results/ecnu-stationary-performance-20261030/summarize.py
```

The restored README contains the original complete protocol and reproduction
commands. If the pinned object is absent in a shallow clone, fetch the exact
recorded revision first. Restoration is local, checks every checksum and never
fetches or executes historical code automatically.

See the [decision note](../../../.agents/notes/implemented/performance/2026-10-30-stationary-ao-capacity-cliff.md)
for the original architecture, invariants and negative evidence.
