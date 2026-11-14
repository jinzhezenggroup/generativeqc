# Historical RTX 5090 RHF-default cohort — 2026-11-11

This is the **PR #3065-only** cohort, not the subsequent
[combined #3064/#3065 refresh](../readme-rtx5090-composed-20261111/README.md).
Measured source `8c9acd542840f5c059ce866dc2e9f11c0aaf245d` is based on master
`d35ca77a4c7142f8d6d3f0cbc1e8e96718d9af30`. All original timing/error/work
samples, software/build identities and acquisition events remain in
`samples.json.xz`; only historical reference-container storage digests change.

| Mode | Cold wins | Warm wins | Qualified cold | Qualified warm |
| --- | ---: | ---: | ---: | ---: |
| Direct | 32/32 | 32/32 | 31/31 | 31/31 |
| DF | 30/32 | 32/32 | 26/27 | 27/27 |

There are 128 selected processes / 512 complete E+force calls: eight methods,
Direct/DF, 3/12/48/96 atoms. Each process has one cold call including setup and
three warm calls from its own frozen cold density (median, min–max bars).
Node1/main Slurm pairs engines consecutively on the same assigned 5090;
first-engine order is balanced 32/32, numerical threads=1, caches retained.
Source-bound Release/SM120/CUDA 12.9.1 uses sccache 0.16.0. GPU4PySCF defaults
are unchanged. Native RHF selects the MO-gradient default; tighter controls
and unsupported owners retain strict behavior. This is not a 1e-7 force guarantee.

Both engines independently qualify 58/64 pairs at E/atom ≤1e-5 Eh, Fmax ≤1e-4
and Frms ≤3e-5 Eh/Bohr. Failures remain: all four DF HF sizes and Direct/DF
M06-2X3. All 1024 cross-engine vector comparisons pass but are not independent
validation. Cold DF HF96 is slower (37.030 vs 34.913 s), as is DF r2SCAN-3c96
(23.548 vs 23.229 s). Direct ωB97M-V96 warm is 9.37× faster; HF48/96 use 10/11
cold physical Focks. CBS is unchanged.

The original GPU4PySCF DF ωB97M-V48 process fails with a CUDA launch timeout
on warm call 2 (exit -6). One unchanged paired retry succeeds. Six displaced
completed calls remain in `superseded_populations`: total 130 processes,
one failed process, 518 completed calls; selected 512. No timing-based replacement.

Replay scalar gates/plots with Matplotlib 3.10.9:

```bash
PYTHONPATH=python:. python -m benchmarks.readme_defaults_summary \
  --campaign benchmarks/results/readme-rtx5090-20261111 \
  --output .artifacts/readme-rtx5090-pr3065-replay
```

This is not fresh full-vector subtraction. Full vectors, build logs, the
independent audit, failures and corrected absolute-path identity check remain
in `.artifacts/readme-rtx5090-default-20261110/` and
`n1:/data/jzzeng/readme-rtx5090-default-20261110-d35ca77a4`.
See the [decision note](../../../.agents/notes/implemented/numerics/2026-11-10-default-hf-orbital-gradient.md)
for guards, fallback boundaries and the historical launcher-check correction.
