# H100 cold DF seed-admission qualification

This source-bound result qualifies the narrow large CUDA DF Hückel seed
product preference in measured commit
`87bbc0b4f20f7696cfd22974bbe5b207e7d69c38`, tree
`99e47d472b89d3ce39c38d4aa3470cf7e82dc3b9`. It covers HF96 DF in three
rotating scalar/default/GPU4PySCF rounds, one HF48 DF numerical size guard,
and a separately rejected WB97M-V96 Direct incremental pilot. It does not
qualify the original full30 matrix, other hardware, or later native sources.

All 15 fresh processes / 60 complete energy-and-analytic-force calls pass the
original independent dense/CPU-spot reference gates. The same explicit orbital
and HF cc-pVTZ-JKFIT tables, geometries, grids and stopping recipes are retained.
No reference density is supplied to native execution.

| Study | Scalar cold median (s) | Candidate cold median (s) | GPU4PySCF cold median (s) | Median paired candidate/scalar cold | Median paired warm |
| --- | ---: | ---: | ---: | ---: | ---: |
| HF96 DF default, 3 rounds | 5.61596 | 5.20865 | 5.68354 | 0.926679 | 0.999193 |
| HF48 DF size guard, 1 round | 1.33067 | 1.34110 | 4.73783 | 1.007841 | 1.011686 |
| WB96 Direct incremental pilot, 1 round | 82.75157 | 146.66875 | 41.62063 | 1.772399 | 2.608771 |

The HF96 default reduces cold time by **7.332%** under the predeclared median
of three paired ratios. This differs from taking the ratio of displayed
process medians. All three cold candidate/scalar ratios are below one and all
three warm ratios pass the 1.05 guard. GPU4PySCF/default cold ratios are
1.762191, 1.093242 and 1.081556; the first 9.17863 s reference observation
remains included. This is no claim of a warm speedup. HF48 is a numerical
boundary guard, without a timing-improvement claim.

The WB incremental pilot passes independent and matched E/F gates but regresses
both phases, so subsequent rounds were omitted by the declared pilot rule and
the selector remains unpromoted. Its six cold delta builds establish execution;
`quartet_work_counters_valid` is false, so no quantified quartet saving is claimed.
Every unfavorable raw sample remains in the receipt container.

Cold means a fresh process's constructor, preparation, SCF, analytic forces and
host export **after GPU context initialization**. Python/process startup,
installation and AOT compilation are excluded. Verified compiled CuPy/CUDA
artifacts are retained in task-owned caches; this is not empty-cache compilation
latency. Warm repeats use each engine's own frozen cold density. The measured
device is H100 80 GB, MIG off, UUID
`GPU-4c6621ae-fe77-2603-be67-d476f2d4d1e4`; core SHA256 is
`4702ac35b5fada834aa15d0569a4f750d1dc756dbca731090ed5b5c6f49d64a9`.

`records.json.xz` is a valid JSON container of exact UTF8/base64 receipt bytes,
with individual SHA256 bindings. It retains all raw endpoints, plans, producers,
original independent references, reuse contracts, input tables, compiled-cache
manifest, platform/build receipts, host-test outcomes and the read-only integrity
review. It losslessly re-encodes the 99 original archive members and adds
provenance; the original transport archive identity is recorded separately.
The integrity review found no blocking production defect and retained a scoped
WARN because the broader goal is unfinished; same-family semantic review is
provisional. These are independent numerical-reference calculations, not
experimental ground truth or an intended-state certificate.

Run the source-bound offline reader, which performs no chemistry or compilation:

```bash
uv run --no-env-file --no-project --python 3.12 --with numpy==2.2.6 python \
  benchmarks/results/h100-cold-df-seed-20261105/replay.py
```

The reader authenticates every member, recomputes every independent/matched
observable gate and paired statistic, and checks against `summary.json.xz`.
It refuses changed scientific benchmark helpers. If those helpers have changed
on later main, pass `--repository /path/to/source-at-87bbc0b4f` with a checkout
of the measured source; it does not fetch code or run a model automatically.
The frozen analyzer is an opt-in historical evidence reader, not a required
source-spelling CI test.

The preliminary wider-selector **8.43%** result belongs to old source
`4370d6c4b…` and is not this changed-default result. Follow
[issue #3004](https://github.com/jinzhezenggroup/generativeqc/issues/3004) and
[PR #3007](https://github.com/jinzhezenggroup/generativeqc/pull/3007) for the
remaining cold work.
