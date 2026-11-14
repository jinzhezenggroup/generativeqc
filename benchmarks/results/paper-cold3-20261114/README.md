# Three independent application-cold processes for paper HF/DFT figures

Both new cohorts use production `efeaa3c7107a759e95c088d366e990d66c551c27`, native source identity
`20d337f7ffdb1c75eb89a3ba6b0c307518e21388a4d4fda7df6b54248b4337e7`.
Each GPU has 128 populations, 384 fresh formal processes and 1536 complete
energy-and-analytic-force calls. All timings and common-gate failures are retained.
These data do not rewrite the older H100/RTX cohorts or qualify an unmeasured later main.

| GPU | Native faster cold medians | Native faster warm medians | Adjacent cold-median decreases | Process gate counts |
| --- | --- | --- | --- | --- |
| H100 | 63/64 | 62/64 | 0 | {'qualified': 348, 'failed': 36} |
| RTX 5090 | 60/64 | 64/64 | 0 | {'qualified': 348, 'failed': 36} |

The separately retained historical one-cold-call cohorts had native-faster
counts 62/64 cold and 61/64 warm on H100, and 64/64 for both phases on RTX 5090.
Their original source revisions and compressed-sample SHA256 values are retained
in `summary.json`. These are descriptive comparisons of separate cohorts, not
paired before/after measurements or proof of a repetition-only improvement.

The new RTX 5090 cold exceptions are all at 96 atoms: Direct HF, DF HF,
Direct PBE0 and DF r2SCAN-3c. The reference/native median ratios are respectively
0.932918, 0.954593, 0.996183 and 0.964741. The first, second and fourth are
slower in all three new pairs; Direct PBE0 is near parity (one pair faster,
two slower). The original cold-all-faster statement must therefore be revised
if these data are adopted. These are retained results, not excluded points.
Their SCF/Fock/veff and preliminary-guess counts match the historical cases.
The current CPU affinity is GPU-local NUMA1, checked through host sysfs;
the historical cohort also declared local affinity on a different host/GPU.
No same-device old/new-source A/B comparison was acquired, so these differences
cannot be attributed to repetition alone or to a specific runtime component.
Full-precision old/new timings, counts and every paired ratio are in `summary.json`.

Cold includes constructor, preparation, SCF, analytic forces and host export.
Imports/CUDA-context startup and a separate 128-process persistent-compilation-cache
prewarming pass are outside timing, as in the paper. Caches are retained.
Each formal process owns one cold call and three warm calls starting from its own
immutable cold density. Cold lines/bars are the median/min/max of three processes.
Warm lines/bars are the median/min/max of their three within-process medians;
all nine warm values are retained. Ranges are descriptive, not confidence intervals.
Engine-first order is balanced 32/32 in each round; the middle traversal reverses.
There is no timing/accuracy filtering, smoothing, monotonic fitting or sample replacement.

The previously circled H100 Direct-HF 12-to-48 and DF-B3LYP 3-to-12 decreases are
absent from the new median curves. An individual middle-round Direct-HF 48-to-96
decrease remains in the raw observations. New source/physical-device provenance
differs from the historical paper cohort, so this does not establish the cause
of the old decreases or isolate aggregation as the cause of every old/new change.

Common independent gates remain energy 1e-5 Eh/atom, maximum-force 1e-4 Eh/Bohr,
and RMS-force 3e-5 Eh/Bohr. Reuse of the unchanged dense Direct reference archive
replays 40 original references, 28 grid-stability pairs and 8 exact-quadrature
CPU/GPU spots at quarter gates. Every formal full force vector is rechecked.
The legacy failed keys are both engines DF-HF at all four sizes (DF approximation
against a dense Direct target), plus M06-2X Direct 3 and DF 3 (grid/force errors).
Failed timings remain plotted. No overall numerical PASS or default promotion is claimed.

## Replay

From the repository root with NumPy and Matplotlib installed:

```sh
python benchmarks/results/paper-cold3-20261114/reproduce.py --verify-only
python benchmarks/results/paper-cold3-20261114/reproduce.py --output .artifacts/paper-cold3-replay
```

The second command writes both 4x4 figures as PDF/SVG/PNG and regenerated summaries.
Figure generation used Python 3.12.13, NumPy 2.5.4 and Matplotlib 3.10.9.
The existing reference file is
`benchmarks/results/readme-refresh-20261026/oracles.json.xz`, SHA256
`d61a3e8ba0a8f8a8c08ee4c4b30c8e3e397e67863b8d75e7ba974e90cf0f85a4`.
No GPU/reference installation or expiring external archive is needed for replay.

`h100.json.xz` and `rtx5090.json.xz` are declared numeric projections, not raw
process-JSON reconstructions. Every scientific sample field, full-precision
energy/force/timing/work/settings value, scientific input and original raw JSON
SHA256 is retained. Private acquisition paths are omitted; shared environments
retain package/toolchain metadata. Original archive/manifest/aggregate identities
and exact-analysis digests remain explicit. Replay recomputes full-vector gates
and all plotted statistics rather than trusting stored pass flags.

## Source and execution reconstruction

`measured-h100.patch` and `measured-rtx5090.patch` reconstruct the actual recorder
and acquisition overlay for each GPU. They were tested with `git apply` against
the exported base and reproduce every measured recorder byte, including mixed
CRLF/LF line endings. The baseline archive was produced by Git 2.51.0.windows.1
with `core.autocrlf=true`; it is not an LF-only Git export. Authenticate the actual
reconstructed archive/native-source identities; do not override an identity or
normalize a mismatched cache/source to obtain a hit. Source archive SHA256 is
`e2b4ddbb22ec2aa81dc111471537a869690f44648899e0429a0581f5d9fa411e`.
The overlaid benchmark files do not change scientific equations/default settings.
`build-contracts.json` retains actual CMake method/AOT/architecture flags and
native image/AOT audit identities; selected per-process packages are authoritative.
The broad last-occurrence distribution catalog is diagnostic only.

New acquisition uses `python -m benchmarks.readme_cold_repeats --help` with
site-specific JSON paths/interpreters, verified persistent sccache 0.16.0+ and
actual qz/Slurm allocation. The finite parent deadline is 46800s; whole-process
bounds are 600s H100 and 1800s RTX 5090. Set source/input paths, selected native and
reference environments and actual allocation explicitly; qz hostname variables
alone are not independent platform authentication. Do not reuse old job IDs.

`platform-receipts.json` retains actual terminal qz/Slurm observations. The first
RTX job 17383 was cancelled during prewarm to correct its insufficient 600s
whole-process bound; 54 complete prewarm processes/108 returned E+F calls plus
one interrupted process remain separate, with zero formal samples. Unreturned
work is unknown. Job 17384 performed the full new cohort with authenticated build
reuse and an AOT audit. No cancelled-attempt timings were substituted.
The initial qz v1 job never reached allocation; atomic assets were staged but
no second GPU Job was submitted once CQ became usable. Original complete and
cancelled archives remain in ignored task storage and their original remote
roots, authenticated by the retained hashes. They are optional acquisition-debug
material; the numeric replay does not depend on private paths or external hosting.

This is descriptive timing evidence for figure review. Figure/protocol
acceptance remains pending; no production-default promotion is claimed.
