# README native refresh: combined PR #2531 + #2532

This publication replaces the native HF and seven DFT series in both README
panels. GPU4PySCF is **not executed again**: its original timing/work populations,
failures, and independent geometry oracles remain byte-identical in the linked
historical reductions. DF-CCSD(T)/CBS is unchanged.

## Frozen source and history

The user requested measurements assuming these two open PRs were already merged:

| Input | Exact revision |
| --- | --- |
| PR #2531: compensated Direct-J and strict-FP64 XC totals | `ba0dadbdcdef7af18715b72a2366595719df4bde` |
| PR #2532: default resident final validation for eligible pure-J KS | `27d3b8558b7db2ae4b7d4051ae9122522379a809` |
| Combined measurement commit | `1593d3b2fea9ed3e9232efc49eedede1beb1787b` |
| Combined source tree | `f158f1c1b69700df6d74b0ebe42c68d370be09f3` |

The merge includes master `890c425b0`, including #2524's shared resource sizing.
It does not assume that GitHub has merged either prerequisite. Later
documentation-only commits do not change the measured source or binary identity.
If either prerequisite changes, these results still describe the exact revisions
above, not an automatically updated PR head or future master.

During measurement, #2531 advanced to `01ed327f4` with three test-only changes,
and #2532 advanced to `1be47f590` with publication/test-only changes.
`source-input-audit.json` records the comparison of the new combined tree:
no native build input or benchmark consumer changes. The frozen source identity
above is deliberately retained instead of relabeling the existing observations.
The #2532 correction retains its original cohort's failed nonregression gate as
an acceptance hold. This fresh figure campaign does not turn that historical
promotion gate into a pass or claim that every native population is faster.

The exact combined source can be reconstructed without merging into a working
checkout: `git merge-tree --write-tree 27d3b8558 ba0dadbdc` produces the source
tree above, which `git archive` can export. Fetch the two exact prerequisite
commits first if they are not already available locally.

The default-performance audit also identified #2398/#2399 (hybrid source reuse
and RHF reference builds), #2494/#2495 (exact-density force-response reuse),
#2511 (FP64 force-power lowering), #2521 (guarded VV10 warp packets), #2522
(density-work-jet point capacity), and #2523 (DF hybrid warm-factor replay).
#2531 changes accumulation used beyond r²SCAN-3c, so this campaign remeasures
all native panel methods instead of mixing the interrupted earlier campaign's
`4e28477e9` observations with the combined-default build. Opt-in SCF CSE,
experimental selectors, and Fast/FP32 modes are not enabled.

## Endpoint protocol

- One Slurm-assigned RTX 5090 per finite `srun` allocation, on n1, n4 or n5.
  Independent method/DF-mode jobs run concurrently on distinct assigned GPUs.
  Node n3 is not used; the scheduler's `CUDA_VISIBLE_DEVICES` is never changed.
- Release/sm_120, CUDA 12.9.1, fast compilation disabled. The source-matched
  core and all 24 stationary-force AOT plugins are built through verified
  ccache 4.5.1, reusing existing caches. `build-settings.json` retains source,
  toolchain, compiler-cache statistics, binary and plugin identities. Full
  build logs and generated artifacts remain in ignored campaign storage.
  Runtime cache-miss compilation also verifies available sccache 0.16.0 on n1;
  other nodes retain the ccache fallback. Shared cache statistics may include
  concurrent compilations and are not treated as scientific work counters.
- Original water clusters at 3/12/48/96 atoms. Spherical def2-SVP is used except
  r²SCAN-3c's defining def2-mTZVPP basis. Both DF engines retain cc-pVDZ-JKFIT.
  The moving, unpruned `(48, 16, 32)` Becke grid, FP64 equation, public
  initialization, convergence and independent accuracy thresholds are unchanged.
- Each measurement population requests one cold call, five fixed-density warm
  calls, one changed-geometry call, and five changed-geometry warm calls. Cold
  includes Calculator construction, preparation and the first complete
  host-returned energy **and analytic forces**. Warm plots show the median and
  min–max of all five samples. Actual SCF iterations/Fock-build counts remain
  alongside times; changed iteration counts are not normalized away.
- Separate 3/12-atom priming processes precede fresh measurement processes and
  Calculators in each method/mode job. No converged density, checkpoint, or
  native allocation crosses a process boundary. Priming costs and receipts are
  retained separately, never subtracted from cold. Larger shapes retain caches
  but are not claimed to have the same cache readiness as historical references.
- The native producer uses only retained independent energy/force arrays for
  validation, never a reference engine or reference density in production.
  Every checked complete call must satisfy `|ΔE| ≤ 1e-8 Eh` and
  `max |ΔF| ≤ 1e-7 Eh/bohr`. The reducer recomputes these errors from raw arrays.
- ωB97M-V DF/48 remains explicitly unchecked: no matching independent oracle
  is retained. DF/96 is not retried because prior work obtained no complete
  native E+F population and these prerequisites do not qualify that case.
  Neither point acquires an invented accuracy pass or speedup.

Earlier paused measurements, incomplete setup attempts and their runtime
packages are retained separately. They are not qualification samples for this
combined source. A successful core build alone is insufficient: the stationary
plugins and their manifests must be source-matched too, or an energy-only result
can fail force completion. This campaign deploys the complete package unchanged
to every participating node.

## Results and evidence replay

All 63 requested native populations complete, retaining 756 timed E+F calls:
744 pass the independent accuracy gates, while the 12 ωB97M-V DF/48 calls
remain explicitly unchecked. Maximum checked errors are `2.2555e-10 Eh` and
`1.6854e-10 Eh/bohr`. The 384 separate priming calls are retained but not plotted.

These are fresh absolute endpoint timings, not a balanced baseline/candidate
promotion experiment. Changes in SCF work remain visible; no individual kernel
speedup, identical cache readiness, or universal population speedup is claimed.

![Direct cold and warm E+F](dft-cold-warm.svg)

![DF cold and warm E+F](df-cold-warm.svg)

See the [direct timing/work table](timings.md) and
[DF timing/work table](df-timings.md). `native-summary.json.xz` retains native
scalar observations, residuals, acceptance gates, semantic work counts,
source/binary/raw-record hashes, Slurm provenance and separate priming receipts.
`job-receipts.json.xz` records all 16 completed finite allocations, assigned GPU
visibility/identity and original log hashes.
Raw force arrays, detailed SCF histories, logs and build products stay in ignored
storage. Only verbose per-iteration histories are omitted from the publication;
final convergence diagnostics remain. Offline replay verifies retained scalar
gates and identities; it does not pretend to reconstruct omitted native vectors.

`manifest.json` binds all publication files and the four unchanged historical
reductions. Four superseded SVGs are archived losslessly as `.svg.xz` to remain
within the 64 MiB evidence cap. Their historical manifests retain original
expanded hashes and update only storage/presentation/replay receipts. No timing,
independent array, source identity or acceptance gate is changed or relabeled.
Reduction rejects raw observations whose native source or binary differs from
the independently retained build settings, even if their own hashes agree.

```bash
PYTHONPATH=.:python python benchmarks/results/readme-refresh-20261015/refresh.py
PYTHONPATH=.:python python benchmarks/results/readme-refresh-20261015/refresh.py \
  --output .artifacts/readme-refresh-replay
```

Replay requires Matplotlib, Scour and defusedxml, as in the previous publication;
no GPU or reference calculation is needed. To reduce retained raw observations
again, use `reduce.py --raw <campaign-directory>`. `restore_references.py`
reconstructs accuracy-only reference receipts from stored arrays, without
executing GPU4PySCF. `reproduce.sh <method> <none|cuda>` runs only the native
engine inside an existing finite Slurm allocation with `README_REFRESH_ROOT`
pointing to a frozen `source/` and complete `runtime/` package.
