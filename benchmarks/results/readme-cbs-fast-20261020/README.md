# Standard and Fast DF-CCSD(T)/CBS: QM7 C1–C7

The homepage curves are fresh measurements of both policies with the same
native binary and physical GPU within each pair. See [complete timings,
observable differences and semantic work](timings.md), and the
[Fast convergence contract](../../../docs/user/ccsdt_cbs.md#fast-convergence).
Fast lowers stopping requirements for SCF, CCSD, Lambda and orbital response,
and defaults to bounded FP32 auxiliary DF-CCSD preconvergence. Final physical
checks, Lambda, triples and analytic response remain FP64. Standard remains
the default policy. Actual FP32 admission and work counters are retained.

## Scientific model and acceptance

The pinned QM7 representatives use zero-based indices **0, 1, 8, 48, 183, 293,
5604**. Dataset SHA-256:
`fd6c26c1086887310518e6a6da778253bc5d3ceb8166007a679a740fff9ee6ea`.
Original float32 Bohr coordinates are preserved without recentering or
optimization. The receipts retain coordinates, basis hashes and AO/auxiliary
counts.

MP2 correlation is extrapolated with spherical cc-pVDZ/cc-pVTZ and exponent 3;
the DF-CCSD(T)−DF-MP2 correction uses cc-pVDZ. HF extrapolation is exponential
with alpha 1.63. The all-electron closed-shell model uses **DF-HF/RI-JK in the
reference, response and gradients**, with cc-pVXZ-JKFIT shared for JK and
correlation within each basis group. The DZ group owns CCSD(T)/MP2/HF and TZ
owns MP2/HF. The explicit total numerical-capacity budget is **32 GiB**, not a
measurement of peak GPU memory. FP32 workspace/provider refusal retains the
bounded FP64 path and is reported explicitly rather than counted as mixed work.

Every plotted Fast point passes energy, maximum force and RMS force differences
against its same-source Standard control, for **all five unique constituents
and the composite**. Composite allowances are `1e-5 Eh`, `1e-4 Eh/Bohr` maximum
and `3e-5 Eh/Bohr` RMS; each constituent allowance is divided by the combined
coefficient L1 sum (`4.196866214617815`). These differences qualify the combined stopping/arithmetic
change on these inputs. Standard is not an independent scientific oracle.

Both policies also pass the independent C2 constituent/composite energies and
one normalized force projection at `1e-4` and `3e-5 Bohr` finite-difference
steps. That PySCF oracle is reused unchanged from the
[2026-10-14 campaign](../readme-cbs-full-20261014/README.md), after exact equality
of every scientific protocol field was checked. Its full receipt and provenance
are embedded in each series. Native times are newly measured; historical native
times are not reused. The other six molecules remain independently unchecked.
A directional check is not a full-vector independent force oracle or a
basis-limit accuracy certificate. Independent conventional/DF small-system
qualification of the FP64 convergence-only ablation is retained
[separately](../cbs-convergence-fast-20261020/README.md). Final mixed DF water
qualification is retained here in `df-water-paired.json.xz`: exact-RHF and RI-JK
models, all constituent energies and force components, independent reconverged
PySCF energies and central differences (`3e-5 Bohr`; planar symmetry supplies y).
Each policy/model retains one cold call and three fresh-owner replays. Oracle
errors refer to the final replay; its embedded driver reproduces the comparison.
These STO-3G/def2-SVP water pairs exercise nonzero triples but are not systematic
physical CBS basis sequences.

## Timing protocol and limitations

Each value is **one API-cold fresh-calculator complete E+F call**, from calculator
construction through returned composite energy and host forces: both groups,
SCF, CCSD, triples, Lambda, orbital/nuclear response and in-call cleanup.
Python startup, preloaded native/CUDA DSOs, dataset/basis snapshots, receipt
serialization and independent references are outside the timer. There is no
hidden Standard calculation inside Fast and no retained converged owner across
measurements. Companion timings overlap and must not be summed; the table
retains disjoint basis-group times, CCSD and response iterations. The current
companion ABI does not expose DZ SCF iterations, so these remain unavailable.

C7 runs on n4 and C6 through C1 run sequentially on n5, both RTX 5090 GPUs.
Slurm assigns one GPU and four CPUs per allocation; numerical thread pools use
one thread. Odd carbon counts run Standard then Fast; even counts reverse that
order. GPU identities, driver versions, power limits, host names and Slurm job
IDs are in the original receipts. C7 uses another host/driver, so cross-size
scaling includes that variation. Clocks are not locked. Each policy has a
3600-second process limit. The n4 pair allocation is limited to 02:05:00 and
the n5 six-pair allocation to 02:30:00. No timed-out call is plotted as completed.

These single observations retain startup/host variability. All successful
observations, including any slower Fast cases, are kept. Small cold calls are
especially sensitive to variability. The series does not
establish a statistical speedup or guarantee that Fast is always faster.
The publication's performance decision is therefore `inconclusive`, with all
measured values and numerical/control outcomes retained.

The measured base is `eac2de2b33544b104bb318e53aa02aecbea5009f` plus the retained
`qualified-source.patch.xz`. The source-matched native library SHA-256 is
`0d6ec6ff673a6f3c1d1aefe8f561fa6ccf47f2e3bc803f449c200bbd44812743`.
It was built locally with GCC 11.4.0, CUDA 12.9.1, Release/sm120 and verified
ccache 4.5.1, then deployed unchanged. Runtime toolchain discovery in a remote
receipt describes that host and must not be mistaken for the build compiler.
`source-manifest.json` records measured source and driver identities. The only post-measurement source difference is a corrected `PrecisionPolicy`
public docstring; executable ASTs match, and its final hash is recorded separately.
The retained source patch reproduces the exact measured version. Exact
launch/environment/preload scripts are retained as reproduction records.

## Reproduce and replay

Set `GENERATIVEQC_LIBRARY`, `QM7_DATASET` and `README_CBS_SOURCE_REVISION` to the
matching build, pinned dataset and actual source. Keep the same native/CUDA
DSO preload convention for both policies. A portable compatible loader was
used on n4/n5; its absolute deployment paths are preserved in `python-preload.sh`.
For one policy and molecule:

```bash
export README_CBS_HF_REFERENCE=df README_CBS_BUDGET_MIB=32768
export README_BENCHMARK_SAMPLES=1 README_BENCHMARK_SIZES=7
export README_CBS_POLICY=fast README_BENCHMARK_POINT_TIMEOUT=3600
export README_BENCHMARK_OUTPUT=.artifacts/readme-cbs-fast
export GENERATIVEQC_PROFILE=off
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --time=01:05:00 \
  bash benchmarks/run_readme_cbs_benchmarks.sh native
```

Repeat Standard on the same allocated GPU; use `readme-pair.sh` to reproduce
the original alternating policy order after adapting deployment paths. Preserve
Slurm's `CUDA_VISIBLE_DEVICES`; never use maintenance node n3. Source and library
hashes, scientific protocols, benchmark driver and physical GPU must match
before the renderer accepts a Fast control.

Verify and render the frozen dataset without CUDA:

```bash
PYTHONPATH=.:python python benchmarks/results/readme-cbs-fast-20261020/verify.py
PYTHONPATH=.:python python -m tools.render_readme_cbs \
  --summary benchmarks/results/readme-cbs-fast-20261020/summary.json.xz \
  --output .artifacts/readme-cbs-fast-replay --readme-only
```

`summary.json.xz` losslessly retains all 14 native receipts and the reused C2
oracle. Its stored gates are independently recomputed by `verify.py` before
comparing the generated table. Publication checksums bind all retained files.
The homepage SVG is reproducible from this summary; a duplicate plot is not
stored in this bundle. Full working runs and logs stay in ignored artifacts.
The superseded Standard-only campaign remains recoverable through its verified
published-Git snapshot, including its source patch and negative evidence.
