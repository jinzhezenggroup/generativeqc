# GPU DF-CCSD(T)/CBS: QM7 C1–C7 complete energy and analytic forces

This campaign measures the production implementation at master
`05844f847b447e426523d4d9a559991ae8ac640b`, including the bounded RHF improvements
from #2399. It does not substitute earlier RHF-only or single-basis CCSD(T)
measurements for the complete CBS endpoint.

See [timings and observed work](timings.md). This superseded homepage campaign's
original figure, receipts, driver and manifest are recoverable through the
published Git revision in [snapshot.json](snapshot.json). All eight original
files were restored and compared byte for byte before retiring their payloads
from the current tree. The snapshot preserves failures and unchecked points;
none are reclassified as completed or qualified.

The later [full DF-HF/RI-JK campaign](../readme-cbs-full-20261014/README.md)
uses a different reference/fitting model. This older baseline is retained,
not silently overwritten or used as an identical-Hamiltonian speedup claim.

## Hardware and build

The primary series uses **node1, RTX 5090 (600 W), driver 580.95.05**, with
dual AMD EPYC 9354 CPUs. Each Slurm job owns one GPU and requests four logical CPUs;
numerical thread pools use one thread. Other jobs can run on other GPUs of the
same host. This campaign requests no GPU clock locking. These are individual endpoint observations,
not an isolated-host scaling law or a statistical latency guarantee.

All native observations use the same Release CUDA **12.9.1 / sm_120** library:
`f814ddfb406b2f97fe1f7fd178a71d9f19d4e47219f13008ac64688319f871a4`.
The build uses eight AOT shards, direct device linking, fast compile disabled,
and verified sccache 0.16.0. The optional CPU dense-linear-algebra provider
resolves to the scalar fallback; this is not an OpenBLAS-enabled build. The
compiled production-source inventory matches the clean master checkout:
`5e023d02920323e93c12e0063222a2a0dea0d58308eff6f2df4d501d3f6afc50`.

The native DSO is preloaded before Python to pin its CUDA 12.9 runtime
dependencies, avoiding an older system cuSOLVER discovered by the Python
loader. Thus API cold **excludes process startup/native DSO loading**; it still
includes calculator creation and the first complete calculation. Initial
loader failures and preliminary C1–C3 checks on node4 are retained separately
in the recoverable `supplement.json.xz`, not pooled into the primary
series. C1–C3 were rerun on node1 to standardize the host/driver/power setting,
not to select the fastest sample.

## Scientific and timing contract

- MP2 correlation extrapolation: spherical **cc-pVDZ/cc-pVTZ**, with matching
  **cc-pVDZ-RI/cc-pVTZ-RI** correlation fitting bases, cardinal exponent **3**.
- Small-basis correction: **DF-CCSD(T) − DF-MP2 in cc-pVDZ/cc-pVDZ-RI**.
- Reference: **conventional exact RHF**, with exponential DZ/TZ HF extrapolation
  (`alpha = 1.63`). “DF” here means correlation-only DF, **not DF-HF throughout**.
  The production RHF initial-guess/screening policy remains unchanged; its final
  reference and response use exact J/K.
- All-electron, no frozen core, closed-shell singlet, FP64. Forces are analytic,
  include fitting-center/metric derivatives, and use the same fixed CBS weights
  as energy. No production PySCF SCF guess or finite-difference force is used.
- The public grouped endpoint executes **two basis groups**: DZ owns CCSD(T),
  its MP2 companion and HF; TZ owns MP2 and HF. Same-basis reference/DF-source
  reuse is retained. Different bases do not share MO coefficients or factors.
- The outer `perf_counter` interval includes calculator construction, both
  complete native groups, SCF, CC/triples, response/Lambda and analytic forces,
  ending only when the composite E+F and host force array have returned.
  Group cleanup inside `singlepoint` is included. Component times overlap
  within each group and must **not** be added to reconstruct endpoint time.
- Dataset loading, imports, basis-record conversion, provenance, JSON writing
  and independent reference calculations are outside the native timer. A fresh
  process/calculator is used for each molecule; compiler caches are retained.
  This is **API cold**, not an empty compiler-cache or installation benchmark.
  The figure uses one cold observation, not a five-repeat median or a claim
  about fixed-density warm performance. Optional further samples are complete
  replays; CBS does not retain a prepared SCF plan across calls.
- The declared correlation memory budget is **64 GiB**. This is an admission
  budget, not a statement that a 32-GiB RTX 5090 has 64 GiB of VRAM, nor a
  measured peak. Returned resource/work diagnostics remain in the receipts.

## C5 performance diagnosis

C5 completes in **2660.194 s**: **178.855 s** in the DZ group and
**2481.338 s (93.3%)** in the TZ MP2/HF group. The latter increases from
243.979 s for C4 although orbital-response iterations increase only from
17 to 18, without restarts. This is not a 40-minute small-basis triples solve.

A separate same-library, traced C5 replay on node4 identifies an exact-ERI
residency cliff: the TZ source needs **14.875 GiB**, exceeding the default
**8 GiB** ceiling, so admission refuses the complete cache. C4's corresponding
canonical inventory is **6.633 GiB**. The replay reaches the exact orbital and
nuclear response after about 91 s inside the TZ native group; correlation DF
does not remove this conventional-reference force work. The first two completed
exact-response Arnoldi actions take **72.850 s** and **72.923 s** each. The
diagnostic is separate from the uninstrumented primary timing series.

See the [diagnosis and proposed follow-up](../../../.agents/notes/proposed/2026-10-13-cbs-exact-response-cliff.md)
for source ownership, calculated C6/C7 inventories, and scientific constraints.
No production algorithm or memory-cap change is included in this benchmark.

## Fixed molecules

Use the first QM7 molecule with formula C*n*H*(2n+2)* at each carbon count.
This preserves the previously investigated **C7 index 5604**. Formula alone
does not establish an isomer; this is not a claim to benchmark every isomer or
an independently constructed n-alkane series.

| Carbon count | QM7 zero-based index | Formula | Atoms |
| ---: | ---: | --- | ---: |
| 1 | 0 | CH4 | 5 |
| 2 | 1 | C2H6 | 8 |
| 3 | 8 | C3H8 | 11 |
| 4 | 48 | C4H10 | 14 |
| 5 | 183 | C5H12 | 17 |
| 6 | 293 | C6H14 | 20 |
| 7 | 5604 | C7H16 | 23 |

QM7 SHA-256:
`fd6c26c1086887310518e6a6da778253bc5d3ceb8166007a679a740fff9ee6ea`.
The original float32 Bohr coordinates are promoted to float64 without geometry
optimization, recentering or unit conversion. Each receipt keeps the coordinates,
orbital/auxiliary identities, AO/auxiliary counts and electron count. PySCF is
used only for explicit benchmark basis snapshots and the independent oracle,
not as a production dependency. The runner does not download the dataset.

## Accuracy and incomplete outcomes

Native and independent reference processes run separately: a missing or slow
reference never prevents the native endpoint from running. Filled markers mean
that **all constituents and the composite** pass independently reconverged
PySCF energy (**5e-8 Eh**) and directional-force (**5e-6 Eh/Bohr**) gates.
One seed-709 normalized direction is checked with central finite differences
at **1e-4 and 3e-5 Bohr**, including displacement of fitting centers. Native
convergence, CUDA execution, finite host force arrays and translational force
sum (**1e-6 Eh/Bohr**) are also required for every constituent and composite.

These are **directional checks**, not an independent full-vector force check
and not an estimate of error against the true CCSD(T) basis-set limit. No
reference runtime is plotted: CPU reconverged finite differences are not the
same analytic E+F workload. No engine speedup is inferred.

Open markers/‡ mean completed native E+F with **null**, not passing or zero,
independent accuracy gates. Missing either prescribed derivative step leaves
accuracy unchecked. Numerically failed or incomplete E+F has **no plotted
completion time**. Whole-process deadlines include setup; they are neither
per-call timings nor speedup denominators. Completed earlier calls survive a
later failure, and incomplete replay sets do not acquire a median. Original
outcomes, native errors, last stages, arrays and work diagnostics are retained.

## Reproduce

Build a Release CUDA sm_120 library at the recorded source revision using a
verified compiler cache. Supply a local checksum-matching QM7 file and run
each native point through Slurm on an allowed RTX 5090 node:

```bash
export GENERATIVEQC_LIBRARY=/absolute/path/to/libgenerativeqc.so
export LD_LIBRARY_PATH=/group/software/cuda-12.9.1/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export LD_PRELOAD="$GENERATIVEQC_LIBRARY"
export LD_BIND_NOW=1
export GENERATIVEQC_PROFILE=off
export QM7_DATASET=/absolute/path/to/qm7.mat
export README_CBS_SOURCE_REVISION=05844f847b447e426523d4d9a559991ae8ac640b
export README_BENCHMARK_PYTHON=/absolute/path/to/python
export README_BENCHMARK_OUTPUT=.artifacts/readme-cbs
export README_BENCHMARK_SIZES=7
export README_BENCHMARK_POINT_TIMEOUT=10800
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --time=03:05:00 \
  bash benchmarks/run_readme_cbs_benchmarks.sh native
```

Do not override Slurm's `CUDA_VISIBLE_DEVICES`. Select C1–C7 individually or
provide a space-separated `README_BENCHMARK_SIZES` list with a finite allocation
long enough for its point limits. Run the independent CPU oracle separately
(`reference` instead of `native`); it never reads native results. All numerical
thread pools are fixed to one thread by the runner. Review the retained host,
GPU UUID, driver, library digest, source and environment for each observation.

Restore the original campaign before reproducing its figure. A shallow clone
must first fetch the revision in `snapshot.json`; recovery never fetches implicitly:

```bash
python tools/restore_retained_evidence.py \
  --manifest benchmarks/results/readme-cbs-20261013/snapshot.json \
  --all --output .artifacts/readme-cbs-history
PYTHONPATH=.:python python -m tools.render_readme_cbs \
  --summary .artifacts/readme-cbs-history/benchmarks/results/readme-cbs-20261013/summary.json.xz \
  --output .artifacts/readme-cbs-replay
```

Raw execution writes only to ignored/local artifacts, never directly into the
reviewed results directory. The compact archive retains native/reference JSON
receipts and outcome hashes; build logs, libraries and raw traces stay local.
The archived `measured-driver.py.xz` retains the actual native runner before
full-DF options were added. Early reference receipts used Python 3.13's
`hashlib.file_digest`; applying the recovered `early-reference-driver.patch`
to that snapshot reconstructs the early runner. The current default exact-HF
protocol remains unchanged; `README_CBS_HF_REFERENCE=df` explicitly selects
the new, separately reported model.
