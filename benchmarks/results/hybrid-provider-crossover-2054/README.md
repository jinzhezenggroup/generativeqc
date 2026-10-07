# #2054 hybrid provider crossover pilot

This directory owns a benchmark protocol, not a crossover conclusion. The
source-matched runner is [`tools/benchmark_hybrid_provider_crossover.py`](../../../tools/benchmark_hybrid_provider_crossover.py).
The current public CUDA KS contract supports exact Direct J/K and full DF-JK
with the existing occupied exchange reuse path. It does not expose DF-J with
exact K. The runner records that arm as `UNSUPPORTED` and never substitutes a
different approximation.

## Frozen question

- PBE0 RKS, def2-SVP spherical, FP64, `GridSpec()` (48 radial × 16 polar × 32
  azimuth points per atom), screening `1e-12`, energy `1e-12`, density `1e-10`,
  180 maximum SCF iterations, E+force including host return.
- Cases: three-atom smoke; 48-atom nested water; non-water formaldehyde holdout.
  The 96-atom case is only attempted after resource and time review. Nested
  water geometry comes from `benchmarks/readme_hf_scaling.py`; formaldehyde
  comes from `benchmarks/dft_force_matrix.py`.
- DF uses explicit def2-SVP auxiliary basis and a 1 GiB DF device workspace
  hint. These are recorded as the DF approximation identity, separately from
  the common orbital problem. Direct has no auxiliary basis in its provider.
- Cold time includes `prepare_batch` and first complete `batch.execute`;
  warm, changed-geometry, and moved-warm each include complete E+force and
  host-return publication. The ABBA process order is Direct, DF-JK, DF-JK,
  Direct, with one independently recorded unsupported-arm disposition. Every
  run is a separate process; retries receive new output directories.
- Native convergence requires finite E and every force component, the CUDA
  backend, the live provider proof, and physical residual RMS at most `1e-8`.
  Independent PySCF PBE0 uses the exported frozen quadrature and grid response
  with the same orbital/auxiliary basis and requires convergence, energy error
  at most `1e-8` Hartree, force max absolute error at most `3e-7` Hartree/Bohr.
  `summarize` rejects absent or mismatched records. No tolerance is relaxed
  after seeing a result.

## Installed H200 pilot disposition

Frozen source `2952481fccec5efcf3f993605b23ccefe01e640a`, CMake-installed
sm_90 portable core SHA-256
`0f8fb7fc05befa133e1dbd44b9d725cfe6789adeb0db0e75f8a70f02ada088b2`,
CUDA 12.9.86, driver 595.58.03 and PySCF 2.14.0 are bound in the
[evidence manifest](evidence/manifest.json). Both CXX and CUDA commands used
task-owned ccache 4.5.1; pre/post statistics show 482 hits in 974 requests
for the installed build. The native Python interface came from the same clean
checkout, with `GENERATIVEQC_LIBRARY` selecting the installed prefix.

| Case | Exact Direct J/K | Full DF-JK, occupied reuse | Disposition |
| --- | --- | --- | --- |
| 3-atom water | two complete ABBA E+force runs passed independent PySCF | two complete runs passed independent PySCF | small pilot accepted; no crossover claim |
| 48-atom water, 384 AO | both cold SCFs converged, but native derivative admission failed and forces are `null` | both four-phase E+force runs completed; independent PySCF maximum errors `6.14e-12` Eh / `1.97e-11` Eh/Bohr | both pair summaries `INCOMPLETE`; no Direct-versus-DF endpoint crossover |
| formaldehyde, 38 AO | both four-phase runs passed independent PySCF | both four-phase runs passed independent PySCF; occupied reuse observed | both holdout pair summaries `PILOT_ACCEPTED`, each for its labeled approximation |

The third DF-J/exact-K arm is `UNSUPPORTED` by the current public KS contract.
The 48-atom Direct error is
`prepared native integral derivatives are unavailable within the admitted
budget; enlarged stationary CUDA domains cannot use AO-task fallback`.
The ordinary public PBE0 force route fixes the stationary allowance at
512 MiB device / 256 MiB host; no public `KsOptions` setting replaces it.
This is an H200 sm_90 portable-build observation, not a general claim about
all Direct devices. No 96-atom campaign ran after the failed 48-atom complete
endpoint, and no production provider or AUTO selector changed.

The [offline verifier](verify_evidence.py) authenticates every retained JSON
member, recomputes all four pair summaries and enforces those exact acceptance
limits without a GPU:

```bash
PYTHONPATH=python:. python benchmarks/results/hybrid-provider-crossover-2054/verify_evidence.py
```

The platform sampled 53 one-minute Pod observations with maximum GPU memory
usage rate 5.51%. This is coarse whole-Pod usage, **not** measured process peak.
DF metric records report a planned allocation peak, not achieved peak. Neither
number is used to infer a crossover or relax the Direct force gate.

## Reproduction

Run only on a source-matched installed build in a finite Slurm or Inspire GPU
Job. An Inspire submission sets `GENERATIVEQC_2054_FINITE_JOB` to its real Job
name; verify that identity against platform status and events. Set
`PYTHONPATH=python:.` and `GENERATIVEQC_LIBRARY` to the loaded library path.
The output below is ignored working evidence; review and compact small records
before adding any to Git.

For the qz H200 pilot, `build-and-smoke.sh` freezes the actual sm_90 portable
build, requires an independently installed task-owned `ccache` binary and
records the real CMake launchers, pre/post cache statistics, binary hash and
three-atom device attempts. It never starts an uncached full build. Its
task-owned wheel/cache overlay lives under `.artifacts/` and is not a
production dependency.
The first CUDA 12.8 attempt stopped at the repository's CUDA >=12.9 configure
gate; its raw log remains separate from the CUDA 12.9 build directory.
The first CUDA 12.9 source-matched core build completed, but its E+force
attempts failed closed because the public PBE0/def2-SVP force route requires
the packaged `pbe0_rks_spd` stationary artifact. That build had omitted AOT
shell bundles and had not built the separate stationary manifest targets.
`complete-aot-and-smoke.sh` pins the exact unchanged core SHA and frozen source,
builds both PBE0 RKS stationary domains with the task-owned cache, and writes
new raw attempts without overwriting that negative result.
Those AOT assets then allowed the full DF-JK arm to complete all four small
E+force phases and proved occupied force-response reuse. Direct's cold E+force
also completed, but the benchmark erroneously queried a DF-only metric on its
exact provider and stopped before warm/moved timing. The runner now leaves
that metric `null`; `retry-small-with-aot.sh` authenticates the unchanged
scientific/build inputs and exact core/AOT hashes before the next clean retry.
The first independent DF-JK PySCF check also rejected `moved-warm`: the runner
omitted the displaced coordinates and replayed the original geometry. It now
passes the moved coordinates on both changed and moved-warm calls. The two
earlier complete DF-JK phases and the changed-geometry phase met their
independent gates, while that moved-warm phase remains retained as a failure.
Both corrected ABBA pairs subsequently passed the independent PySCF 2.14.0
energy and full-force gates on the H200 source-tree build; this remains only a
three-atom pilot and has no crossover-claim eligibility. The installed qualification
uses `build-installed-pbe0.sh`: a source-matched sm_90 build with only the
PBE0 RKS stationary profile, an official CMake-installed native prefix, actual
CXX/CUDA cache-launcher commands, and a new independent three-atom acceptance
check before the 48-atom and non-water workloads. The prefix identity is
recorded separately from the earlier source-tree core/AOT binaries.
`run-installed-48-holdout.sh` pins that installed prefix and exact measured
source, runs Direct/DF-JK ABBA processes for 48-atom water and formaldehyde,
and collects separate intrusive profiles. Its terminal `NATIVE_RECORDED` is
only a work-completeness statement; independent PySCF E+force oracles and
fail-closed pair summaries are required afterward.
`qualify_oracles.py` runs each eligible native record's PySCF reference in a
fresh bounded CPU process, writes `NOT_RUN` for failed Direct 48 attempts,
retains timeouts/crashes and all per-pair `INCOMPLETE` or accepted summaries,
and never converts a platform `SUCCEEDED` state into scientific acceptance.
The small retained evidence under `evidence/` preserves the raw JSON byte count
and SHA-256. Native/oracle/profile records use deterministic gzip; summaries
remain readable JSON. Grid NPZ exports and full trace streams stay in the
ignored task-owned qz experiment directory.
GitHub may collapse these generated JSON diffs; expand an individual record or
run `verify_evidence.py` to inspect/check every full-precision value.
`evidence/platform/job-metrics.json` retains the platform's one-minute Pod
samples, including GPU memory usage rate. Its sampled maximum is not a
per-process measured peak and cannot replace the provider allocation ledger.

```bash
python tools/benchmark_hybrid_provider_crossover.py run-campaign \
  --cases water-3 water-48 formaldehyde \
  --output .artifacts/benchmarks/hybrid-provider-2054/native-1
```

Run each supported native record's independent oracle in a separate process,
preferably a CPU HPC allocation with the same source and PySCF version:

```bash
python tools/benchmark_hybrid_provider_crossover.py run-reference \
  --native .artifacts/benchmarks/hybrid-provider-2054/native-1/water-3-direct-0.json \
  --output .artifacts/benchmarks/hybrid-provider-2054/oracle-water-3-direct-0.json
```

In a separate GPU diagnostic process per supported arm, run `run-profile`
with `--case`, `--arm`, `--trace` and `--output`. It records actual trace
counters and requires a completed occupied projection reuse receipt for the
DF-JK arm. Never add profile durations to clean ABBA timings.

For each ABBA pair, pass its Direct, DF-JK and unsupported records, the
matching two PySCF records via `--oracle`, and both diagnostic records via
`--profile` to `summarize`. Review both pair summaries and every raw losing
record. A passing pair is a **pilot** only; full #2054 still needs a supported
third arm or explicit issue disposition, a successful Direct48 complete
endpoint, 96-atom coverage, measured memory/work decomposition and reviewed
source-matched crossover evidence before any policy discussion.

Grid NPZ exports and full trace streams remain in task-owned shared storage.
Small raw attempt and independent-oracle JSONs are retained as deterministic
gzip members in `evidence/`.
Provider metric records describe allocations and planned peaks; they are not
measured device peak memory. Missing work counters stay absent. Diagnostic
traces and profiles must be collected in a separate pass and never mixed with
clean endpoint times.
