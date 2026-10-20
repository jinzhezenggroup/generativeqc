# CBS Fast Lambda FP32 qualification

CUDA DF-CBS Fast force calls now request bounded FP32 Lambda auxiliary
preconvergence, then complete a fresh FP64 solve, independent transpose audit
and analytic force response. The requested auxiliary cap compensates for the
existing warmup planner's halving, retaining the ordinary Q32 ceiling subject
to memory admission. CCSD warmup and the existing Fast convergence allowances
remain unchanged. Standard, energy-only and standalone Lambda defaults retain
their previous behavior.

## Complete endpoint observations

The [C1–C7 table](timings.md) and homepage curve use freshly measured Standard
and Fast complete E+F calls from the same candidate binary and GPU within each
pair. This campaign includes the two newer master SCF changes absent from the
previous screen, so no historical timing is reused as a new measurement.
There is one observation per curve point; the curve alone does not establish
statistical significance or guarantee a speedup for every molecule/device.

A separate C4 ablation compares the exact newly built master-based prior Fast
with this candidate. Three fresh processes per variant run in alternating order
on the same allocated n1 RTX 5090. All six E+F results pass every constituent
and composite allowance against the new Standard C4 control. The observations,
medians and ranges are retained in `qualification.json.xz`. A further C6
prior-Fast/candidate pair on n5 checks whether the improvement survives the
larger shape and memory pressure; both are complete fresh-process endpoints.

| C4 Fast variant | Median E+F (s) | Range (s) |
| --- | ---: | ---: |
| baseline | 87.304 | 87.204–87.351 |
| candidate | 80.513 | 80.453–80.718 |

Candidate median time decreases by 7.78%.

The C6 prior-Fast/candidate observation is 611.895 → 539.205 seconds,
a further 11.88% reduction. The independently scheduled candidate curve call
is 539.012 seconds. This larger ablation is one pair, not a statistical cohort.

The repeated C4 cohort supports enabling this bounded warmup in Fast. Its
scope is one shape/device, not a universal performance claim. The curve's
larger C6/C7 cases expose the planner at greater memory/work pressure. Actual
Q batches (C7 Standard reduces to Q4 within the 32 GiB numerical cap),
FP32 GEMMs/casts, strict/mixed iterations and shared transpose
operator actions are retained, along with Lambda/source/orbital phase times.
The action counts exclude the separate independent audit; nested phase timers
must not be summed twice. Numeric capacity is not measured peak VRAM.

## Scientific qualification and boundaries

The model is unchanged: all-electron cc-pVDZ/cc-pVTZ MP2 extrapolation plus the
DZ DF-CCSD(T)−DF-MP2 correction, exponential HF alpha 1.63 and MP2 exponent 3.
The reference, correlation and analytic response use consistent RI-JK/JKFIT
auxiliaries. QM7 C1–C7 indices are 0, 1, 8, 48, 183, 293 and 5604; the dataset
SHA-256 is `fd6c26c1086887310518e6a6da778253bc5d3ceb8166007a679a740fff9ee6ea`.
All protocol identities and exact geometries are in the raw receipts.

The existing composite allowances are energy `1e-5 Eh`, maximum force
`1e-4 Eh/Bohr` and RMS force `3e-5 Eh/Bohr`. Component allowances divide by the
combined coefficient L1 norm. Every Fast curve point passes full-vector and
energy controls against its matched Standard, for the composite and all five
constituents. Standard is not an independent oracle.

C2 also passes the original independent reconverged PySCF constituent/composite
energies and normalized force projection at `1e-4` and `3e-5 Bohr`. That oracle
is reused only after exact full scientific-protocol equality; the original
receipt is retained in the new summary. Independent full-vector alkane checks
are not claimed for C1/C3–C7.

A separate water algebraic basis pair (STO-3G/def2-SVP, not a physical CBS pair)
qualifies all force components against reconverged PySCF central differences
at `3e-5 Bohr`. Both conventional-RHF/correlation-only DF and full RI-JK models
are checked at original and displaced geometries. All 20 complete endpoints
pass composite and constituent gates: each case contains two Standard and two
Fast 2 GiB calls plus one Fast 256 MiB call. The first call includes calculator
construction; later calls are complete process-warm replays, with no retained
converged density across basis groups. Every sample is independently gated,
not just the last repeat. Raw displaced energies, analytic forces and errors
are retained so the verifier can reconstruct finite differences.

The attempted 64 MiB complete water endpoint fails explicitly at joint DF
triples-response admission. Its earlier completed samples and exact failure log
remain in the bundle; it is not a published incomplete result. Separate native
GPU tests qualify the Lambda optional-memory/provider FP64 fallback, finite
warmup exhaustion and sticky nonfinite failure. The default warmup retains
`1e-6` handoff and at most 16 iterations; no tighter handoff or extra FP32 epoch
is introduced. Prepared batches remain unsupported by this CBS API.

## Measurement and source provenance

C1–C5 and the repeated C4 cohort run on n1; C6 on n5 and C7 on n4. Every real
GPU execution uses a finite Slurm allocation and preserves assigned visibility.
No GPU work runs on maintenance node n3. Numeric thread pools use one thread;
profiling is off during timing. GPU UUID, driver, power limit, Slurm ID and
controls are retained. Clocks are not locked; other scheduled jobs may use
other GPUs/host resources. Pairwise GPU identity is checked before plotting.

Timers include fresh calculator construction, both complete basis groups and
returned host forces. They exclude Python/preloaded DSO startup, dataset/basis
snapshots, JSON serialization and independent oracle work. Odd carbon sizes
run Standard then Fast; even sizes reverse the order. Every point has a
3600-second process limit; allocations use explicit finite limits. Companion
times overlap and must not be added. No duplicate Standard endpoint is hidden
inside Fast.

The durable reconstruction base is merged master
`5e0257826b65fcab52c02df4f364a643d20304c1`. The measured follow-up HEAD
`07c3933d7` has byte-identical native, Python and benchmark source at baseline.
`qualified-source.patch.xz` reconstructs the candidate from the durable base;
all 1,177 retained source/driver hashes were checked against that reconstruction.
The source manifest, both library hashes, driver hashes and cache observations
are inside `qualification.json.xz`. Builds use GCC 11.4.0, CUDA 12.9.1,
Release/sm120 and verified ccache 4.5.1. Cache statistics are shared observations,
not isolated campaign hit counts. No native source changed after compilation.

Native/Python transport tests cover unavailable versus measured-zero telemetry,
owner cleanup, invalid pointers/spans, batch exclusion and energy-only behavior.
The GPU suites pass nine complete DF endpoint tests and 31 Lambda numerical,
refusal and failure tests. Compute Sanitizer memcheck also runs the two Fast
complete DF tests with `--error-exitcode 99 --target-processes all`. Its
final run passes both tests with `ERROR SUMMARY: 0 errors`. The independent
PySCF oracle uses `PYSCF_MAX_MEMORY=32768` MiB because sanitizer
process overhead exceeded PySCF's default allowance on the first attempt. The
first attempt and final logs are both retained; native budgets and numerical
gates are unchanged. CPU convergence/transport tests pass 12 cases with
one CUDA case skipped in the CPU run. Remaining validation details are retained
with their exact logs in `qualification.json.xz`. The CI Sphinx build with
warnings treated as errors passes. An additional nitpicky run reports the same
252 unresolved cross-reference warnings on baseline and candidate; no new
warning was introduced.

## Reproduce

Adapt the preserved absolute paths in `candidate-environment.sh`,
`baseline-environment.sh`, `python-preload.sh`, `readme-pair.sh`, `cohort.sh`
and `c6-ablation.sh`.
Build the baseline from the pinned master, then apply the candidate patch and
rebuild with a verified compiler cache. Keep separate native/Python deployments
for the two variants. The full independent water driver is embedded verbatim
in its receipt. Always use finite `srun`, for example:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=01:10:00 bash readme-pair.sh 7
```

Replay scientific gates and regenerate the figure without CUDA:

```bash
PYTHONPATH=.:python python benchmarks/results/cbs-fast-lambda-20261021/verify.py
PYTHONPATH=.:python python -m tools.render_readme_cbs \
  --summary benchmarks/results/cbs-fast-lambda-20261021/summary.json.xz \
  --output .artifacts/cbs-fast-lambda-replay --readme-only
```

Recomputed composite water energy-error diagnostics permit only the standard
dot-product roundoff bound between BLAS reductions (the observed difference is
one ULP, `1.42e-14 Eh`). Component/raw values and all physical acceptance gates
remain unchanged; force diagnostics retain the tighter reduction comparison.

The publication manifest binds every retained file. Large working logs and
binaries remain ignored locally; no Release, tag or external archive was created.
The prior [screen](../cbs-fast-diagnosis-20261021/README.md) and
[CCSD-only Fast campaign](../readme-cbs-fast-20261020/README.md) retain their
original measurements and narrower conclusions.
