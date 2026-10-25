# Application precision qualification and reproduction

`PrecisionCalculator` selects FP32 arithmetic from an application error budget.
The user-facing behaviour — presets, allowances, fallback and audit semantics,
result fields and failure modes — is documented in
[Application precision and explicit audits](../user/application_precision.md).
This page owns the other half: how a Fast claim is qualified, which runners
produce the evidence, and what a reported ratio does and does not establish.

The cross-cutting policy that precision is an execution-schedule dimension, not
a global performance flag, is in
[Performance engineering](performance_engineering.md).

## What must be demonstrated

A precision claim needs three independent things, and none substitutes for
another:

1. **Executed arithmetic.** The receipt must come from operator-level evidence,
   not from the requested policy name. `Calculator(precision="auto")` is not
   `policy="fast"`, and neither name proves FP32 ran.
2. **Numerical acceptance.** An independent reference must pass the energy,
   maximum-force and RMS-force allowances in the same scientific model.
3. **Complete-endpoint comparison.** The timed object is the whole public call —
   facade/plan construction, SCF, strict refinement, analytic forces, any audit
   and cache release — against the optimized FP64 path with its reuse intact.

A kernel-level or isolated-operator speedup is not (3). Container or SGEMM
utilization is not (1).

## Runners

`benchmarks/pbe0_precision.py` compares ordinary FP64, native AUTO and explicit
`audit_auto` at fixed PBE0/def2-SVP spherical models and moving quadrature. It
requires an independent PySCF reference and retains first calls separately from
alternating measured repeats. All runs start from the same Hcore guess; warm
timings reuse process caches, not converged densities. The endpoint includes
calculator construction, SCF, analytic forces, host export and cache release.
Audit timings include the complete audit/fallback work. The current benchmark
schema is version 2; older measurements retain their original names and source
identities and are not relabeled as results of the current revision.

With the source tree on `PYTHONPATH` and the matching native library selected,
generate the CPU reference first (PySCF is an optional benchmark dependency):

```bash
python -m benchmarks.pbe0_precision --engine reference \
  --output .artifacts/pbe0-precision/reference.json
```

On an available, permitted Slurm GPU node, run the native comparison through
the scheduler, preserving its assigned device visibility:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --time=00:30:00 \
  python -m benchmarks.pbe0_precision --engine native \
  --reference .artifacts/pbe0-precision/reference.json \
  --output .artifacts/pbe0-precision/native.json
```

Speed ratios are emitted only after independent energy/force gates pass.
The report preserves native operator evidence and explicitly flags incomplete
precision inventories instead of inferring FP32 execution from the request.

For FP32 J/K Fast against already-optimized FP64, run
`python -m benchmarks.pbe0_fast_fock --engine reference --output <oracle.json>`
in one finite Slurm job, then `--engine native --oracle <oracle.json>
--output <comparison.json>` in another. The runner retains first calls and
alternating complete endpoints, checks independent energy/force budgets, and
rejects missing FP32 exchange or hidden extra solves. A source-matched existing
independent oracle is reusable only when its complete scientific protocol matches.

For the other qualified Fast routes, run `python -m benchmarks.pbe0_fast` inside the
same finite Slurm allocation, with `--reference`, `--output` and `--repeats 5`.
This reports the current ordinary FP64 default and budget-selected Fast, including
facade construction, strict refinement, analytic forces and any fallback work.
Use `--initial-guess automatic` to measure the default native Modified Huckel seed policy,
or `--initial-guess minao` for the explicit policy; the default benchmark uses
Hcore. Both timed modes always receive the same seed policy, and the independent
oracle's Hcore protocol remains separately recorded. It checks the same
scientific target; speed ratios are not kernel ratios.
Historical no-reuse/FP64 facade and audit measurements
remain bound to their original sources and are not relabeled.

The README endpoint runner also accepts `--policy fast`:

```bash
python -m benchmarks.readme_dft --method pbe0 native --policy fast \
  --atoms 3 --basis-file path/to/def2-svp.json --grid 48 16 32 --repeats 5 \
  --reference path/to/independent-reference.json \
  --output .artifacts/readme-fast/pbe0.json
```

Run this inside a finite Slurm allocation. Generate the matching reference with
the same runner's `reference` engine, which remains FP64. Add
`--density-fitting cuda --auxiliary-basis-file path/to/jkfit.json` to both engines
for DF. The default native `--policy fp64` still uses ordinary optimized
`Calculator`, not an audit facade. `--fast-budget ENERGY MAX_FORCE RMS_FORCE`
sets explicit allowances in Eh and Eh/bohr; all three independent oracle gates
must pass. Fast records keep the unchanged strict-parity `fp64_gate` separately
and use a distinct schema; they must not be folded into historical FP64 figures.

## Timing scope

Cold includes facade/plan construction; frozen post-cold, moved, and frozen
post-move calls include all requested work, including audits and seed restoration
when needed. A successful Fast request outside a qualified FP32 domain uses
the eligible Standard fallback or an audited FP64 call. Its report records which.

Runner support alone is **not** qualification of every README method or size, not
proof that FP32 arithmetic ran, and not a speedup claim. Private seed/force
experiments cannot be combined with Fast in this runner.

## Three-way production comparison

For the user-facing Standard-versus-Fast-versus-reference-engine comparison, use
[Production-default three-way benchmark](production_default_comparison.md). That
protocol owns the default-grid and default-budget rules that must match before
any ratio is meaningful.

## Evidence retention

Retained receipts, their source identities and their archive lifetimes follow
[Evidence retention](evidence_retention.md). A published ratio must name the
source revision, executed arithmetic evidence, independent-reference identity
and complete endpoint population; a bare speedup number is not evidence.

## References

- [Application precision and explicit audits](../user/application_precision.md)
- [Performance engineering](performance_engineering.md)
- [Production-default three-way benchmark](production_default_comparison.md)
- [Evidence retention](evidence_retention.md)
