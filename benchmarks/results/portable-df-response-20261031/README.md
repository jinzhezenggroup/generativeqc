# Portable factored DF response

This campaign qualifies the device-independent long-charge schedule and the
full-rank pure-J factored metric derivative. Production source identity:
`1714f26e6f6bd4ecac0867f29744ebbf57ae717d3e7ceeda533a7a57365bca8b`.
The receipts retain each actual library hash; the Git tag passed to the runner
alone does not identify a build with these then-uncommitted source changes.

Run from the repository root with its Python/reference dependencies:

```bash
PYTHONPATH=python:. python benchmarks/results/portable-df-response-20261031/report.py --verify
```

The report authenticates original receipt bytes, independently replays every
retained complete population's dense energy/force gates, checks matching
scientific inputs and accelerator identity for comparisons, and recomputes all
medians and paired output errors. It prints cold and warm results separately;
positive percentages mean the new native endpoint is slower than its control.
`native-change` compares against the preceding native build; `gpu4pyscf`
compares against fresh GPU4PySCF observations. Neither is inferred from kernel
timings. Incomplete or failed calls are retained and never ranked.

The final bundle contains 101 original populations: 98 complete and independently
qualified, plus three incomplete controls. The final seven-functional warm
comparison is faster than GPU4PySCF throughout the measured RTX 5090 and PRO 6000
cohorts. A100 retains these warm losses at the explicit 1024-point force tile:

| Functional | Native (s) | GPU4PySCF (s) | Native overhead |
| --- | ---: | ---: | ---: |
| PBE0 | 3.258623 | 3.209719 | 1.52% |
| B3LYP | 3.268752 | 3.185835 | 2.60% |
| r2SCAN-3c | 4.216938 | 4.030315 | 4.63% |

A100 PBE is now 2.222298 versus 2.306539 seconds. All seven A100 cold endpoints
remain slower (18.73–82.49%). RTX 5090 cold PBE and r2SCAN-3c remain slower by
1.48% and 14.39%; PRO 6000 cold r2SCAN-3c remains slower by 6.96%. These sampled
observations do not claim a universal speed ordering.

## Measurement contract

The main population is water96, spherical def2-SVP (mTZVPP for r2SCAN-3c), DF,
Standard, stock quadratures, one math thread, one cold setup/energy/full-force
call and three warm calls from the engine's frozen cold density. Native uses
the public explicit `force_tile_points=1024` control; the SCF tile stays at its
default. This does not promote a new automatic force tile. Separate A100 size,
split and larger-force-tile controls retain their own labels and settings.
GPU4PySCF retains its stock convergence and initial guess and includes grid
response; r2SCAN-3c includes D4/gCP in both complete endpoints.

All device work uses finite Slurm jobs on n4/n2 or the ECNU A100 allocation on
gpu22. The reporter requires identical PCI identity and UUID where available.
A100's old host cannot supply the UUID through the reference environment, so
native/reference comparisons additionally require the same LSF job and assigned
visibility. The same original numerical oracles and audited environment-reuse
contracts from the preceding force-tile campaign are authenticated by file
hash; their producer identities and numerical gates are unchanged.

## Correctness and work

- Libcint five-point finite differences check signed metric factors through
  g/g in both Cartesian and spherical representations. The native probe also
  checks additive output, zero/changed graph replay, exact-capacity refusal and
  allocation-free replay. Long-charge and partitioned derivative cases retain
  their independent oracles.
- Complete endpoints check moved geometries, frozen replay, Standard/Fast and
  raw fallback. Separate larger-system and constrained-budget tests qualify
  resource behavior. PRO 6000 memcheck covers all eight new metric cases.
- The host fault-injection fixture retains all 31 occupied-response failure,
  bounded retry and lease checks. Its unused optional metric callback fails if
  accidentally entered.
- Every timing receipt retains iterations, Fock work, settings and available
  grid/nonlocal diagnostics. Factored response avoids `naux^2` dense metric
  weight elements per nonzero Coulomb term. Its ordered primitive-interaction
  bound is the square of the auxiliary primitive count; this is a bound, not a
  measured nonzero-source count. Charge partitioning preserves the complete
  source domain and the existing partial-capacity bound.

Separate A100 traces record one factored metric call avoiding 13,075,456 dense
weight elements for PBE96 and 46,457,856 for r2SCAN-3c96. Their instrumented metric
regions take 2.267 and 3.525 ms, respectively. These are component diagnostics;
`stationary_force` is inclusive of other force work and must not be described
as grid-only time. The bounded response workspace reservation is unchanged.

## Retention and limits

`receipts.json.xz` and `fp64.xz` retain the complete original JSON including force vectors,
timing samples, settings, work and source/binary/scheduler identities, together
with independently reproducible qualification results. `provenance.json.xz`
retains launchers, source changes, build/cache identities, test and sanitizer
outputs, and failed setup/control logs. The shared exact-FP64 v3 reader restores
all 101 original receipt strings and the complete original compact JSON container,
checking their original SHA-256 digests. The numerical bits, formatting, failed
populations and all reference identities are preserved; `sha256.txt` authenticates
the compressed streams. `provenance.json.xz` remains a JSON XZ stream. The original
files and large compiler outputs remain under ignored
`.artifacts/a100-remaining-gaps-20261031/`.

The initial A100 compiler/sysroot mismatch, stale build identity after archive
mtime restoration, shell-driver failure, incomplete WB97M-V timeouts, cancelled
retry and transient post-cancellation CUDA allocation failure are separate
records. Successful reruns do not overwrite or turn those failures into passing
measurements. The early A100 metric binary has a different source identity and
is not included in the final native/reference ranking.

These are sampled static default-grid endpoints, not a universal accuracy or
performance guarantee on unmeasured GPUs or every geometry/size. HF is outside
this seven-functional cohort; the prior HF/DF fitting-energy and small M06-2X
force-gate failures remain unresolved. WB97M-V retains the previously disclosed
different long-range fitting contracts between engines. Cold-start convergence
and remaining A100 performance gaps require further work.

Negative controls are retained: A100 2048/4096-point force requests are slower
than 1024 under the unchanged geometry scratch cap. The earlier
[batching investigation](../../../.agents/notes/proposed/2026-10-31-stationary-grid-batching.md)
explains why larger tiles require joint capacity admission. A separate public
core-initial-guess diagnostic on PRO 6000 increases PBE0/r2SCAN-3c/WB97M-V cold
iterations from 9/10/28 to 15/15/34 and worsens their cold latency. Neither
control is promoted. The diagnostic wrapper and actual settings are retained.

See the [decision note](../../../.agents/notes/implemented/performance/2026-10-31-portable-factored-df-response.md)
for the mathematical factorization, fallback boundaries and revisit conditions.
