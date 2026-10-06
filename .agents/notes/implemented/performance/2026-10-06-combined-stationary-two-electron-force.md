# Combined stationary J/K cotangents for total-force consumers

Status: implemented; GPU evidence predates the latest base rebase
Date: 2026-10-06

## Why this boundary

GPU4PySCF 1.8.1's `RYS_per_atom_jk_ip1` first combines the J/K density products
into `dd_cache`, then contracts one weighted derivative into atom gradients.
Our separate J'/K' scheduler already shares shell traversal, cached primitive
geometry and Boys moments, but still evaluates the weighted derivative once
per active source. Shared traversal is therefore not equal arithmetic work.

An explicit two-cotangent DAG would preserve source decomposition while sharing
geometry expressions. Static scalar-DAG inspection found only approximately
4--8% arithmetic reduction for most p/d classes (ssss was 37.5%). It does not
remove the duplicated weighted derivative and is not the first experiment.

## Decision

Normal full-range Direct total-force consumers may request `Combined`: the
existing compiler-generated density coefficients precontract J/K together
before invoking the existing weighted derivative roots. Independent-source
exports retain `Separate`. Both use the public shell-class/resident/bounded
scheduler, primitive cache, precision, screening and translational scatter.
Order-four/five weighted workers now admit both layouts; unsupported classes
retain the existing exact Cartesian fallback. No recurrence math is copied.

The private snapshot v2 bridge makes the layout explicit. It publishes
`[one_electron, overlap_pulay, two_electron]`, never a combined result falsely
labelled J and a fabricated zero K. Compiler reduction accepts this complete
source grouping only when explicitly selected for full-range sources. The v1
bridge and raw J'/K' export retain their independent layouts and meanings.
Range-separated and fitted consumers retain their current contracts. Full-range Direct CUDA total-force execution defaults to `Combined`;
`GENERATIVEQC_DIRECT_FORCE_REDUCTION=separate` retains independent source
contraction. Older native libraries without the v2 symbol use the complete v1
owner; execution errors or malformed v2 output do not silently fall back.

## Acceptance

Require independent Libcint J'/K' and combined-total gates at the same final
RKS/UKS density, Cartesian/spherical def2-SVP, J-only and hybrid weights, all
bounded/angular/resident schedules, and separate-after-combined replay. Verify
public PBE0 moving-grid forces, reconverged finite differences, warm and moved
geometry replay. Retain clean complete 48/96-atom endpoint timings, force/J/K
component diagnostics, actual output-layout receipts and SCF work counts.
Do not promote the default without a complete-endpoint benefit.


## Evidence and provenance

The real-device qualification used commit `8da20992c` on base `0133f2b66`,
with an RTX 5090 allocated through Slurm's `main` partition. It passed 18
independent Libcint fixed-density derivative/replay gates, 33 public hybrid
force gates (RKS/UKS, Combined/Separate, AUTO/FP64, PySCF and finite-difference
oracles), and two resident/spherical cases under Compute Sanitizer with zero
errors. Local C++/CUDA builds used verified ccache launchers and retained cache
statistics and source/binary receipts.

The complete PBE0/RKS, spherical def2-SVP energy/force endpoint includes the
synchronized host force return. Each mode has five warm and five moved-warm
samples; each of these samples has one SCF iteration and one Fock build.
Tracing is disabled for these endpoint timings.

| Atoms / AOs | Phase | Separate median (s) | Combined median (s) | Time reduction |
| --- | --- | ---: | ---: | ---: |
| 48 / 384 | warm | 10.087066 | 9.621586 | 4.6% |
| 48 / 384 | moved-warm | 10.078850 | 9.600984 | 4.7% |
| 96 / 768 | warm | 28.106340 | 26.602920 | 5.3% |
| 96 / 768 | moved-warm | 28.122811 | 26.705073 | 5.0% |

The cold endpoint is not a matched-work speed comparison: Separate/Combined
needed 24/26 iterations at 48 atoms (99.217058/104.360145 s) and 29/26 at 96
atoms (256.208228/232.324455 s). The first moved endpoints both needed 12
iterations: 54.153477/53.891338 s at 48 atoms and 127.130785/125.869075 s at
96 atoms. All repeats passed the independent reference gates; maximum energy
and force errors were below 1.1e-10 Eh and 3.1e-11 Eh/Bohr respectively.

Separate component-traced runs at `8da20992c` measured 96-atom warm J, K,
and two-electron force medians of 2.417897, 1.985972, 8.297453 s for Separate,
and 2.425470, 1.955386, 7.177313 s for Combined. The force region saves 13.5%;
this is not the complete-endpoint speedup. The 48-atom pilot at `14afa7a53`
measured the force region at 3.114818/2.646490 s (Separate/Combined).

The pilot also retained a negative 96-atom moved-warm endpoint result:
27.572428/28.038181 s, with Combined samples spanning 26.907--32.459 s while
other builds/qualification jobs were active. The later cohort above is the
reported final measurement; the pilot remains preserved and does not support
a universal speedup claim.

Pilot work instrumentation recorded identical admissions in both modes at
each geometry. Warm counts were 23,220,006 shell, 320,981,006 AO, and
1,511,432,246 primitive-AO quartets at 48 atoms; at 96 atoms they were
92,233,228 shell, 1,263,186,780 AO, and 5,944,643,268 primitive-AO quartets.
The improvement comes from contracting the combined derivative weight,
not from dropping admitted quartets or relaxing numerical gates.

The PR was subsequently rebased onto #2020 head `a63d8364b`, incorporating
the latest pre-AO CSR scheduler and materialized pair derivative fallback.
The conflict resolution preserves that fallback and prevents duplicate
order-four/five consumption in either layout. After rebase, 209 host tests
passed and the compiler structure check covered 480 modules with zero
dependency errors. The user explicitly requested opening the PR without
further tests. The pending build chain was stopped before its automatic GPU
test launch, and no rebased native/GPU qualification is claimed. The measurements
above must not be presented as GPU qualification of the rebased PR head.

Raw endpoint, trace, work-count, gate and sanitizer receipts remain in ignored
`.artifacts/force-jk/evidence/{pilot,final}/`; the host-only analysis is in
`.artifacts/force-jk/summarize-results.py` and `qualification-summary.json`.

## Revisit when

Requalify the rebased native path when requested. Reconsider the default for
unsupported shell classes, resource limits or workloads where matched-work
complete endpoint timing does not improve. Independent-source consumers must
continue to receive separate J'/K' channels; any future shared-DAG design must
show complete-endpoint benefit beyond this precontraction.
