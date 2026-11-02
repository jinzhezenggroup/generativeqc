# Decision: bind two-center metric construction to the raw value schedule

Status: implemented
Date: 2026-11-01

## Problem

Source creation already freezes separate compiler-owned mappings for raw and
transformed DF values. Automatic transformed outputs use a cooperative warp
because they reduce over source auxiliaries. The two-center auxiliary metric
incorrectly selected that transformed mapping even though each output is a raw
integral with no transformed auxiliary sum. For short auxiliary contractions,
this wastes most of the warp. A frozen 5090 r2SCAN-3c/96 profile spent about
0.874 s in the metric setup kernel, enough to explain the remaining small cold
endpoint deficit.

## Decision and invariants

Use the existing raw mapping for both metric launch geometry and the kernel
mapping argument. Automatic selection uses one thread per output; explicit
auxiliary/component/primitive controls keep their original meanings. The
transformed consumer, integral equations, strict FP64 policy, screening, grids,
convergence, resource capacities and host/native ownership are unchanged.
Selection depends on the consumer's work, with no GPU-name or architecture
restriction. The explicit primitive control remains available for studying
other contraction domains.

Do not replace the transformed schedule globally. A preliminary auto/auxiliary
control identifies a plausible cold benefit, but it also changes a separate
consumer. Only the focused source change is the production candidate.

## Source and evidence

The predecessor source is
`6527e5625b096f10a8c431ce2829b67d44a98abb656889f486f2b13786dc419c`;
the candidate source is
`cbd56578885c044e5ccbc636f4d0151e37a039834102a8cbb33da546d1652e25`.
Frozen images, original source patches and complete raw results are retained in
`.artifacts/metric-value-schedule-20261101/` (control pilot) and
`.artifacts/metric-value-schedule-candidate-20261101/` (focused change).
Publication commits do not replace these measured source identities.

The pilot's twelve conditioning/measured populations independently qualify.
5090/PRO 6000 cold medians improve 3.37%/3.18%, with warm changes of
+0.12%/+0.02%. The focused r2SCAN-3c comparison has nine populations per device:
one conditioning owner for each implementation, then parent/candidate/GPU4 and
the reverse order, each measured owner having cold plus three warm calls.
All eighteen populations pass the independent README gates; every paired native
SCF iteration and Fock-build count is unchanged. Maximum paired energy and force
differences are below 2e-12 Eh and 2.3e-12 Eh/bohr.

| GPU | Parent cold | Candidate cold | GPU4 cold | Candidate warm | GPU4 warm |
| --- | ---: | ---: | ---: | ---: | ---: |
| RTX 5090 | 23.361227 s | 22.523749 s | 23.294689 s | 6.839443 s | 9.890556 s |
| RTX PRO 6000 | 21.841829 s | 21.136324 s | 22.774373 s | 6.419302 s | 9.154639 s |

The metric/raw/J tests cover sixteen automatic and explicit mapping cases,
Cartesian/spherical bases, auxiliary g, rank-deficient metrics, signed density,
ragged tiles and differing batch layouts. Four mixed-representation derivative
cases and forty-eight independent raw/metric/J/K value runs also pass, including
long contractions. Six public cold/warm/moved endpoint cases independently check
HF/PBE0 forces, generic fallbacks and constrained response memory. These are
scientific qualification gates, not timing populations.

Six direct native processes pass Compute Sanitizer with zero errors and zero
bytes leaked, and their raw/metric/J/K outputs independently pass against the
previously constructed libcint/NumPy tensors. The Python-wrapper sanitizer is
retained too; the direct native runs make the instrumented execution explicit.

The 108-atom control completes cold, retained, deliberate eviction, replay,
moved and moved-warm phases for both sources. Cold improves from 30.748 s to
29.827 s and moved geometry from 22.430 s to 21.503 s; paired energy/force
differences stay below 2.8e-12 Eh and 2.6e-12 Eh/bohr. This larger control is
paired lifecycle evidence, not a new independently qualified production ranking.

A separate complete profiled endpoint confirms 0.875058 s versus 0.100226 s in
metric construction. Its measured grid shrinks from 11,614,464 to 362,952 blocks
of 128 threads. Both still publish 46,457,856 matrix elements for 6,816 auxiliary
functions; the unchanged symmetric domain implies 23,232,336 unique metric
pairs. That pair count is derived from dimensions, not an instrumented arithmetic
counter. All 69 recorded downstream work/capacity receipts agree exactly.
Profiler timing is excluded from the clean endpoint medians.

The completed seven-method/two-device campaign retains 126 conditioning and
measured owners for the tables, plus one scientifically qualified but resource
unqualified owner. All fourteen comparisons beat GPU4PySCF at the sampled
README water-96 defaults. Cold/warm latency reductions are 3.31–37.62% /
8.25–76.22% on 5090 and 7.19–40.58% / 8.95–87.71% on PRO 6000. Small positive
changes against the native predecessor remain visible, including +0.276% cold
and +0.040% warm for 5090 WB97M-V; these observations do not establish universal
non-regression. The replay retains full sample dispersion.

Slurm job 15148 timed out before its last parent WB97M-V receipt was written.
Although that receipt is numerically complete, per-phase wall timestamps are
absent, so none of its timings enters the resource-qualified medians. The
original values and timings remain independently gated and separately reported.
Two recovery attempts did no numerical work: the first failed on disabled
accounting storage, and the second found all receipt files complete before the
resource audit identified the late owner. Job 15183 repeats only that owner in
a fresh finite allocation on the same physical GPU. Its actual scheduler
identity, successful end and receipt timestamp are checked explicitly. A
foreground inner timeout expires before the outer Slurm deadline and has a
finite kill grace. Never infer resource qualification solely from a complete
receipt or a runner's zero exit code when the scheduler reports TIMEOUT.

The [complete evidence and offline replay](../../../../benchmarks/results/metric-value-schedule-20261101/README.md)
retain every conditioning, measured, excluded and diagnostic population. Normal
and optimized Python replay all 141 independent population gates identically.

The isolated aTrust VPN portal authenticates, but the A100 SSH entry resets
connections before key exchange. Host-direct TCP probes time out and are not a
test of the isolated VPN. This source has no new A100 measurements. Retained
older A100 numbers must not be assigned to this source. Neither this local
repair nor two-device evidence establishes a universal GPU guarantee.

## Revisit when

Measure a larger or longer-contracted auxiliary domain that benefits from a
different mapping. Any new selector needs complete endpoints, independent raw
metric and force gates, retained work/resource counts and explicit fallbacks.
Do not infer production benefit from lane utilization alone. The two discarded
cold-start experiments are documented in
[the rejection note](../../rejected/2026-11-01-r2scan-cold-controls.md).
