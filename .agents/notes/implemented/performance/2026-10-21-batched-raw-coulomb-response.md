# Decision: batch source-only Coulomb response charges in existing scratch

Status: implemented
Date: 2026-10-21

## Problem

The full-rank source-only J response introduced by #2660 avoids repeated metric
refits, but its charge preparation requests one dense AO slice for every
auxiliary function. The README r2SCAN-3c DF96 diagnostic has 1,088 orbital and
6,816 auxiliary functions; the previous slice reader spent about 359.5 seconds
preparing charges. A single logical source pass is not sufficient evidence that
source generation is efficient.

## Decision

The private CUDA response interface accepts an optional pair-major raw-panel
reader. During charge preparation, the existing tile-sized derivative-weight
interval is dead. Generate each bounded raw panel into that interval and use a
column-major GEMM directly against all density terms. The last short panel uses
its actual column count; the output charges retain the full auxiliary leading
dimension. The source and consumer remain ordered on the same stream.

After contraction, preserve the existing metric factor order: project with
`Q^T`, divide by the retained eigenvalues, then apply `Q`. Do not form an explicit
inverse, transpose the source, retain an additional tensor, or change the rank
cutoff. Only full-rank, source-only, pure-J BLAS response is admitted. Exchange,
rank truncation, borrowed/retained/fitted sources and diagnostic algebra keep
their established routes.

`GENERATIVEQC_DF_COULOMB_RESPONSE=auto` selects bounded source panels where
eligible; `slices` retains the previous raw reader and `panels` retains the
general fitted-panel comparison. The selector now participates in resource and
restart identities. Older checkpoints retain their original absent-control
provenance and require explicit warm compatibility rather than being relabelled.

The existing `response_charge_blas_dots` counter measures logical scalar
charges, not library invocations. New `response_raw_coulomb_source_panels` and
`response_raw_coulomb_charge_gemms` counters expose the actual batching work;
do not interpret the unchanged logical charge count as unchanged launch count.

## Evidence and remaining bottlenecks

Source-matched receipts and launchers are retained under
`benchmarks/results/df-response-source-panels-20261021/`. Measurements are
diagnostic populations, with one warm repeat, not README medians.

The first candidate used base `cb2db7f380179e1315e92261c4afd1d90d14d572` plus the
retained patch, binary SHA-256
`33fef04e13023eec0e70fe493d8d9d03c82f6855d20f366c5a6da61e99b98f36`.
On RTX 5090 Slurm job 11852, ordinary DF96 charge preparation becomes 24 panels
and takes 62.081 seconds. Complete cold energy/force takes 2,085.248 seconds with
22 cycles and passes the unchanged dense-reference gates. Warm does not finish
within the 40-minute job limit; an attempted finite extension was denied by
Slurm. Keep this population incomplete. This is not evidence that the complete
endpoint became six times faster.

Explicit shell derivatives in job 11855 complete cold/warm in
1,641.120 / 174.027 seconds (22 / 1 cycles), passing the same gates. Three-center
derivative contraction falls from 471.383 seconds on the generic route to
32.857 seconds on the explicit shell route. Automatic source-only shell
promotion is a separate change, not implemented by this patch.

An ignored diagnostic interposer in job 11861 retains exact native packed raw
values on the host. With the same explicit shell route, cold/warm become
325.252 / 118.553 seconds and pass the same gates. Peak additional host storage
is 32,303,259,648 bytes under an explicit 40 GiB cap; 37 misses generate one
packed tensor and 1,665 hits replay 1,453,646,684,160 bytes. Later J builds take
about 5.3 seconds instead of 67--68 seconds. This diagnoses repeated integral
generation; it is not a production default or an accounted resource plan. The
job requested only the scheduler's default 8 GiB host allowance, below the
observed peak. Future cache qualification must request enough Slurm memory.
The retained sidecar binds the interposer, loaded library, launcher, cap and
scheduler receipt. No host cache is included in this production patch.

Changing only raw generation mapping did not solve the bottleneck: auxiliary
mapping takes 67.157 seconds per first J, component mapping 94.399 seconds, and
primitive mapping was cancelled after more than 7 minutes without completing
one J. These deliberately one-iteration runs are not endpoint populations.

## Validation and invariants

- Independent extended-precision adjoints cover signed multidensity terms,
  ragged/full panels, condition numbers through `1e10`, and refusal of exchange
  and rank truncation. Both slice and batched readers are exercised.
- Independent PySCF molecular energy/force endpoints cover PBE, r2SCAN-3c,
  auxiliary g functions, constrained memory, moving grids and a displaced
  geometry. The displacement is a correctness/lifetime gate, not a benchmark.
- Compute Sanitizer checks all 36 adjoint cases; no device errors were reported.
- Host resource/restart tests cover policy changes and historical checkpoints.
- Preserve complete endpoint timing, actual source work and the unchanged
  common energy/force gates. No oracle density enters the timed paths.

The final formatted implementation is based on merged master `0e691d400`.
Its binary is
`567b18fcd68e2e5d49e51bc7d37fac952ef6decc2b00ae76c55beead1dc1f2dc`.
The first final-runtime packaging attempt omitted upstream expression assets:
36 native adjoints passed but molecular setup failed. The corrected immutable
runtime binds 5,852 inputs, including those assets; failed attempts remain in
the evidence. Original large-system receipts keep their original source identity.

On that final binary, the complete r2SCAN-3c DF48 source-only comparison uses
forced dense values and a 4 GiB DF budget on both arms. Slice cold/warm take
254.762 / 97.029 seconds; bounded-panel cold/warm take 245.782 / 88.854 seconds.
Both use 16 / 1 cycles and pass the independent gates. Explicit shell execution
on the same candidate takes 185.129 / 29.443 seconds and also passes. These are
single-population diagnostics, with the same unchanged scientific controls;
they do not establish a default-resource or whole-README ranking. Initial 1 GiB
attempts correctly refuse a budget too small for metric/contraction storage.

## Revisit when

Qualify source-only shell dispatch independently. For source reuse, introduce
explicit native ownership, host/device budget admission, stream/lifetime rules,
allocation-failure fallback and complete endpoint validation before adopting
the temporary host-cache mechanism. Mapping changes alone do not remove the
repeated-source cost. Remaining WB97M-V range-separated work and broad README
ranking are outside this patch's demonstrated improvement.
