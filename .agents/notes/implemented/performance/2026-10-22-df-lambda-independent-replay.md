# Decision: independently admit smaller Lambda primal replay batches

Status: implemented on `perf/cbs-lambda-replay-20261022`; experimental qualification
Date: 2026-10-22
Source base: `afa0a1690be3c0c46774b0f3c2044b8f9a37fd9e`

## Problem

The Lambda owner previously attempted fresh original-graph matrix replay only
at the selected operator/audit Q batch. If the larger arena failed admission,
it retained the matrix audit but reverted replay to one-Q scalar execution.
The retained CBS-memory C6 trace showed 742 replay batches, zero GEMMs and
41,729 generated kernels despite an admitted Q32 independent matrix audit.

The first scratch analysis used the compiler's default coloring rather than
the retained binary's `GENERATIVEQC_NATIVE_CROSS_SIZE_ARENA=ON`. For that actual
build, the relevant capacities in bytes are:

| Shape | Existing audit | Replay at operator Q | Smaller replay |
| --- | ---: | ---: | ---: |
| C6, o25/v129/Q742, operator Q32 | 13,912,303,288 | 22,417,496,456 | Q16: 11,250,350,728 |
| C7, o29/v149/Q858, operator Q16 | 13,006,442,808 | 20,200,022,280 | Q8: 10,174,695,304 |

These are symbolic scratch capacities, not total memory, measured peak VRAM or
speedup estimates. Both coloring policies suggest the same opportunity.

## Decision and invariants

Keep the original same-batch admission attempt unchanged. If it fails, finish
selecting the operator, independent audit and immutable core cache, then try
successively halved replay batches against the existing shared scratch lease.
No additional device arena is admitted, and this fallback cannot displace an
incumbent operator batch, audit or cache. The existing same-batch attempt can
still expand scratch before core-cache admission; this change does not alter
that older priority policy.

Bind replay whole/tail contraction descriptors separately from audit descriptors.
The conservative existing descriptor charge remains; any additional replay tail
variant must fit the remaining host budget. Keep this extra charge outside the
core-cache rollback snapshot, so an allocation retry cannot erase the charge
for still-live replay descriptors. Audit/provider fallback clears replay and its
extra charge. Arithmetic, binding and driver failures continue to propagate.

The private owner records the selected replay batch; no public/native diagnostic
struct ABI changes. Progress traces emit `df_lambda_primal_batch_size` and
`df_lambda_primal_matrix`. These are observations, not additive `lambda_*` work
counters. Replay still uses immutable original t1/t2/B factors, original expanded
equations, strict FP64 contractions and original Q accumulation order. It runs
outside any active mixed-precision warmup epoch. Accepted solver residuals and
cached reduced cuts are never substituted for an independent fresh replay.

The generated scientific header is byte-identical to the baseline. Only host
binding wrappers change in the generated CUDA unit/header.

## Validation and measured scope

The pilot uses a two-object overlay on the qualified CBS-memory native library,
not a clean whole-master build. Only `src/cc/df_lambda_cuda.cu.o` and
`generated_df_lambda_cuda.cu.o` are replaced. All borrowed objects/archives and
the original library were hashed before/after linking and remained unchanged.
The relevant CC source/generator closure has no intervening drift from that
qualified campaign to the source base above. Build commands, object identities,
source patch and compiler-cache observations are retained in
`.artifacts/cbs-replay/build.json` and its adjacent logs.

- Baseline library SHA-256:
  `65a401ed3db8fb303a5146f134cdb227af78928affe27e1f7571c09f270112de`.
- Candidate library SHA-256:
  `08274fd0125d3b934397f17bb4819f882ae4360a54e73b1f7aa8dac337851b73`.
- GCC 11.4, CUDA 12.9.1, Release/sm120, cross-size arena ON; verified ccache 4.5.1.
- 772 related tests pass with every GPU test enabled through Slurm.
- 18 new cases pass Compute Sanitizer memcheck with zero errors. They cover
  separate/serial scratch, cache on/off, external energy source, whole/partial Q,
  independent odd-tail host charges and audit refusal. Published amplitude,
  Lambda and parameter/factor arrays match the disabled-replay controls bitwise.
- Nine independent full-DF native/CBS tests pass.
- Six water CBS Fast calls, at original and displaced geometries, pass fresh
  reconverged PySCF energies and full-vector central differences at 3e-5 Bohr.
  This uses the algebraic STO-3G/def2-SVP pair, not a physical CBS convergence
  claim. Maximum composite errors are 6.291e-9 Eh energy, 2.916e-8 Eh/Bohr maximum
  force and 1.492e-8 Eh/Bohr RMS force. Calls include process-warm reuse and a
  256 MiB constrained budget. The nine displaced energy pairs are retained.

Initial test failures were fixture mistakes: the probe returns an array list,
and the first o2/v7 shape allowed the pre-existing same-batch expansion to
consume capacity before core-cache admission. The revised o2/v12 fixture
actually exercises the independent fallback. No native-code change was required
for these failures. Preserve the initial failed log alongside final passing logs.

The first clean C6 pair measures 491.029810 -> 468.282156 s (4.6326% lower
complete E+F time). Lambda measures 209.618433 -> 186.092614 s. Both variants
reserve 30,412,407,625 bytes of numeric capacity and 16,677,596,608 bytes of owned
device capacity, with identical solver work counts. Composite and constituent
energies are identical; maximum paired force difference is 1.544e-13 Eh/Bohr.
C4 is unchanged at 67.351781 -> 67.325862 s. These are within-GPU observations;
other scheduled jobs may share host resources and clocks are not locked.
A separate same-GPU trace pair measures replay 51.260839 -> 29.036757 s and
complete E+F 484.757 -> 462.594 s. Replay uses 742 -> 47 Q batches,
41,729 -> 4,454 generated kernels and 0 -> 1,457 GEMMs. Its semantic summands
are 12,664,396,601,772 -> 12,664,087,911,222, essentially unchanged; this is an
execution improvement, not a new asymptotic algorithm. Matrix replay records
1,067,254,539,952 logical packing-output bytes; this is not measured DRAM traffic.
The existing operator remains Q32, while the new trace observes replay Q16.
Trace timings are excluded from clean timing cohorts. A second C6 clean pair,
with baseline first on another GPU, measures 488.277940 -> 466.059190 s
(4.5504% lower complete time). The two within-GPU pairs are not pooled into a
cross-device median. The baseline-first C7 pair measures 1098.699206 ->
1034.298850 s (5.8615% lower complete time); Lambda measures 515.600787 ->
451.045458 s. Numeric/device capacities remain 33,459,900,883 /
15,714,564,872 bytes, with the same published solver work counts and operator Q16.
C7 has one clean pair and no separate replay trace. Across all clean pairs,
all component/composite energies are identical and the maximum force difference
is 1.762e-13 Eh/Bohr. Both variants pass the matching retained Standard
component/composite allowances after exact protocol equality; these controls
are explicitly historical and are not independent oracles for C6/C7. Full working
receipts, reproduction scripts, original failed logs and build diagnostics remain
in `.artifacts/cbs-replay/`.
`report.md` summarizes the results; `analysis.json` retains recomputed checks,
and `trial-evidence.json.xz` bundles exact receipts, source/build identities,
traces, scripts, failed/final logs and fresh water oracles. `analyze.py` verifies
all full vectors, unchanged capacities/solver work, source/library identities,
closed nonoverlapping Lambda trace counts and reconstructs the water finite
differences. Run it from the checkout with `PYTHONPATH=.:python`.

The PR retains raw clean timings, quantitative water errors, work/capacity
comparisons, binary/source identities and the measured patch in
[the pilot summary](../../../../benchmarks/results/cbs-independent-replay-20261022/README.md).
The repository had only about 12 KiB of evidence budget remaining, so full
vectors, oracles, closed traces and exact build commands remain in the
checksum-bound local archive. The summary does not independently replay those
raw vectors. Publication accepts scoped numerical evidence; broad performance
promotion and a clean whole-master build remain unclaimed. Slurm jobs 12923,
12935, 12924 and 12926 carry the first C6, reverse C6, C7 and qualification/trace
cohorts respectively; all use finite main/gpu:5090:1 allocations on n1.

## Rejected alternatives and revisit conditions

Do not silently grow replay scratch, shrink the iterative operator batch or drop
core retention to improve an optional one-shot replay. Those choices require a
separate complete-endpoint policy comparison. Blindly increasing Q can increase
packing/intermediate traffic even when memory fits. Removing replay or reusing
accepted residuals would weaken the independent physical acceptance check.

Further work should profile original replay/audit lowering and triples response
after this admission defect is removed. A broader default-performance claim
requires a clean source build, repeated matched endpoints and qualification over
more methods/resource regimes. Energy-only CBS does not execute this Lambda
replay and should not be assigned the observed force-endpoint benefit.
