# Decision: reduce per-Fock J and XC work after the default promotion

Status: implemented
Date: 2026-10-19

## Problem

Similar complete cold endpoints concealed different iteration counts. At the
previous default, 96-atom r2SCAN-3c used 24 native Fock builds; the fresh released
GPU4PySCF 1.8.1 run used 59 SCF iterations and 60 `get_veff` calls. The counter
wraps `get_veff`, including its pre-loop evaluation, rather than counting J and K
separately. Dividing energy-plus-force time by iterations mixes preparation,
forces and SCF work, so it is not a valid per-iteration measurement.

Separate n4:1186 profiling confirmed the user's concern. Native J+XC had a
4.587006 s median interval over 24 actual Focks. GPU4PySCF's median `get_veff`
was 2.173046 s over 55 calls in a 54-iteration instrumented trajectory. These
are different SCF populations; the clean 59/60 counts must not normalize the
instrumented 54/55 times. Default AO cutoffs and J omission contracts also
differ, so these are actual per-round costs, not equal executed work.

## Decision

Keep the default complete primitive J and spatial XC. Improve their existing
execution with two independently measured changes; no new opt-in is required.

1. After `E D` precontraction, bound the represented primitive Hermite source
   by `sum_t abs((E D)_t) * norm(Lambda_t)`, with outward arithmetic. Use the
   minimum of this triangle certificate and the old primitive norm times the
   shell density envelope. The old envelope discards cancellation before it
   reaches the Hermite source. Cauchy-Schwarz bounds each directed output by
   `q_output * source_norm`; the old cumulative allowance remains
   `min(screening, 1e-12)/(maximum_pair_primitives * primitive_count)`.
2. Emit explicit FP64 `fma` in tiled XC density products, cross potential
   products and the range-safe diagonal reuse branch. Production NVCC disables
   implicit contraction with `--fmad=false`; spelling multiply/add separately
   therefore costs separate arithmetic. Preserve FP64 storage/accumulation,
   index order and one accumulator. Cross products add their two legs
   sequentially, so this is an independently qualified rounding change.

The source-norm pass initializes all real hierarchy leaves, including zero for
inactive system windows. Existing `qD` storage holds the certificate. Hierarchy
maxima use that same certificate at coarse and leaf tests. A q-sorted prefix
retains a primitive if either direction can contribute, using maximum source
norm or maximum output norm respectively. Every replay recomputes the norm and
prefix on the owning stream; pointer identity never authenticates density.
No additional per-primitive allocation is introduced. Floating-point transform
and accumulation error remain separate from the mathematical omission budget.

Keep reference/incumbent exact-AO-mask semantics, optional-capacity fallback,
unchanged K and force providers, and canonical moving-force grid ownership.
XC's unsafe diagonal range fallback retains its original two-product
expression; FP32/mixed arithmetic is unchanged. The SCF algorithm, convergence
thresholds, quadrature and AO cutoff are unchanged.

## Evidence

[Compact samples and reproduction](../../../../benchmarks/results/pr2609-per-iteration-20261019/README.md)
retain raw timing samples, work counts, errors, source reconstruction and
binary identities. Full logs/traces remain in `.artifacts/pr2609/per-iteration/`
and source-matched n2/n4 snapshots. Timing hardware is RTX 5090, CUDA 12.9.1,
Release/sm120, FP64/direct/noDF, 48x16x32 unpruned moving grid (2,359,296 points).
This is a matched-grid implementation diagnostic, not an independently
grid-converged production ranking. Every GPU command ran in finite `srun`.

| Native r2SCAN-3c per-Fock measure | Previous default | Candidate |
| --- | ---: | ---: |
| J kernel median | 1.572935 s | 1.456541 s |
| XC kernel median | 2.987349 s | 1.773992 s |
| J+XC elapsed median | 4.587006 s | 3.257288 s |
| Focks in profiled cold trajectory | 24 | 24 |

This is a **29.0% reduction** in the measured J+XC interval. Initialization,
solver and forces are excluded from these intervals. Native CUPTI records have
zero dropped records/callback errors. Upstream normal-cutoff J/XC event medians
are 0.719470/1.453108 s, so per-round parity is **not** established: native
J+XC is still about 50% slower in these actual configurations.

Fixed-density controls isolate the mechanisms:

- MINAO raw J: r2SCAN basis 1.561526 to 1.434851 s; PBE0 basis 0.487096 to
  0.449098 s (five warm medians; upload, J and host publication included).
  These clean J timings use the source-norm-only snapshot; its MD sources are
  identical to the final combined candidate, which supplies the counted work
  and full endpoint measurements below.
  Pinned upstream MD remains 0.629825/0.348320 s with different screening
  semantics. Candidate versus previous J differs by at most 2.84e-14; versus
  pinned MD by 1.62e-10, within `atol=1e-9, rtol=1e-10`.
- Separate counted r2SCAN J has 2,244,588,658 to 2,176,376,338 radial roots,
  4,253,320,880 to 4,108,320,392 directed products and 126,929,687,520 to
  119,130,181,760 Hermite summands. PBE0 summands fall from 26,247,756,768 to
  24,808,667,616. Both retain zero residual work. Counters are absent from clean
  timing runs; `uniform_*` is the historical counter name for these MD tasks.
- Same core density, exact spatial grid and cutoff 1e-16: complete native XC
  3.016372 to 1.797135 s, 40.4% less time. Five warm samples after one separately
  retained first call; setup is separate. Active AO sum 2,008,216, maximum 450,
  point-AO-square work 122,486,172,672 are unchanged. Same-input GPU4PySCF
  occupied/dense XC at cutoff 1e-16 is 1.918083/2.444948 s; default 1e-10
  occupied XC is 1.447156 s. Native versus upstream Vxc max error is 5.82e-14
  at matched cutoff, 1.34e-11 at default upstream cutoff. Increasing native
  tiles to 1024 remains slower (1.985625 s), so 256 is retained.

Clean full E/F, n4:1189; cold/moved n=1, warm medians n=5:

| Endpoint | Previous default | Candidate | Candidate Focks |
| --- | ---: | ---: | ---: |
| r2SCAN-3c cold | 158.655704 s | 125.702913 s | 24 |
| r2SCAN-3c warm | 25.963102 s | 24.697389 s | 1 |
| r2SCAN-3c moved | 97.468332 s | 80.255866 s | 13 |
| r2SCAN-3c moved-warm | 25.944059 s | 24.708059 s | 1 |
| PBE0 cold | 62.452559 s | 57.834300 s | 17 |
| PBE0 warm | 10.373085 s | 10.112376 s | 1 |

All counts match the preceding default; per-Fock AO work also matches. All 18
independent E/F gates pass (`1e-8` Eh, `1e-7` Eh/bohr); maximum errors are
9.10e-12 Eh and 4.51e-11 Eh/bohr. Fresh clean upstream r2SCAN cold is
153.314540 s (n4:1180); candidate cold takes 18.0% less time, while still having
slower individual rounds. Warm endpoints remain dominated by forces.

Candidate source identity:
`cb3cee83afc8d8f066b89ceb1548b5605a17ee7e329dbb10436427b4e69f3888`.
Library SHA256:
`f886f18ad70c6330cb83ab589139f83ab55976cdc5b940fd204d674e2022bbdc`.
Base revision is `992c744575b2953390fb4b54b41096dc12666988`; the retained patch
reconstructs the measured production sources. Six stationary-force manifests
were regenerated and deployed with matching libraries. Verified ccache 4.5.1
was used; retained shared-cache observations show 18 hits, 96 misses and no
new cache errors during builds and host qualification, not exclusive job counts.

Qualification: 818 host MD tests, 123 host XC tests, 114 device MD tests
(n2:3486), and 133 device follow-up tests (n2:3488). The new cancellation test
captures one owner, cancels antisymmetric/opposing-spin sources, perturbs the
cancellation, then restores density. Independent libcint checks every replay.
Native XC suites cover all retained semilocal families/spins, variational
behavior, tails/maps, graph replay, mixed handoff and resource refusal. All four
Compute Sanitizer tools pass representative small MD and spatial-XC probes;
these are not 96-atom sanitizer claims. Full pinned upstream package tests
remain excluded because that package is unavailable; retained pinned-MD and
independent libcint matrices supply their explicitly stated separate gates.

Resolved qualification setup failures remain visible: an initial host command
omitted the library path; the semantic counter needed to recognize fused as
well as unfused emitted terms; n2's default gcc lacked `cc1plus`. Corrected
invocations pass, with `NVCC_CCBIN=/usr/bin/g++-11` for the n2 fixture builds.
No numerical gate or semantic work assertion was relaxed.

## Consequences and rejected alternatives

This closes a large XC arithmetic gap without relaxing AO screening, but J
and the deliberately different omission contracts remain the main per-round
gap. Do not claim that copying an algorithm establishes identical work,
compiler instructions, convergence or whole-endpoint timing.

The earlier [MD radial FMA experiment](../../rejected/2026-10-19-md-j-explicit-fma.md)
is still rejected: it did not speed the actual native J consumer. This XC
change is a different measured bottleneck. Broadly enabling global implicit
FMA, loosening screening, promoting slower BLAS/tile candidates or changing
solver policy are not justified by these measurements.

Revisit J scheduling and tighter rigorous omission budgets using counted work
and independent matrices. Revisit XC factorization only when complete
same-density XC and E/F improve on the now faster generated baseline.
