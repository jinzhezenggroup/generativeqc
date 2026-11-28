# Decision: qualify wider RKS panels on an interacting water cluster

Status: proposed
Date: 2026-11-28

## Problem

The 96-H Hessian has 96 Cartesian STO-3G AOs. A 32-water cluster with
the same 96 centers and 82,944 moving quadrature points has 224 AOs.
FP64 Direct and DF both refused at XC workspace admission on `ff2c0ae91`.
The bounded generated affine fallback exceeds the default per-operation
work limit; the existing factored feature kernel has exactly three
32-orbital strips and cannot compute a wider panel.

## Candidate

Use the existing generated potential transform and fused AO adjoints, with
bounded, pedantic FP64 BLAS tiles for stationary feature and potential
contractions. For a feature bilinear, preserve both derivative legs:
`d(Aᵀ P B) = dAᵀ (P B) + Aᵀ (dP B + P dB)`.
No wider generated affine fallback is admitted under the reduced work count.
Reserve the tensor provider separately from the mandatory and optional arena
bytes. Neither the SCF functional policy nor the existing smaller route needs
to change. Bounded admission is not evidence of a speedup.

Jobs 20924/20925 passed XC admission with the prototype but refused at
the next moving-grid gate. A 224-AO s/p grid needs 371,589,120 retained
spatial values and a 1,114,767,360-term jet preparation bound; unlike the
s-only fixture it cannot use the four-plane radial representation. Grant
a separate finite geometry envelope to the wide route while charging its
entire arena against the same aggregate lease.

Passing geometry exposed two further, independent gates: Direct first
exceeded the unscreened response `224^4` census, then the certified nuclear
J/K source census; DF exceeded the ordered three-center raw nuclear work
bound. The exploratory host envelope probe reports 64,148,639 checks and
29,450,901 retained canonical groups under the original `1e-14` absolute
omission ledger (82,953,640 groups before the certificate). The Direct
trial bounds candidate checks and replay at `2^26`. For the full retained
set (not just the overflowing prefix), the host census is 9,328,461,147
primitive components, 890,453,780 ordered projections and 31,581,468,970
bounded reduction terms. This explains why a small increase to the old
cumulative work limit cannot admit the workload. Do not infer throughput
or correctness from the census.

For DF, batch the existing linked s/p AOT closures instead of raising the
ordered single-group launch bound. Preserve deterministic FP64 projection
and reduction, a single-use directional tape, and a full rebuild after
direction mismatch. Prepare both mandatory fitted maps before admitting
optional tape storage. The s/p path requires enough memory for its checked
local-jet tape; an ordered fallback at millions of groups is not practical.

The first DF profile (node4 job 2115) attributed 81% of sampled GPU kernel
time to scalar-launch nuclear attraction. The complete compact batch and
cached frozen H/S columns fit under separately bounded arenas but not
the narrow retained-storage limits. A second profile (node5 job 2687)
then exposed generated affine fixed-grid feature and potential actions
as the biggest cost during CPKS, so the candidate uses the same tiled FP64
product provider for both nuclear and induced grids. Neither route changes
the ordinary SCF path or the CPKS residual tolerance.

## Evidence and qualification

Independent GPU4PySCF 1.9.0 ran the reproducible interacting 32-water cluster
with Cartesian STO-3G, the unpruned 12×6×12 moving rule and second-order
STO-3G auxiliary-basis response for DF. Slurm 20922 (Direct) measured
139.87 s; DF jobs 20923, 20934, and 20935 measured 122.37 s, 120.86 s, and
119.83 s. Jobs 20935 and 2116 (older native DF variants) timed out after
12 and 11 minutes. Neither exhausted device memory; the latter peaked at
17,818 / 32,607 MiB. See the local ignored
`build/1905-gpu4pyscf/scaling/water96-endpoint.py` and `water96-n1.path`.
Require matched paired fresh endpoints on each card, all 288 raw columns,
no host fallback, a max matrix error of `2e-8 Ha/Bohr²`, and measured memory
peaks. This proposal does not qualify a new stock policy yet.

**Basis correction.** The original experiments used the same catalogue name
but not the same primitives (first oxygen exponent: `130.7093214` native,
`130.70932` PySCF). The raw cross-engine errors in those runs, including 20946,
2693 and 20970, are diagnostics for different models, not accuracy-gate results.
`benchmarks/water_hessian_gpu4pyscf.py` supplies the same explicit primitives
for the orbital and DF auxiliary roles and rejects inconsistent input and
output receipts. Do not change the XC domain or relax the numerical gate to
offset a mismatched basis.

Matched-basis job 20979 on one RTX 5090: 48-atom DF took 42.47 s native vs
19.10 s GPU4PySCF (raw maximum error `1.49e-9 Ha/Bohr²`); 96-atom DF took
300.00 vs 120.45 s. The 96-atom DF raw max error, asymmetry and translation
residual were `1.64e-9`, `4.04e-12` and `6.63e-12 Ha/Bohr²`, respectively.
Native performed 288 DF value and 288 contraction passes, 288 physical CPKS
checks, and 82,944 logical grid mixed contractions; all columns completed
with no numerical host fallback. The native/reference memory peaks were
16,771 / 4,703 MiB out of 32,607 MiB. **Precision passes, speed does not.**
The matched Direct reference completes in 138.56 s on job 20980 (4,841 MiB);
it is not a paired native speed result.

Five final-binary targeted tests passed in job 2689 before the full-size
timing arm: mixed s/p libcint values and gradient/JVP (arbitrary and unit
directions), all-s tape reuse, and 100-AO fixed/moving grid parity on a
partial last 1,024-point tile with the independent LibXC check. This is not
the full 224-AO Hessian accuracy gate.

### Full-size endpoint and follow-up

Job 20946 on one 32,607-MiB RTX 5090 completed all 288 native DF columns
with zero host numerical fallbacks: 469.23 s native vs 120.37 s GPU4PySCF,
18,763 MiB sampled native peak. The raw maximum error was `3.074e-8`, raw
asymmetry `4.03e-12`, and translation residual `6.62e-12` (Ha/Bohr²).
Neither accuracy nor performance qualifies. In Direct job 20942, the native
endpoint timed out at 28 min with a 15,839 MiB sampled peak; the GPU4PySCF
Direct baseline was 139.87 s. These are work limits, not an observed OOM.

The DF occupied–virtual projection combines the semilocal grid contraction,
the whitened fitted Coulomb Gram, and the orbital gaps. Reuse the resulting
FP64 operator as a solver aid, not as a replacement for the per-direction
physical residual audit. A rejected projection must take the bounded native
response route. On 48 atoms the projection cut physical induced Fock actions
from 4,314 to 288, 224.57 to 53.68 s; it changed the native result by only
`1.31e-14`. On wider fitted grids with all point linearizations valid, compute
one triangle of the mathematical XC Gram in rectangular blocks. Mirror only
this private operator estimate, never the published molecular Hessian.

Frozen `P·AO` jets use an independently charged optional cache. The frozen
density direction is exactly zero, so with this cache the moving features
need no density-direction GEMM. Fold physical AO potential planes into a
single GEMM reduction per point tile; logical quadrature is unchanged. The
pre-folded follow-up job 2690 completed 96-atom DF in 419.44 s against
152.07 s on the same RTX 5090; peak was 19,364 MiB and raw error `3.080e-8`.
The folded variant ran 48 atoms in 51.17 s vs 19.29 s in job 20961, with
`3.449e-8` maximum raw error. Cross-card times cannot isolate each change.

On the 48-atom case, tightening the reference CPKS or AO screen changes the
raw reference matrix by approximately `1.2e-10` in either experiment. An
independent five-point derivative of the reference gradient (including
moving grids and auxiliary response) validated the worst column within
`4.62e-9` at `h=0.001` Bohr, while its discrepancy against native remains
`3.28e-8`. Retain the accuracy gate; neither the solver nor AO screening
explains the difference. Compare the stationary XC domain and the separate
analytical molecular jet contributions before considering any domain change.

For Direct, per-atom projection spans reuse one host census across the
three directions of a frozen center. Trim inactive ends without moving any
projected offset or changing the reduction order. If metadata does not fit,
retain the incumbent replay. This reduces host scans and empty launches,
but cannot eliminate primitive replay; a source-driven multi-axis sweep
is the remaining high-leverage design. Default and eager cuBLAS 13 FP64
emulation on the 5090 gave no speedup on the tested wide moving-potential
and OV tile shapes, so no new deployment dependency is justified.

### Canonical DF and low-degree fitted jets

The dense Cartesian s/p source publishes only the requested raw JVP for the
frozen RHS. It launches the per-atom active groups, and a distinct orbital
shell pair or metric pair owns both transpose entries. In the pullback, sum
the two weights before applying the local jet, but retain all Cartesian
components of equal-shell groups. Stationary first derivatives in inactive
groups are still necessary for the cotangent JVP. No change to the signed
directional API, ordinary SCF, or the spectral metric contract is implied.

The compiler can select the same bounded s/p polynomial jet used for Direct
for full-range fitted batches. The fitted family fixes the existing authentic
unit-Gaussian embedding and still uses the DF Boys evaluator. The 32-lane fold
keeps radial addition order even for more than 32 terms. Slurm 20972 passed
52 raw DF tests, including an independently checked 64-record PPC contraction
over two chunks with signed primitive coefficients. The optimized full-range
dispatch does not change the LR or higher-angular-momentum implementation.

- Job 2693, canonical DF only: 316.15 s native, 151.98 s GPU4PySCF on the
  same RTX 5090; 18,220 MiB peak; `3.075e-8 Ha/Bohr²` raw error.
- Job 20970, canonical plus bounded fitted jets: 299.56 s native, 120.52 s
  GPU4PySCF on the same RTX 5090; 16,771 / 32,607 MiB sampled peak; maximum
  raw error `3.084e-8`, asymmetry `4.04e-12`, translation `6.63e-12 Ha/Bohr²`.
  All 288 columns, zero numerical host fallbacks. This is the superseded
  unmatched-basis comparison. The traced native arm is
  diagnostic and cross-card timing does not isolate the leaf improvement.
  The two builds' raw 96-atom matrices differ by `5.67e-13 Ha/Bohr²`.
- Job 20970 midpoint: fitted RHS 169 ms, fixed/moving grid RHS 145 ms, CPKS
  250 ms, fitted HVP 130 ms, moving grid HVP 176 ms. The first CPKS column
  still spends about 29 s including the full OV operator construction. Do
  not trade the independent physical residual audit for a cached residual.

Fuse the local Cartesian s/p component projection and its serial component
fold within the producer warp. Keep the original component order, coincident
atom incidences and both transpose cotangents. The surviving inter-group
scatter remains deterministic. This removes 15.9 MB of transient global
component-projector storage on 96 atoms; no integral passes are removed.
The 96-atom matched job 20982 completed in 291.57 vs 119.86 s, at
16,755 / 4,703 MiB, with `1.64e-9 Ha/Bohr²` raw error and no numerical
fallback. The compact CPKS/grid/raw DF regression passed 52 cases. Use a
same-device A/B before assigning the timing difference to this change.

Separate generated full-rank Coulomb Fock motion and source-weight motion
where the Hessian consumer requests each individually. The immutable fitting
state and mathematical IR are unchanged. Keep the joint maps for hybrids,
spectral metrics, and admission fallback. On wide dense semilocal r²SCAN DF,
the trial also retains the explicit moving XC Fock and the physically audited
first-order density once per column. Equality of the mixed XC partials gives
`H_relax[L,R] = F_xc,L : D_R`; add this in one private FP64 contraction
before the existing raw dense symmetry and publication gates. The fallback
retains the per-column induced grid covector if the tables or BLAS allowance
do not fit. Arbitrary-direction HVPs do not use this dense table.

The matched-basis run 20988 (RTX 5090, 32,607 MiB) completed the whole
96-atom DF endpoint in 273.44 s vs 118.44 s on the same device. Peak usage
was 16,983 / 4,703 MiB; all 288 directions completed without numerical
host fallback. Raw max difference was `1.64e-9`, asymmetry `4.04e-12` and
translation residual `6.62e-12 Ha/Bohr²`. `grid_right_directions` decreased
from 576 to 288, with 82,944 mixed contractions, 288 fitted value passes,
288 fitted pullbacks and 288 independent physical CPKS checks. This clears
the `2e-8` accuracy gate, not the speed gate.

Job 20990 interleaved the split-map build (baseline) and XC-relaxation
build (candidate) in fresh processes on one RTX 5090. For 48 atoms the
endpoints were 40.50 / 39.31 s (baseline / candidate) and 40.80 / 39.61 s
in reverse order, with peaks of 4,495 / 4,531 MiB. Those samples compare
only the two native builds, not the independent speed target.
For 96 atoms, the corresponding native arms took 283.33 / 273.95 s
(3.31% less elapsed time), peaking at 16,755 / 16,983 MiB. Their raw
Hessians differ by `1.14e-14 Ha/Bohr²`. The paired GPU4PySCF arm took
119.53 s (4,703 MiB) and the candidate passed the raw gate with max error
`1.62e-9 Ha/Bohr²`. One pair at this size is not a speed distribution.
The separately traced candidate spent about 29.1 s in first-column CPKS
including setup. At the midpoint the GPU regions were 146 ms fitted RHS,
145 ms grid RHS, 250 ms CPKS, 104 ms fitted pullback and 142 ms grid HVP.
These are diagnostic region durations, not independent endpoint timings:
putting effort into only one region is unlikely to close the full gap.

The s/p projector uses one relaxed status read per warp in both forward
and transpose modes; a sticky flag is an early-out hint, not a publication
fence. The existing stream gate still rejects a failed producer or projector
before any public output is written.

For a future full-rank Coulomb-only dense source, let `b = B:D`,
`q = M⁻¹ b`, and `s_R = b_R − M_R q` at fixed density. The frozen fitting
response splits into the source-local second derivative
`qᵀ b_LR − ½ qᵀ M_LR q` and the auxiliary-space contraction
`s_Lᵀ M⁻¹ s_R`. The density relaxation is `F_L:dD_R`. This split could
replace the per-column all-group cotangent sweep with a moving-group
second-derivative reduction and a small auxiliary matrix contraction, while
keeping the physical CPKS audit. It does not apply to a moving spectral
subspace or fitted exchange without the corresponding generated adjoints.

### Direct follow-up

Spans alone (20963) still time out at 30 minutes, with 16,413 MiB of sampled
device memory. The paired reference SCF did not converge; the earlier
139.87-s reference remains a separate-run control. For a certified replay
only, a wide window may use available space under its existing batch lease
to retain its complete prepared geometry; the narrow limit and bounded
unprepared execution remain. This candidate still hit the 45-minute endpoint
timeout on job 20971 at 16,507 MiB, not OOM. `cuobjdump` on the associated
SM120 object showed 168–255 registers/thread and up to 2,272 bytes of stack in
the indexed prepared leaves. The compiled-resource report is diagnostic; it
does not explain away algorithmic work amplification. The trial all-s leaf
uses the existing separated affine-coordinate Graph instead of the generic
prepared array path, without changing the translation anchor or FP64 audit.

Slurm 20989 (48 atoms, matched primitives, RTX 5090) completed in 259.08 s
vs 21.93 s, peak 7,609 / 2,571 MiB. The raw error was `1.59e-8`, asymmetry
`3.42e-12` and translation residual `5.39e-12 Ha/Bohr²`. It replayed 5.46B
exact primitive components and took 4,196 induced Fock actions. The matched
96-atom Direct GPU4PySCF reference-only arm (20980) finished in 138.56 s at
4,841 MiB. This is not a same-device paired 96-atom Direct result.

An offline max-density weighting of both Coulomb orbit incidences used the
96-atom Direct GPU4PySCF density from Slurm 20973. Charging each right-atom
ledger at the same normalized `1e-14` left 22.31M canonical sources and 7.29B
primitive components (vs 29.45M / 9.33B unweighted), even before replay. This
is sizing, not a production omission certificate: an authenticated native
density, separate charge bounds for the RHS/response, and end-to-end error
validation would still be needed. Density weighting alone is not a realistic
route to the Direct target. A source-driven second-order local contraction
must reuse source work before the outer directional sweep.

For a full-range pure Coulomb orbit, reuse the compiler's pair-separable
weighted ERI Graph. Project each ordered pair density onto its Hermite
coefficients before combining bra and ket; for the explicit Fock derivative,
differentiate the weighted first derivative with respect to each pair
weight. Fold canonical orbit multiplicities and component normalization at
the pair inputs, including the equal-pair case. This provides both the
stationary second derivative and the two Fock incidences without requiring
an AO quartet tensor. An offline signed-weight prototype compared the
9-by-9 independent-center second derivative and both Fock incidence blocks
against Libcint for all 16 s/p shell patterns, with max differences
`6.47e-15` and `4.16e-15` on the chosen primitive fixtures. It is not a
production schedule: physical-owner aliasing, all contracted primitives,
audited screening and deterministic bounded reductions still need coverage.
The raw Graph has as many as 30,352 live nodes across both outputs for
`pppp`; use physical-owner or Hermite-state tiles instead of materializing
that entire Jet per thread.

The offline SM120 resource trial confirmed that constraint: one thread
materializing the full `pppp` block used 255 registers and 43,168 bytes of
stack. Even one right owner's three columns, with both Fock incidences, used
255 registers and 20,824 bytes of stack. Do not promote either expanded leaf.
A second offline prototype projected the two pair decay/shift jets before
contracting a shared radial derivative prefix, then lifted the 9 affine
coordinates to physical slot owners. All 128 signed-weight checks passed
against independent Libcint derivatives, including coincident centers and
aliasing slot owners. Maximum errors were `5.67e-14` (second derivative)
and `3.47e-14` (Fock incidences). A separate 3-by-3 relative Hessian tile
compiled with at most 38 registers/thread; that figure excludes radial
generation, reciprocal Fock and physical accumulation. Keep those stages
separate when measuring memory traffic and complete endpoint profitability.

The standalone SM120 cubins also passed real-device tests of the shared
radial prefix plus all nine ordered 3-by-3 tiles, and of both reciprocal
Fock incidences against separate array contractions. Max GPU errors were
`3.91e-14` and `3.56e-15`, respectively. Both Fock incidence leaves used at
most 38 registers/thread. These are synthetic checks, not molecule timings:
production needs a bounded pair/geometry lease and in-window reductions.
Do not write one 81-field Hessian or both AO incidence panels per *primitive
product* to global memory. Fold primitives into shell/physical-owner tiles
before writing, then emit only the sparse owner entries needed by the
existing dense-column reducer, with the translation anchor and sticky audit.

The post-format scoped build passed 423 tests (16 skipped by inventory or
optional molecular probe selection) on Slurm 20993, including the raw
fitted nuclear tests, CPKS, Direct J/K and moving-grid regressions. An earlier
deployment failed before execution because its loader path omitted the
staged `libgenerativeqc.so.0`; the repeated suite used the same binary with
the correct library path. The 48-atom Direct endpoint on that build then
passed again (`1.59e-8 Ha/Bohr²` raw max error, 259.52 / 22.13 s,
7,607 / 2,569 MiB). The 96-atom matched Direct job 20994 uses the same
build, a finite 80-minute wall budget, and the unchanged numerical gates.

### Source sweep and audited response checkpoints

Job 21004 used the full-rank Coulomb split above: 244.98 s native versus
119.99 s GPU4PySCF, at 16,987 versus 4,703 MiB. The raw max error was
`1.63e-9 Ha/Bohr²`. Retain the complete source-weight path outside the
full-rank dense Coulomb contract. An optional active-group list also lets
the fixed-weight pullback skip inactive producer warps. Its original
ordered scatter slots remain zero-filled and its general-direction and
allocation-refusal paths still enumerate the complete window. Job 21013
passed the matched 96-atom raw gate; it did not isolate the sparse change
in a same-card native A/B. Job 21014 passed 328 focused cases, with 12 skips.

The Cholesky factor of the admitted FP64 occupied-virtual operator supplies
an initial guess, not the authority for convergence. Keep the exact cached
operator and original strict GMRES and physical Fock audit. An optional
factor must have symmetric input, positive pivots, a bounded independent
copy and explicit aligned solve scratch. Job 21007 failed with CUDA error
5 before the explicit BLAS scratch retry. Job 21009 accepted all 288 seeds
with zero further GMRES iterations and 288 independent physical audits.
With the generated moving-potential transform lowered as `X + Xᵀ` instead
of contracting both legs separately, the complete 96-atom DF comparison
was 221.32 vs 118.92 s at 17,887 vs 4,703 MiB; maximum raw error `1.64e-9`,
asymmetry `4.20e-13` and translation residual `4.02e-13 Ha/Bohr²`.
No numerical host fallback. This passes accuracy, not speed. That binary's
midpoint spent 143/77/210/31/140 ms on fitted RHS, grid RHS, CPKS, fitted
pullback and grid pullback respectively; first-column CPKS took about 29 s.

The frozen-density mixed grid transpose can reuse resident `P_s A` and
form `P_s dA` as four bounded FP64 BLAS tiles, where `P_s=(P+Pᵀ)/2`. The
generated epilogue uses the same checked rational AO transpose coefficients
as the fused dense build. Consume every input jet of a tile before replacing
its right AO values; a missing cache, checked call, or density-JVP term
retains the general fused path. Neither SCF nor the physical CPKS audit uses
the new lowering.

For Direct, diagnostic job 21012 replayed all 29,450,901 certified groups
against the actual converged 32-water density. Six contracted shell-pair
kernels and the FP64 atomic molecular scatter took 54.23 s in aggregate
GPU time (not an endpoint). 432 local Libcint-backed assembly cases across
those classes agreed within `5.0e-16`. Full-table asymmetry and translation
were `1.42e-14` and `9.12e-13 Ha/Bohr²`; the frozen Fock translation
residual was `6.89e-14`. The equal-pair test caught a metadata pointer
alias: select source side by field, not by pointer equality. This experiment
does not certify the omission error or the full Direct Hessian. A production
sweep needs compiler-owned CUDA, deterministic sparse owner folds, linked
resource bounds and the existing certificate and sticky publication checks.
Retain the established path for unsupported shapes/providers and do not
infer a speed win from these component timings.

### Wide exact response and the remaining gap

DF job 21018 completed the matched 96-atom endpoint in 181.27 s vs
121.64 s, peaking at 17,887 vs 4,703 MiB on one 32,607-MiB RTX 5090.
All 288 independently audited columns passed: raw max difference
`1.64e-9`, asymmetry `2.27e-13`, translation `1.81e-13 Ha/Bohr²`, and
no numerical host fallback. The profiled middle column used 101.4 ms
for fitted RHS, 78.6 ms for moving-grid RHS, 212.9 ms for CPKS (two
physical Fock actions), 31.3 ms for fitted HVP and 49.6 ms for grid HVP.
The first CPKS region was 29.46 s including operator construction. The
packed forward JVP is not independently qualified by this trace alone.

The Direct raw matrix from job 21006 was still `2.45e-8 Ha/Bohr²` away
from the tightened GPU4PySCF SCF reference from job 21029. The two
reference matrices differ by only `6.66e-10`. This cross-run diagnostic
does not establish an accuracy pass or a same-card speed comparison;
it rules out the looser reference SCF as an explanation of the miss.

For pure-Coulomb RKS, retaining the exact canonical pair rows can avoid
replaying quartets during physical CPKS. Replace the old 96-AO width
restriction with the existing checked row-byte and product caps; hybrid
exchange keeps the old policy. A dense candidate factors the projection
as two AO-to-OV transforms instead of multiplying a dense pair-by-OV
transition panel through the four-index matrix. The first half keeps
one packed AO triangle per OV rotation; the second adds the RKS factor
of four. Both halves count submitted products, and the arena, input
identity, and owning stream are checked. The projected operator and its
optional Cholesky guess are not the convergence oracle: a full physical
Fock residual still audits every column. Missing rows, scratch, or XC
linearization keep the bounded independent physical solve.

The 48-atom trial (job 21027; 112 AOs) admitted 320,753,664 pair-row
bytes, applied 11,467,653,120 Direct projection products, completed
144 independent physical residual checks with 288 Fock actions, and
took 61.16 s including SCF, peaking at 8,249 MiB. This was a preliminary
binary and has no paired timing claim. The candidate repeats the MO
frame in its small transform tile rather than relying on provider-specific
zero batch strides. Blocking the first occupied index reduces the 224-AO
intermediate from 2,064,384,000 to 206,438,400 bytes, before tile scratch.
It adds bounded rereads, not repeated GEMM terms. An unused CPKS matrix is
released if the exact transition builder does not obtain its lease.

Job 21022, with only wide pair-row admission, completed in 1,433.68 s vs
138.28 s. It retained 5,083,545,600 bytes of rows and sampled 23,783 MiB
out of 32,607 MiB; no OOM or host fallback. The 8,480 physical actions
and the `2.41e-8` cross-engine error remain. Source preparation took 348 s
and the column loop 944 s, besides SCF/owner preparation. This result
does not qualify a larger stock cap for hybrid response.

Reference diagnosis: job 21036 tightened the integral cutoff from
`1e-14` to `1e-17`; the two tight-SCF Direct references agree to `1.89e-11`.
Only raw entry `(261,261)` in the zero-based 96-atom matrix exceeds `2e-8`
against the tightened reference. Job 21037 differentiated analytical
moving-grid gradients with centered differences and fourth-order
Richardson extrapolation at steps of 0.002, 0.001 and 0.0005 Bohr. The
extrapolants agreed at that entry to `2.92e-10`. The finer result differs
from the native entry by `3.67e-10`, and from the analytic GPU4PySCF entry
by `2.42e-8`. Over the full column, the native error vs extrapolation is
`6.74e-9`. This supports a reference-side discrepancy, but is not an
independent full Hessian or permission to move the cross-engine gate.

Rejected wide fixed-grid schedules: restricting the FP64 GEMM to one
AO triangle passed the independent grid/endpoint accuracy gates, but
took 297.07 s vs 181.51 s on the same card in job 21030. Keeping the
full GEMM and coalescing AO tile loads with a shared transpose likewise
failed to help in job 21032: 179.27 s vs 179.36 s, against 120.45 s for
GPU4PySCF. Both private grid and physical CPKS remained accurate. Neither
experiment was retained; target the source visits and physical actions
instead of rearranging GEMM staging alone.

### Retiring duplicate Direct storage

The wide pure-Coulomb response owner reconstructs and checks the converged
Fock before it requests the repeated-RHS rows. Once those rows are complete
and fenced, both the AO→OV projection and the independent physical action
can use them without the canonical value table. Retire that duplicate on
the private induced-response plan only. Track live bytes at the provider and
Hessian owner, without resetting the completed source evaluations. The
224-AO inventory drops 2,540,260,800 live bytes after publication; both
allocations are still needed during conversion. Smaller and hybrid plans,
SCF and the frozen/displaced nuclear plans are unchanged. Row-budget refusal
leaves the original exact plan intact.

Job 21048 passed all three Libcint-backed response-row cases, including the
104-AO case after retirement and capacity/product refusal before publication.
Job 21040 passed 425 scoped GPU cases (16 skips) and a separate three-case
row test on the preceding retained build. Host queue, budget, comparator and
fitted compiler checks passed 30 tests with retirement enabled.

The untraced DF arm of job 21047 measured 180.88 s vs 119.56 s (GPU4PySCF
1.9.0), 17,963 vs 4,651 MiB peak, raw error `1.64e-9 Ha/Bohr²` and 288
physical checks with zero host fallback. All 288 Cholesky proposals
were accepted without a further GMRES iteration. Reducing GMRES work alone
cannot close this DF gap.

The Direct Cholesky proposal previously selected the BLAS context prepared
by the DF relaxation fold. Direct does not prepare that fold, so the factor
may now acquire a stream-bound BLAS context explicitly under its own budget.
The immutable operator and independent physical residual remain mandatory;
admission refusal keeps GMRES. Job 21049 traces the Direct certificate,
window inventory, numerical bind, and factor use separately from untraced
timing. Neither the handle fix nor table retirement constitutes a speed win.

### Contracted Direct frozen sweep

For certified dense-only Coulomb replay with Cartesian s/p and exactly three
primitives per shell, prepare weighted and unweighted Hermite pair jets on the
authenticated device, then visit the certificate once in bounded pair-class
windows. The weighted legs include the public AO projection factors; the
unweighted Fock leg applies them at publication. Deterministic atom-pair and
atom/shell-pair owners fold both incidences, including a half factor for a
diagonal canonical quartet. The sticky density, finite, and symmetry gates
still run before the frozen table is accepted. The workspace and semantic work
bounds remain finite; unsupported topology and a plan-capacity refusal retain
certified legacy replay. This path is not available to arbitrary HVP, hybrid
exchange, or ordinary SCF.

Job 21054 independently checked 432 Libcint-backed entries across all six
shell-pair classes (maximum absolute difference `4.45e-16`) and traversed all
29,450,901 certificate hits. Its 69.9 s device time and 25.7 s host inventory
time are component diagnostics, not complete Hessian timing. The production
integration replaces the expanded legacy eightfold inventory with a bitset
count plus stable bounded owner folds. Job 21058 passed 68 native J/K cases
(4 skips), including the contracted frozen table, a legacy replay comparison,
and failure on out-of-bound density or nonzero density motion. Host comparator,
fitted compiler and budget checks passed 29 cases.

Job 21056 finished native Direct in **353.04 s**, down from **660.97 s** on
job 21047 (different card of the same model). Its 23,197 MiB sampled peak did
not exhaust the 32,607 MiB card. It visited 9,328,461,147 primitive Cartesian
equivalents, did 288 independent CPKS checks and 576 induced Fock actions, and
completed all 288 columns with zero numerical host fallback. The 21056
GPU4PySCF Direct arm failed the reference SCF/basis gate, so that job is not a
paired Direct comparison. Comparing saved matrices on identical input hashes,
the two native results differ by `3.20e-12 Ha/Bohr²`. The old GPU4PySCF matrix
(139.11 s in 21047) still differs by `2.45e-8` at `(261,261)` only. The new
native raw asymmetry and translation residual are `1.42e-14` and `1.64e-13`.
The prior gradient-difference diagnostic remains relevant, but does not waive
the independent accuracy gate.

Job 21056 **did** complete the DF pair: 179.87 vs 119.90 s and 17,887 vs
4,703 MiB, with `1.64e-9 Ha/Bohr²` raw maximum error. It preserved full
auxiliary response and all 288 physical checks. A 4,096-point fixed-grid tile
trial (21059) took 180.41 s on its native DF arm and used 17,999 MiB. It did
not close the gap; retain the smaller tile. A host-only 96-atom test of
inlined binary64 outward stepping reproduced the original certificate
census, bound and checksum, but took 28.10 vs 28.01 s. Sorting and ledger data
movement, rather than `nextafter` calls alone, need investigation before
changing certificate arithmetic.

## Contracted fitted source and Direct triangle

The contracted Coulomb leaf now emits 45 local mixed derivatives instead of
81. It still folds every ordered physical atom incidence into the raw
matrix; it does not average the assembled output. Wide dense-only Direct
can also retain the explicit XC Fock derivative and induced density in the
same bulk relaxation fold as DF. Reserve the CPKS factor first; a missing
table retains the old grid covector. Job 21063 measured 303.52 s and 23,559
MiB. Its native matrix differs from job 21047 by `3.23e-12 Ha/Bohr²`.
In trace 21065, the frozen sweep, grid RHS, grid HVP and CPKS took 68.85,
22.53, 14.23 and 93.14 s, respectively. The forward Direct J builds also
took about 28 s and the derivative certificate about 26 s.

The GPU4PySCF Direct SCF energy drifted by up to roughly `2.6e-10 Ha` per
iteration even after its orbital gradient reached `1e-10`. A larger DIIS
space alone did not fix that. Rebuilding J rather than incrementally
updating it, with DIIS space 12, met the same `1e-13` energy and `1e-9`
gradient targets. The new paired job 21070 measured **288.52 vs 138.99 s**,
**23,411 vs 5,837 MiB**. The raw cross-engine maximum remains `2.45e-8`,
so Direct fails both the speed and accuracy gates. Independent gradient
finite differences diagnose the same coordinate but are not a waiver.

For full-rank, pure, full-Coulomb Cartesian s/p RKS fitting, borrow the
accepted `q` and sweep the source once. A constant partner reduces the
primitive products to 27 for three-center and 9 for metric jets. The
standalone derivatives passed 800 independent Libcint contractions (max
`1.51e-14`). Stable owners fold `qᵀb_LR − qᵀM_LR q/2`, `A_R q`, and
`s_R = b_R − M_R q`, without allocating a raw nuclear derivative per column.
Complete the fitted solve with `Wᵀ(W s_R)`, add `A q_R` to the explicit
Fock motion, then keep both relaxation folds and the physical CPKS check.
The angular, representation, rank, work and workspace gates restrict this
optional schedule, not the endpoint. An inadmissible or unavailable sweep
retains the raw replay; failures after launch veto publication.

In DF job 21064 the complete paired endpoint took **153.32 vs 118.82 s**
and **18,101 vs 4,703 MiB**. It completed all 288 columns and 288 independent
CPKS checks, with 576 physical Fock calls and zero numerical host fallbacks.
Raw max error/asymmetry/translation residual were `1.64e-9`, `2.19e-13`,
`4.41e-13 Ha/Bohr²`. The maximum native-to-native change is `5.26e-13`.
Trace 21068 counted 2,073,680 source
groups, 1.86 s for the fitted sweep and 9.47 s for all frozen sources; the
legacy DF value and pullback counts both fell from 288 to zero. The first
CPKS solve still includes roughly 29 s of transition operator construction.

Do not prioritize exact-zero grid compaction for this geometry. An independent
GPU4PySCF grid census found only 1,137 of 82,944 points with exactly zero
weights. Magnitude-based AO and weight screening offers more potential, but
requires a new complete force/response error certificate and independent
qualification, not removal of small nonzero terms by default.

Factoring only the fixed-grid feature evaluation through occupied orbitals
did not pay off at 160 occupied out of 224 AOs. Job 21072 ran in 153.28 s,
versus 153.32 s for the preceding native DF binary on the same card, and
raised sampled peak usage from 18,101 to 18,507 MiB. Revert that experiment;
it still formed the full AO Fock and did not reduce transition construction.

Doubling the transition tile and replacing the MO-jet kernel with four
DGEMMs did not improve DF (job 21075, 154.03 s); keep the existing tile and
MO-jet kernel. An experimental CUDA 13 cuBLAS relink using FP64 emulation
passed the DF numerical gates at 142.10 vs 118.96 s; eager execution took
144.46 s. This diagnostic is not a stock dependency or an endpoint win.

## Direct Certificate Ordering

Sort candidates by nonnegative binary64 exponent first, then by the same
exact bound and lexicographic shell-pair ordinals. For a shared owner the
combined quartet bound dominates the two single-pair bounds; compute each
unique owner's outward-rounded ledger update once and commit only when all
fit. In an affordable canonical rank space, mark accepted ranks and publish
them in order; sparse, large rank spaces keep the bounded output sort.
Neither the budget, candidate checks, nor omission order changes.

The 96-atom host-only A/B returned the same 64,148,639 candidate checks,
53,502,739 omitted canonical groups (ordered checksum `8c6e66c972635634`),
and `9.9999994362247406e-15` bound. Wall time fell from 33.57 to 18.52 s and
peak host RSS from 6,022,360 to 3,388,228 KiB. These are certificate-only
measurements, not GPU endpoint timing or VRAM measurements.

Job 21083 completed the Direct native endpoint in 268.81 s at a sampled
23,411 / 32,607 MiB, with all 288 columns, 288 independent physical CPKS
checks, 576 physical Fock calls and no numerical host fallback. Its maximum
change against the input-matched 21070 native matrix was `2.48e-13`;
asymmetry and translation residual were `1.63e-14` and `1.60e-13`.
GPU4PySCF did not meet the tight Direct SCF gate on the 21083 card. This is
not a new paired Direct benchmark; against the previous successful
reference the maximum error is still `2.45e-8 Ha/Bohr²`.
Job 21089 ran the unchanged reference procedure first and completed both
Direct arms: **270.98 vs 139.94 s**, **23,411 vs 5,779 MiB**. Raw maximum
error was `2.44954e-8`, asymmetry `2.37e-14`, and translation residual
`1.57e-13 Ha/Bohr²`. The native source regression (21090) passed 122 tests
with 8 build-inventory or companion-probe skips.

DF job 21084 completed both arms in 154.03 vs 120.11 s, sampling 18,165 vs
4,639 MiB. Raw max error was `1.66e-9 Ha/Bohr²` and both structural gates
passed. The full-rank Cartesian source-sweep and bounded raw replay probes
also passed against the independent Libcint fixture. Neither path has met
the complete-endpoint speed target.

## Sparse CPKS Metric Features Pilot

For a single Cartesian displacement, overlap motion is supported on the
owning atom's AO rows and columns. Write `S' = E U + Uᵀ Eᵀ` with `E` selecting
owner rows (halving the owner-owner block in `U`). The RKS metric density
is `−P S' P/2`. Given frozen `P A` for each of the four grid jets, each
metric-feature bilinear is the sum of two products between `Eᵀ P A` and
`U P A`. This is a rank factorization, not magnitude screening. The same
operator and independent final physical residual are still required.

A synthetic shape-matched FP64 GPU pilot (21087; 224 AOs, 160 occupied,
82,944 points) measured 21.96 ms for dense metric features, versus 0.73 ms
for one owned AO and 3.03 ms for five, with max errors under `1.7e-16` on
this fixture. The 594,542,592-byte frozen table is charged even if an
implementation can borrow a compatible existing lease. Feature savings
alone extrapolate to only about 6 s for 288 columns, so the pilot is not
a sufficient speed fix. Revisit together with batching the metric RHS
projection into the transition build; do not bypass the final physical
residual or apply the sparse identity to arbitrary directions.

Job 21095 spent 1.68 s in 648 shape-matched FP64 GEMMs and output adds for
that extra projection, with a 23,592,960-byte rotation-by-coordinate
output. This excludes real metric-feature packing, XC differentiation,
source authentication, and CPKS. A candidate could keep the first column's
ordinary action, add the metric projection to the existing transition build,
then use it only for the initial metric RHS of subsequent columns. The
independent final physical operator, weighted-density reconstruction, and
per-column acceptance remain unchanged.

The isolated cuBLAS 13.10.1.6 relink (21093) measured **137.73 vs 119.63 s**
and **19,691 vs 4,703 MiB** for a complete DF pair, with `1.63e-9` raw max
error and both structural gates passing. Matched-shape GEMM microbenchmarks
(21092) did not show a standalone fixed-point speedup. This remains a
diagnostic: it neither beats the reference nor qualifies a stock toolchain
switch.

## Batched metric projection

Use the full RKS metric density `−(K S' P + P S' K)/2`, where
`K=C Cᵀ` includes the virtual frame and `P=2 Cocc Coccᵀ`. The occupied-only
formula from the sparse feature pilot is not the CPKS metric density. Pack
the nonzero overlap rows of each frozen atom axis, and project their grid
bilinears against the weighted transition tangents while building the OV
operator. Only the initial RHS can borrow this projection. The final physical
Fock reconstructs both the residual and weighted-density response, and a
failed residual cannot publish a column. Unsupported topology, correction
points, or a refused lease keep both original physical XC actions.

The 96-atom traces contain zero transition correction points, 1,223,059,046,400
additional scalar projection products, and 287 initial metric projections
in the fully traced Direct arm. The 576 induced-Fock calls now include 287
complement-only RHS calls; the final 288 physical checks are unchanged.
Job 21100 passed all 126 CPKS tests, including correct, wrong-overlap and
corrupt metric projections.

- DF job 21096: **126.64 vs 119.01 s**, **18,135 vs 4,703 MiB**. Raw error
  `1.637e-9`, asymmetry `2.21e-13`, translation `4.40e-13 Ha/Bohr²`.
  The native matrix changes by `5.33e-15 Ha/Bohr²` against the DF baseline.
- Direct job 21099: **243.40 vs 141.00 s**, **23,619 vs 5,255 MiB**. Raw
  error `2.451e-8 Ha/Bohr²` still fails qualification. Asymmetry and
  translation are `1.36e-14` and `1.56e-13 Ha/Bohr²`. In the separate
  traced job 21097, the Direct frozen source sweep took 68.83 s; the
  response columns took 103.15 s including the transition build.
- Relinking this candidate against cuBLAS 13.10.1.6 (21102) did not close
  the DF gap: **126.89 vs 119.26 s**, **18,527 vs 4,703 MiB**, raw error
  `1.62e-9 Ha/Bohr²`. Do not switch the default CUDA dependency on this basis.

### Integer tensor projection pilot

Job 21103 used radix-128 signed digits and INT32 accumulation on synthetic
`5120 × 5120 × 640` transition tiles. Concatenating the digit products by
equal exponent kept each integer sum bounded. Eight diagonals took 5.06 ms
vs 19.63 ms for DGEMM, including split/pack/reconstruction, with maximum
absolute difference `2.67e-14`; six diagonals gave `6.94e-12`. These are
single-rectangle diagnostics, not complete endpoint or scientific evidence.
The molecular candidate must include all triangular blocks, individual
rotation-column exponents and the independent, uncompressed FP64 audits.

### Integer tensor endpoint

The complete builder normalizes each rotation column by a binary exponent,
then applies the same product to eight radix-128 diagonals. The largest
INT32 sum has at most `8 * 640` terms of magnitude at most `64²`. A capped
workspace and SM12x shape gate keep the original DGEMM builder available;
if range validation or library submission refuses, discard the partial
operator and rebuild it with DGEMM. Physical density, reconstructed Fock,
the final residual and the published Hessian never use integer storage.

Job 21105 completed DF **107.01 vs 119.50 s**, **18,505 vs 4,635 MiB** of
32,607 MiB, with all 288 columns and physical checks and no numerical host
fallbacks. Raw error `1.638e-9`, asymmetry `2.18e-13`, translation
`4.40e-13 Ha/Bohr²` pass. The native matrix changed by `5.33e-15 Ha/Bohr²`
against the batched-metric FP64 builder. The trace counted
821,895,679,180,800 INT8 scalar products (36 per corresponding triangular
logical scalar product), not a reduction in logical transition visits.
Job 21107 passed 127 CPKS tests, including independently accumulated signed,
differently scaled panels with a shorter final tile and extreme-range refusal.

Direct 21106 completed native in **222.32 s** at **23,755 MiB**; the
GPU4PySCF Direct arm did not pass its SCF gate, so this is not a paired
qualification. Against the previously input-matched 21089 reference its raw
error stayed `2.450e-8 Ha/Bohr²`, and the native matrix moved only `3.25e-13`
against the preceding batched-metric Direct build. The overall Direct + DF
goal remains open. PR #3141 comment `7231972162` records the DF pass.

### Coordinate-specialized Direct sweep and certified value omission

Job 21109 completed the raw Direct endpoint in **216.04 s** at **23,755
MiB**. The traced arm agrees to `2.36e-13 Ha/Bohr²`; its frozen-source
sweep took 62.42 s, down from 68.67 s before coordinate specialization.
Response columns still took 82.58 s (`45.75` CPKS, `22.45` moving-grid
RHS, `14.20` grid HVP), and host certificate construction took 14.91 s.
Against the previously input-matched Direct reference, the raw error is
`2.45e-8 Ha/Bohr²`, still outside the gate. Job 21111 independently passed
the DF pair in **107.03 vs 120.49 s**, with `1.64e-9 Ha/Bohr²` raw error.

The optional value certificate only applies to the private wide, dense-only,
pure Direct RKS response plan. The canonical Cartesian source packs AOs in
shell order, and s/p component normalization factors equal one. The
outward-rounded zeroth-order shell-pair envelope uses a common component
majorant multiplied by the ordered component count, so dividing that count
back out with upward rounding bounds each public AO pair. Schwarz then limits
each omitted integral to `2^-70`. For any signed density, the induced error
in one J entry is at most `2^-70 ||D||_1 <= 2^-54 ||D||_max` at the admitted
maximum of 256 AOs. This is an operator error budget, not a bound on the
total Hessian error; the raw independent gates still apply.

Keep the original canonical slot count and initialize omitted slots to
zero. Count evaluations with the same binary64 predicate using sorted
norms, rather than dividing by the number of CUDA threads or reporting
slot capacity. Unknown topology or optional-resource refusal keeps exact
replay. The ordinary CUDA kernels, SCF, and nuclear derivative source do
not consume this certificate. A host census predicts 216,110,909 omissions
out of 317,532,600 for this water model; it predicts neither a GPU runtime
nor a memory saving. An independent Libcint test verifies component bounds
and a nonzero signed Coulomb error with integral screening tightened.

The initial admission incorrectly excluded every sorted canonical table;
zero-threshold sorting does not remove slots. Job 21116 confirmed the full
317,532,600 evaluations. Same-card jobs 21114 and 21115 measured **223.27
vs 139.43 s** for Direct, and **106.82 vs 119.36 s** for DF. Direct still
fails with `2.452e-8 Ha/Bohr²`; DF passes at `1.640e-9 Ha/Bohr²`.
The corrected admission checks the complete slot inventory. The other
candidate change uses two capped pinned staging pages for contracted
window metadata to overlap CPU incidence construction with GPU work.
It preserves single-stream order and reuses a page only after its event;
pinned-capacity refusal retains the original fenced pageable copies.

The existing density-weighted pilot leaves 7.29B primitive equivalents
at the same normalized `1e-14` budget (9.33B unweighted). Even eliminating
those evaluations does not close the measured gap on its own; do not trade
the raw comparison gate or independent residual for an estimated speedup.

Job 21117 counted **101,421,727** retained value evaluations, down from
317,532,600. The native endpoint was 212.32 s with 23,811 MiB sampled peak;
it differed from the unscreened native matrix by `2.58e-13 Ha/Bohr²` and
still differed from the input-matched reference by `2.452e-8 Ha/Bohr²`.
The separate trace took 206.89 s, with 59.96 s frozen and 83.06 s response.
Job 21119 passed 68 native nuclear cases (4 skipped). The slot layout
still allocates the same number of device bytes; this is a work reduction.

### Reuse half-contracted primitive jets across coordinates

In the large Direct quartet leaves, the same Hermite-state sums were
recomputed per Cartesian output. First contract the right weighted jet
against the radial jet in independent pair/separation coordinates. Reuse
those value, first and mixed fields for the Hessian and left Fock incidence.
Contract the opposite orientation to first order for the right incidence.
Only the final product composes these with the physical center tangents.
Keep the compiler-owned radial recurrence and product AD. The sliced
radial and half-jet buffers fit the prior 54,432-byte shared limit;
zero-degree ket and fitted leaves keep the original contraction tree.

The same-card row-local reassociation trial (21120) regressed from 208.37
to 216.92 s; its reference SCF failed. The trial is not retained. Increasing
block width after early radial stores and grouping several small quartets
per block also offered no material microbenchmark win. In contrast the
bounded half-jet pilot reduced elapsed device time for 4,096 groups from
17.47 to 5.31 ms (1/1), 57.65 to 15.72 ms (2/1), and 171.59 to 25.63 ms
(2/2), with independent Libcint absolute errors no larger than `1.56e-15`.
These isolated tiles do not establish a complete endpoint speedup.

Job 21127 measured **201.01 s** with the half contraction versus **216.33 s**
for the preceding native binary, both at **23,755 MiB** on the same card.
The separate traced arm spent 47.97 s in the frozen sweep and 82.58 s in
response columns. The raw native A/B agrees to `5.89e-13 Ha/Bohr²`. Job 21128
passed 68 independent nuclear tests (4 skipped). The GPU4PySCF arm in 21127
failed strict SCF, so this is not a paired cross-engine result.

### Outward-rounded certificate construction

The finite nonnegative certificate arithmetic can obtain the adjacent binary64
encoding directly, including subnormals, instead of calling libm after each
primitive operation. Preserve the original result for signed-zero and nonfinite
inputs, and keep the same column budgets, candidate ordering and record layout.
The 96-atom host A/B reproduces the exact bound, candidate count and canonical
source-list checksum in 13.99 s versus 18.58 s. The 17 host tests include
independent Libcint comparisons and `nextafter` boundary equivalence. A separate
compact-record attempt reduced host RSS from 3.35M to 1.97M KiB but increased
time from 18.59 to 19.59 s; it is not retained.

Diagnostic job 21137 lowered the GPU4PySCF AO-screen threshold from `1e-10`
to `1e-15`. It still differed at raw `(261,261)` by `2.452e-8 Ha/Bohr²`,
the sole entry outside the gate. This does not qualify either endpoint.

### Aggregate reserve for negligible source bounds

The dense certificate already orders right pairs by their dominant bound.
Before constructing sortable quartet records, find the prefix whose both-owner
bound fits a uniform tiny threshold. Use the maximum of the three right-pair
orders for every term in that test; monotonicity and the both-owner bound cover
every affected column, including shared atoms. The full triangular rank count
bounds the number of tiny omissions. The outward-rounded sum of the accepted
prefix reservations is charged to every atom before the usual per-owner ledger.
Keep the same hard column budget, canonical source order, bitmap ceiling and
explicit census charges. Without a bitmap or room for a prefix search, use the
existing per-source path.

The 96-atom host comparison reproduced the 53,502,739-source list and its
checksum. Its bound changed conservatively from `9.9999994362e-15` to
`9.9999996771e-15`, below the unchanged `1e-14` allowance. The candidate
census, including searches, is 64,256,060 versus 64,148,639 previously.
Time and host peak RSS were 7.66 s and 1,685,924 KiB, versus 13.99 s and
3,388,236 KiB. The 25 certificate/oracle and benchmark host tests passed.

Job 21160 completed both Direct endpoints at **181.28 vs 141.60 s** and
**24,013 vs 5,321 MiB**, using the experimental cold SCF resident source.
It still misses the raw gate at `2.450e-8 Ha/Bohr²`. Do not call this a speed
or accuracy win or infer that the SCF source is ready to enable by default.
Job 21161 attributes 50.09 s to the frozen sweep, 22.62 s to moving-grid
RHS, 14.28 s to moving-grid HVP, 29.01 s to physical CPKS grid actions and
11.56 s to the one-time projected seed. A single faster kernel is not the
remaining end-to-end fix.

At the outlying entry, the fixed-density GPU4PySCF frozen analytical J matches
sliced Libcint to `2.80e-11 Ha/Bohr²` (21159). Its first nuclear J derivative
at the same atom differs from independently contracted Libcint by at most
`1.50e-11 Ha/Bohr` at the stock Hessian cutoff, or `1.88e-13 Ha/Bohr` at the
tighter diagnostic cutoff (21166). The frozen moving-grid XC column agrees
with its fixed-density Richardson gradient derivative to `7.09e-12 Ha/Bohr²`
(21167). Keep these fixed-density checks distinct from the relaxed matrix
gate and the fresh benchmark.

The reference CPKS solve explains the outlier. Using one atom and the same
GPU4PySCF operator, lowering the Krylov dependence floor changed its diagonal
relaxation by `-2.45683e-8 Ha/Bohr²` (21168). Applying the same change to the
full block was unstable: job 21169 had a `1.66e-4` true preconditioned residual
and `3.19e-6 Ha/Bohr²` matrix error. Do not use that arm as a reference.
The per-RHS residual correction instead retains the stock solver as an initial
guess and audits the complete response against the same GPU4PySCF operator.
Job 21172 reduced the maximum preconditioned residual from `1.96e-6` to
`7.89e-13`; its full matrix differs from the input-matched native run by
`5.85e-10 Ha/Bohr²`, with `7.52e-11 Ha/Bohr²` at `(261,261)`. The correction
is a validation diagnostic, not part of the stock GPU4PySCF speed baseline.

Job 21164 measured the prefix-reserve native build at 181.81 s vs 187.63 s
on the same RTX 5090, both at 24,113 MiB and with native A/B raw difference
`5.11e-15 Ha/Bohr²`. Its stock Direct reference did not converge. The same job
completed DF in **110.40 vs 120.55 s**, **18,545 vs 4,675 MiB**, with full
auxiliary response and `1.62e-9 Ha/Bohr²` raw error. The separate frozen-owner
metadata consolidation was not retained: 170.20 vs 168.20 s in 21170, at
identical peak memory, despite passing 68 nuclear tests (4 skips, 21171).

The contracted native owner now consumes the validated canonical-rank bitmap
without expanding it into 53 million four-shell records. Check the census,
tail bits, count and arena limits before building the complementary retained
bitmap. Keep the sorted-list path for simpler callers and bitmap refusal.
Job 21175 passed 68 independent/malformed-lease nuclear tests (4 skips); the
host checks include decoding the ranked result to the same source list.

Job 21174 repeated the residual-audited Direct comparison on one card:
**170.95 vs 262.43 s**, **24,107 vs 6,385 MiB** of 32,607 MiB, and
`6.48e-10 Ha/Bohr²` raw error. Asymmetry and translation residual were
`1.94e-14` and `1.77e-13 Ha/Bohr²`. The audited reference residual was
`7.89e-13` after correction. This is a same-card pass at audited accuracy,
not a claim of beating the unmodified GPU4PySCF 1.9.0 Direct runtime.

The moving-potential INT8 dyadic pilot (21176; 81 tiles × `224 × 224 × 4096`)
took 27.73 ms vs 66.35 ms for FP64 GEMM, with relative maximum difference
`1.31e-15`. This isolated product is not a scientific endpoint. The opt-in
trial confines it to private moving RHS assembly; stock CUDA, SCF, fitted
response, and physical CPKS operator evaluation retain their normal paths.

### Wide-grid BLAS workspace and reference audit

The moving RHS radix pilot regressed end to end (174.11 vs 170.12 s in
21177). The explicit FP64 rank-slice experiments also failed the physical
CPKS gate before publishing the first 96-atom column (21182, 21187, 21189).
Sampled rank-slice GEMMs matched an independent CPU product; neither the
microbenchmark nor this sample establishes complete response correctness.
Do not retain either handwritten grid contraction.

The short-wide FP64 contraction was provider-workspace limited, not a
compiler optimization issue. In the CUDA 12.9 `224 × 224 × 4096` probe,
81 pedantic FP64 products took 66.08 ms with no workspace, 24.07 ms with
4 MiB and 23.36 ms with 32 MiB. Use the existing per-handle 4 MiB optional
workspace inside the existing 96 MiB provider allowance. Workspace
refusal retains the original same-stream FP64 path. Ordinary SCF and the
physical residual's independent assembly algorithm are unchanged.

The isolated cuBLAS 13 diagnostic in 21180 reached 145.63 s Direct vs
175.85 s on the native baseline, differing by `5.03e-15 Ha/Bohr²`.
It agreed with the residual-audited oracle to `6.48e-10 Ha/Bohr²`.
The fitted pair reached 86.48 vs 120.71 s with `1.62e-9 Ha/Bohr²` raw error.
This diagnostic does not add a cuBLAS 13 package dependency. The stock
Direct arm in that job did not converge; the completed stock arm in 21182
took 140.89 s.

The private wide dense Direct owner can defer SCF Fock reconstruction until
the certified Coulomb rows are admitted. Reconstruct `H + J + XC` with the
prepared response operator and compare against the authenticated converged
SCF Fock before any action; when rows are refused, still perform the
canonical reconstruction. Numerical disagreements fail rather than retrying
with looser gates. The public single-action route keeps eager auditing.

### Stock timing and independent accuracy

Reserve at most `1/16` of the unchanged `1e-14` per-atom omission budget
for the globally charged tiny prefix. The host probe dropped from 6.53 to
4.75 s while omitting 53,475,250 instead of 53,502,739 groups. This trades
some GPU work for less host sorting; do not interpret it as a looser bound.
The paired native control in 21195 took 132.35 s against 131.68 s for the
larger reserve, with a `1.23e-13 Ha/Bohr²` raw matrix delta.
With deferred Fock auditing and the CUDA 12.9 BLAS workspace, job 21195
measured 131.68 vs 139.08 s Direct against stock GPU4PySCF 1.9.0, with
24,021 vs 5,349 MiB sampled on the same 32,607 MiB RTX 5090. The native
endpoint evaluated 9,336,706,380 exact nuclear primitive components,
completed all 288 columns and physical CPKS checks, and had no numerical
host fallback. Against the input-matched residual-audited matrix in 21174,
the raw max error, asymmetry and translation residual were `6.48e-10`,
`1.79e-14`, and `1.90e-13 Ha/Bohr²`; all pass `2e-8`. Comparison to the
unmodified timing arm gives `2.45e-8` raw error and must not be labeled an
accuracy pass against stock.

The Direct win used `GENERATIVEQC_EXPERIMENTAL_R2SCAN_LEDGER_SOURCE=1`.
Reversing arm order in job 21196 reproduced it: 133.57 vs 140.77 s,
with 24,131 vs 5,761 MiB sampled. Without the experimental source that
same job took 154.51 s. Its native Hessian execution portion was 127.70 s,
leaving 26.81 s in the fresh SCF and endpoint setup, versus 6.47 s with
the experimental source. The native default/experimental raw matrices
differed by `7.40e-13 Ha/Bohr²`. Do not promote the experimental SCF lease
on the basis of the Hessian A/B alone. Keep the ordinary CUDA SCF policy
unchanged until separate source-admission and energy/force gates qualify.
DF in 21195 took 86.20 vs 118.50 s and 18,453 vs 4,703 MiB, with full
auxiliary response and `1.64e-9 Ha/Bohr²` raw error versus stock.
Job 21196 repeated DF with the experimental SCF source disabled: 83.11
vs 120.31 s, 18,563 vs 4,685 MiB, `1.62e-9 Ha/Bohr²` raw error.

The standalone comparator requires matched Slurm job and GPU for performance.
`--accuracy-reference` permits a distinct input-matched, residual-audited
Direct matrix, verifies the artifacts, and still reports the stock matrix
discrepancy. This prevents the slower diagnostic solve from inflating the
performance claim.

### Scope the cold SCF source to the Hessian

Replace the experimental environment switch with a private synchronous cold
reference entry point for a budgeted dense CUDA Hessian. The entry point
accepts one fresh energy-only item and does not persist authority across batch
calls. The native model gate restricts the cache to non-warm strict all-electron
pure Direct RKS r²SCAN, with s/p shells and 97–256 Cartesian AOs. Only unpromised
ledger capacity can be used, and the existing 3 GiB optional value ceiling,
device capacity and incumbent exact fallback remain. Audit the source and
release its allocations before the Hessian acquires the stationary state. Do
not enable this cache for ordinary energy/force, HVP, or DF jobs.

Job 21205 tested admitted, capacity-refused and ordinary runs on the 96-atom
shape: admitted 32,617,962 FP64 values, refused with no values built, and did not
attempt storage on the ordinary path. Final live bytes matched the refusal
arm; energies matched within `1e-9 Ha`. Slurm 21203 finished default Direct in
134.65 vs 139.38 s, with 23,837 vs 5,339 MiB sampled. It passed the independent
raw oracle at `6.48e-10 Ha/Bohr²`; stock Direct is still only the timing arm.
Default DF in 21198 took 82.64 vs 118.94 s, sampled 18,453 vs 4,703 MiB, and
matched the timing arm within `1.64e-9 Ha/Bohr²`.
The DF pair in 21203 also passed at 86.33 vs 120.68 s with `1.61e-9 Ha/Bohr²`
error. The reversed Direct candidate in 21206 took 134.39 s, but stock SCF
did not converge and is excluded from timing comparisons. Its separate ordinary
energy/force regression against the pre-scope build differed by `1.36e-12 Ha`
and `1.85e-13 Ha/Bohr`, respectively.

## References

PR #3141 (qualification updates `7226339713`, `7227252875`, `7232376549`);
Slurm receipts and Nsight summaries in the ignored scaling evidence directory.
