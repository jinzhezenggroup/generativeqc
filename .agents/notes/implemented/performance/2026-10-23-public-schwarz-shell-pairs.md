# Decision: specialize public-AO Schwarz preparation by shell pair

Status: implemented in the local candidate
Date: 2026-10-23
Base: `1e77b239e551856741abedc0b45a6e8461f261b9`

## Problem

Direct DFT cold still pays for generic public-AO self-integral recurrences even
when the packed basis proves s/p/d shell extents. The Cartesian source owner
already uses specialized recurrences. This is preparation work, separate from
the number of SCF iterations or the much cheaper floating-point block predicate.

A source-matched PBE0/96 Nsight trace attributes 1.130920 GPU seconds to the
five public Schwarz kernels. Orders two and four account for 0.466170 and
0.406593 seconds. The candidate reduces the same five kernels to 0.122300
seconds, evaluating the same 589,824 oriented AO pairs. These intrusive traces
are mechanism evidence, not clean complete-endpoint timings.

## Decision and invariants

Parameterize the compiler's existing public Cartesian-term contraction by its
primitive evaluator. The generic wrapper retains its original evaluator;
Schwarz s/p/d pairs select existing shell-class recurrences with exact extents.
Native dispatch proves the pair angular order before choosing a specialization.
Keep the scalar ssss shortcut and generic f-shell path.

- Preserve normalization, signed spherical coefficients and contracted
  primitive/term summation order. Primitive recurrence rounding may differ.
- Compute both oriented AO pairs independently. Do not fold the triangle or
  substitute a looser transformed Cartesian bound at screening boundaries.
- Preserve FP64, the final screening tolerance, all SCF/force gates, dense
  output layout, finite checks and exact capacity charges. No allocation is
  added and no owner lifetime is extended.
- `GENERATIVEQC_DIRECT_SPECIALIZED_SCHWARZ=0`, captured at preparation,
  restores the generic public recurrence. Later Fock/force policy is unchanged.
- Production does not consume reference densities, CPU integrals or PySCF.

## Complete endpoint evidence

Three alternating-order rounds share one Slurm RTX 5090 allocation on n1.
Each fresh process measures constructor, preparation, first SCF, analytic
forces and host export, followed by three calls from its own frozen density.
Both versions use the README default grids and Standard convergence settings.
All samples are retained, including the first PBE0 baseline cold outlier.

| Method / 96 atoms | Baseline cold median (s) | Candidate cold median (s) | Reduction |
| --- | ---: | ---: | ---: |
| PBE | 21.861572 | 21.035664 | 3.78% |
| PBE0 | 28.665403 | 27.615102 | 3.66% |
| r2SCAN-3c | 49.488149 | 46.486468 | 6.07% |

Cold reductions exceed max(2%, twice the sum of relative MADs) in all three
populations. Cold iterations remain 11/9/10 respectively; all warm calls use
one iteration. Warm medians are 5.453777/5.483969, 7.190364/6.958069 and
16.074791/16.165229 seconds. No warm acceleration is attributed to this
preparation optimization; PBE and 3c median increases are below 0.6%.

The core campaign and 3/12/48-atom plus Fast holdouts contain 136 complete
calls. Another 18 calls cover cold, warm and a 0.02-Bohr changed geometry for
PBE0/12/96 and 3c/96. Every call passes its unchanged independent energy and
force gates. Static references retain their dense-grid stability and CPU/GPU
spot receipts; changed geometries are freshly reconverged with tighter,
full-density GPU4PySCF and full grid response. Standard paired maxima are
1.365e-12 Eh and 1.662e-9 Eh/Bohr. These gains reduce, rather than eliminate,
the remaining gap to GPU4PySCF; no reference timing is relabeled as current.

One Fast 3c/96 holdout differs by 4.372e-8 Eh/Bohr, failing the separately
recorded 1e-8 paired diagnostic while passing the unchanged independent
acceptance contract. Its physical residuals are about 1.59e-7 in both arms.
Two further alternating Fast pairs add 16 independently qualified calls,
bringing the total to 170. Comparing the three fresh processes within each
unchanged binary gives maximum force variation of 4.042e-8 for baseline and
3.015e-8 for candidate. The new baseline/candidate pairs differ by 3.129e-8
and 3.176e-8. This establishes baseline variation on the same scale, without
proving its cause. Retain every stricter paired diagnostic failure; do not
increase its threshold or claim bitwise equivalence for Fast.

## Validation and provenance

- Native GPU bounds are checked against CPU integral diagonals and the generic
  GPU schedule for signed contractions, s/p/d/f, spherical/Cartesian AOs,
  distinct batch items, exact mandatory capacity and nonfinite failure.
- Compute Sanitizer memcheck reports zero errors. Six additional high-angular
  GPU tests pass, including RHF/UHF energy and forces against Libcint/PySCF.
- 112 host tests pass; 48 explicitly gated CUDA cases skip in the host-only
  selection. Compiler structure, code ownership, formatting and structural
  lint checks pass.
- The existing `--canonical-values-only` suite fails with `range fallback
  displaced an independently generated J/K channel` on both the untouched
  baseline and candidate. This is retained, not silently counted as a pass.
- The first baseline attempt lacked the packaged `pbe0_rks_spd` artifact.
  Both complete AOT packages were subsequently built from their respective
  sources; that incomplete attempt is excluded from timing populations.
- Builds use verified sccache 0.16.0, CUDA 12.9.1, Release/sm120 and unchanged
  cache contents. All 5,026 baseline and 5,027 candidate build/runtime/test
  inputs match their retained manifests. Compiler commands, cache statistics,
  exact core/plugin hashes and failed attempts remain in ignored artifacts.

The [benchmark report](../../../../benchmarks/results/public-schwarz-cold-20261023/README.md)
retains compact endpoint qualifications, input-manifest hashes and reproduction
scripts. Full raw arrays, source manifests, logs and Nsight reports are at
`n1:/data/jzzeng/direct-dft-cold-fix-20261023/` and the worktree's `.artifacts/`.

## Rejected alternatives and revisit conditions

Do not credit the earlier unqualified triangular-Schwarz experiment, promote
FP32 block screening, or change convergence tolerances for these gains.
Symmetry folding changes independently rounded boundary values; this change
instead narrows the proven primitive workspace. Revisit further preparation
work only with remaining complete-endpoint attribution. The much smaller
residual Schwarz cost does not justify a new independent recurrence or a
benchmark-specific dispatch rule.
