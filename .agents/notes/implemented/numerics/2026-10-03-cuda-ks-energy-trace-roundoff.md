# Decision: compensated CUDA KS energy traces

Status: implemented; molecular endpoint qualification passed
Date: 2026-10-03

## Problem and evidence

Full-rebuild GPU4PySCF references now converge at every PBE0 README size.
Native 96-atom runs expose a separate strict-convergence problem. The
pre-integration candidate completes cold and five original warm endpoints,
but times out after 3600 seconds inside the changed-geometry call (Slurm 5328).
The integration candidate cold takes 61 iterations despite a final density
RMS of `5.44e-15`; the subsequent first warm replay also stalls. These are
incomplete endpoints, not qualified timings or proof of an algorithmic hang.

The scalar CUDA diagnostic accumulates each signed AO energy trace naively.
An independent exact-sum stress test using the actual diagnostic body loses
up to `4.37e-11 Eh` at 768 AOs, greater than the unchanged `1e-12 Eh` stopping
threshold. All six RKS/UKS size combinations fail before compensation and pass
afterwards. This establishes a reduction defect, not yet proof that it is the
only source of the observed molecular replay instability.

## Candidate decision

Apply Neumaier compensation to the one-electron, Hartree and full/range
exchange energy traces. Accumulate each channel across both spin blocks;
publish the correction before finite-value checks. Explicit round-to-nearest
addition prevents NVCC FMA contraction from invalidating the error term.

The density, Fock operator, XC quadrature, residual norms, convergence gates,
iteration budgets and fallback rules remain unchanged. There is no reference
density in the production path and no tolerance floor based on system size.
Three scalar corrections do not add an allocation, transfer, launch or sweep.

## Qualification required

- Exact analytic-sum host and real-NVCC GPU regressions at 24/384/768 AOs,
  including both spin blocks and simultaneous full/range exchange terms.
- Independent physical-component, analytic-force and reconverged finite-
  difference checks, plus a sanitizer run on the actual diagnostic kernel.
- Fresh complete cold/moved/ten-warm PBE0 endpoints on the corrected library.
  Do not relabel pre-correction source/binary evidence or select successful
  repeats from an incomplete attempt.

Raw failed and candidate runs, source patch and binary identities remain under
`.artifacts/pbe0-large-20261002/`. Revisit additional XC/provider reduction
noise if compensation alone does not close the strict molecular protocol.

## Completed qualification

The corrected 96-atom cold and changed-geometry endpoints take 26/12 SCF
iterations and 547.365/289.353 s, followed by ten one-iteration warm replays.
Both earlier uncorrected campaigns instead time out at 3600 s. All 72 native
endpoints across 3/6/12/24/48/96 atoms pass unchanged independent gates, with
maximum energy/force errors `1.0141e-10 Eh`/`3.1044e-11 Eh/Bohr`.

Six exact-sum host cases fail before the repair and pass afterwards; six NVCC
GPU cases also pass. A focused trace/physical-diagnostic memcheck run passes
27 tests, skips one and reports zero memory errors. An unrelated first-warm
history assertion fails identically on the unchanged integration binary and
is retained as a known baseline failure, not silently fixed or counted as
passed. The corrected library also passes 19 snapshot/native, 14 independent
cooperative-force and six hybrid analytic/finite-difference cases.

Review follow-up adds eight independent nonbinary-product/null-exchange cases
and seven failure-bit/nonfinite cases. Together with the six original GPU
cases and six host cases, all 27 pass memcheck in Slurm job 5380 with zero
errors. NumPy explicitly rounds each FP64 multiplication before an independent
`math.fsum` oracle; unequal spins and near-canceling full/range terms exercise
NVCC's contraction behavior. No production change or gate relaxation is needed.

See the [complete report](../../../../benchmarks/results/pbe0-def2-svp-20261003/README.md)
for raw all-repeat force evidence, failed attempts and separate binary hashes.
