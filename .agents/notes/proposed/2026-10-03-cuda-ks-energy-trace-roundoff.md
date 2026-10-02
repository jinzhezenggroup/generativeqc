# Qualification: compensated CUDA KS energy traces

Status: candidate implemented; molecular endpoint qualification pending
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
