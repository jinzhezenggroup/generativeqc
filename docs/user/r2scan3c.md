# r2SCAN-3c

The Python selectors `r2scan-3c`, `r2scan-3c-rks` and `r2scan-3c-uks`
use the same r2SCAN-3c model. It combines r2SCAN electronic energy,
the spherical def2-mTZVPP basis, D4(BJ)-EEQ-ATM dispersion and gCP.
The built-in parameterization covers H–Ar (atomic numbers 1–18).
Changing the basis or correction parameters changes the method.

## Supported calculations

| Calculation | Energy | Analytic forces | Prepared batches |
| --- | --- | --- | --- |
| CPU RKS/UKS, direct | Yes | Not published | Yes |
| CPU RKS/UKS, density-fitted | Yes | Conditional | Yes |
| CUDA RKS/UKS, direct | Yes | Conditional | Yes |
| CUDA RKS/UKS, density-fitted | Yes | Conditional | Yes |

Force support also depends on the system size, basis, device and available
generated kernels. Query `Calculator.capabilities` for the actual settings.
`method_capabilities()` provides only the more limited backend-independent
method summary.

For supported calculations the total energy and force include each
correction exactly once:

```text
E = E(r2SCAN) + E(D4) + E(gCP)
F = F(r2SCAN) - dE(D4)/dR - dE(gCP)/dR
```

Density-fitted forces differentiate the three-center integrals, fitting
metric and moving integration grid. D4 and gCP add their own geometry
derivatives; on CUDA the gCP component runs on CPU.

## Current limits

- CPU density-fitted force calculations are limited to 8 atoms, 64 AOs
  and 128 primitives.
- The current CUDA s/p/d stationary-force path is limited to 32 atoms,
  128 AOs, 1,000,000 grid points and 100,000,000 grid-pair visits.
  Some CUDA paths require NVCC for generated derivative kernels.
- Unsupported elements, modified defining components, missing compiler
  support or insufficient resources produce an error rather than a
  partial energy or force.

For open-shell molecules, changed and freshly prepared geometries can
converge to different spin-density states. Compare the states when forces
differ; the program does not automatically follow the same root.

See `tests/python/test_r2scan3c_execution.py` for numerical tests and
`benchmarks/run_readme_dft_benchmarks.sh` for GPU benchmarks. The
[method table](../public_methods.md) gives the current selector inventory.
