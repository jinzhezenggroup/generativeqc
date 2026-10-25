# r2SCAN-3c

The Python selectors `r2scan-3c`, `r2scan-3c-rks` and `r2scan-3c-uks`
use the same r2SCAN-3c model. It combines r2SCAN electronic energy,
the spherical def2-mTZVPP basis, D4(BJ)-EEQ-ATM dispersion and gCP.
The built-in parameterization covers H–Ar (atomic numbers 1–18).
Changing the basis or correction parameters changes the method.

## Supported calculations

| Calculation | Energy | Analytic forces | Prepared batches |
| --- | --- | --- | --- |
| CPU RKS/UKS, direct | Yes | Conditional | Yes |
| CPU RKS/UKS, density-fitted | Yes | Conditional | Yes |
| CUDA RKS/UKS, direct | Yes | Conditional | Yes |
| CUDA RKS/UKS, density-fitted | Yes | Conditional | Yes |

Force support depends on the basis, device, available native providers and
checked workspace capacity. Query `Calculator.capabilities` for the actual settings.
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
derivatives. Both corrections remain inside the native prepared KS owner;
on CUDA, gCP runs on the device. Prepared replay applies each correction
once and invalidates receipts after a failed calculation.

## Current limits

- CPU and CUDA Direct forces use packaged s/p/d derivatives. Density-fitted
  forces require a supported orbital and auxiliary basis. Native CUDA FP64
  requires three partition iterations. Query the prepared calculation for
  exact property support.
- Electronic atom/AO/primitive and grid-work fixture ceilings no longer
  determine force availability. Checked workspace sizes and actual provider
  limits still apply. Integral work is streamed through bounded pages.
- gCP retains coordinate-index and dispersion-budget limits; D4 retains its
  EEQ pivot-capacity bound and must fit the correction budget.
- Unsupported elements, modified defining components, missing compiler
  support or insufficient resources produce an error rather than a
  partial energy or force.

For open-shell molecules, changed and freshly prepared geometries can
converge to different spin-density states. Compare the states when forces
differ; the program does not automatically follow the same root.

See `tests/python/test_r2scan3c_execution.py` for numerical tests and
`benchmarks/run_readme_dft_benchmarks.sh` for GPU benchmarks. The
[method table](../public_methods.md) gives the current selector inventory.

The native C/C++ preparation API and CLI use the same compiled method graph
for supported CPU/CUDA energies and analytic forces. Custom graphs, older
libraries and unsupported contexts retain their explicit compatibility
routes. See the [shared derivative contract](native_cpp.md#density-fitting-and-derivative-boundary)
for the native admission boundary.
