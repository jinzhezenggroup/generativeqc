# Native numeric defaults aligned with GPU4PySCF 1.8.1

Date: 2026-10-19
Status: proposed; real GPU E+F verification outstanding.
Agent: ChatGPT
Model: GPT-6

**This changes production defaults, not the explicit README benchmark protocol.**

| Control | Old native | New native | GPU4PySCF |
| --- | --- | --- | --- |
| SCF max iterations | 100 | 50 | 50 |
| SCF energy tolerance | 1e-10 | 1e-9 | 1e-9 |
| Direct J/K cutoff | 1e-12 | 1e-13 | 1e-13 |
| Eligible initial guess | Modified Hückel | MINAO | MINAO |
| CUDA local XC AO cutoff | 1e-16 | 1e-10 | 1e-10 |
| VV10 rho admission | 1e-8 | 1e-10 | 1e-10 |
| DIIS space | 8 | 8 (unchanged) | 8 |

Limitations: native J/K cumulative screening is not GPU4PySCF's predicate;
native density/physical residual convergence differs from orbital-gradient
convergence, so its independent density_tolerance=1e-8 stays intact.
MINAO remains unsupported for native CUDA RHF and fitted targets; the guarded
native Hückel/Hcore fallback is retained there. Native Gauss-Legendre grids
are not Treutler/Lebedev with NWChem pruning. Default incremental Fock
strategies differ, and native physical full moving-grid forces are retained
instead of adopting the reference's incomplete grid_response=False default.
Do not label timing populations, force definitions or error budgets identical.

Existing SCF settings and Hückel are still explicitly selectable, and
README's pinned 1e-12/1e-10 matched scientific protocol is unchanged.
The new VV10 and AO cutoffs must pass independent numerical E/F validation.

Before merge: test CPU/CUDA default admission, convergence, independent
strict energy/analytic-force and grid-response finite-difference gates;
run 3/12/48/96-atom PBE0, r2SCAN-3c, WB97M-V complete cold/warm/moved
comparisons on a source-matched GPU. Existing published 62.45/158.66-second
results must not be relabeled as measurements of these new defaults.
