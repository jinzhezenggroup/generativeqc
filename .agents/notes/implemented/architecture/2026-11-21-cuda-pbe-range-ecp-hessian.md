# Decision: Keep scalar ECP outside both range-PBE fits

Status: implemented
Date: 2026-11-21

## Decision

Extend the [range-PBE radial model](2026-11-21-cuda-pbe-range-fitted-hessian.md)
to scalar semilocal ECPs without adding a separate derivative implementation.
The accepted SCF state supplies the effective nuclear charges; the linked
CUDA scalar-ECP owner supplies the local and projector derivatives. The
ECP lives only in the orbital Hamiltonian. Both fitted sources require an
explicit ECP-free auxiliary basis on the same centers; never infer one by
stripping the orbital basis's ECP metadata.

Preparation uses the same unscaled PBE graph predicate as LR source selection.
The Hessian still authenticates the strict physical FP64 state and the actual
radial providers and checks each fitted spectrum against its cutoff. Fitting
and exact retry remain pre-SCF choices. The native radial/setup inventory
already counts the orbital ECP workspace along with the exact LR or split fit
lifetimes; the response separately reserves its ECP derivative workspace.

## Evidence

- Slurm 2294 (RTX 5090, SM120): budgeted dense Hessian and HVP agree for all
  six synthetic Na/H scalar-ECP cases (RKS and UKS, Direct, fitted primary
  with exact LR, and two fitted operators). The separately reconverged
  Libcint/Libxc moving-grid energy oracle meets the `2e-6` Hartree/Bohr² HVP
  gate at `omega=0.33`, including signed LR-minus-SR corrections. Missing and
  ECP-bearing auxiliary bases are rejected by planning and native preparation.
- Slurm 2293: all six all-electron range-PBE cases and five WB97M-V radial
  resource regressions pass. Its ECP cases reached the analytic endpoint but
  the initial test incorrectly requested per-component results from the
  aggregate-only CUDA API; 2294 validates the corrected test.
- 202 host ECP and KS resource-inventory tests pass. Binary checksums, build
  logs, compiler-cache stats and Slurm logs are under
  `build/hessian-gpu-sm120/pbe-ecp-*`.

See the [follow-up CLI transport decision](2026-11-21-cuda-pbe-range-cli-method-ir.md).
Qualification outside SM120 remains open.
