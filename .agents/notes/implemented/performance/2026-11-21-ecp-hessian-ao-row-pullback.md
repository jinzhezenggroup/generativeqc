# Decision: Reduce ECP Hessian pullbacks by AO

Status: implemented
Date: 2026-11-21

## Problem

The validated WB97M-V f/g fitted-LR ECP matrices were numerically sound but
slow. The Slurm 2105 RTX 5090 trace attributed 174.35 s (93.7% of summed GPU
kernel time) to 5,376 `contract_jets` launches. Each left coordinate used
one thread to traverse every AO and quadrature row, even when the left atom
could not move that AO relative to the ECP center. Factored DF integrals
accounted for only 6.09 s of kernel time on this trace.

## Decision

The derivative of the relative AO position is `δ(left, ECP) - δ(left, AO)`.
Use one block per AO and Cartesian axis to contract the existing generated AO
motion and cotangent algebra across the rows, then apply this incidence to
the same compact result for both atoms. For an AO centered on the ECP atom
the incidence vanishes identically. Both local and projected rows follow
this schedule; their relative directions and separate energy signs are
unchanged.

Use a fixed FP64 tree and an ordered, non-atomic gather to the radial sum.
Only six doubles per AO are staged; no per-left-coordinate jets or floating
atomics are needed. The workspace and tree/scatter work are explicitly
bounded. Keep the same physical quadratures, all validation gates, stream
authentication and private publication. Do not tune the grid or discard
nonlocal channels to hide a traversal cost.

## Evidence

- Slurm 1678, RTX 5090: 17 residual-source tests passed, including the 31-AO
  two-ECP-center spherical fixture, independent Libcint gradients and
  moving-density HVPs, and output-preserving failure gates. A 48-AO two-center
  trial exceeds the unchanged generated affine-work cap and is not an
  admitted fixture.
- Same node5 and compiler-AOT model, Slurm 2106/2107: complete **native test
  probe** wall time (SCF, prepare, HVP, dense, lifecycle) was 187.64 → 13.95 s
  for RKS and 300.57 → 24.97 s for UKS. Dense application alone was
  158.92 → 10.13 s and 266.01 → 17.99 s, respectively. Per ECP contraction,
  generated node pullbacks fell from 64,862,208 to 3,603,456 (RKS) and from
  108,103,680 to 7,206,912 (UKS). SCF inputs, completed directions, DF radial
  passes and CPKS/induced work counts were unchanged. The complete leased
  workspace increased by 512 bytes for each system.
- The public CLI A/B on the same RTX 5090 (Slurm 2107) took 160.88 → 12.04 s
  for RKS and 268.00 → 19.96 s for UKS, including SCF, all Hessian columns
  and JSON publication on the same 8 × 4 × 8 molecular grid settings. SCF
  energies were identical. The maximum matrix changes were 2.78e-17 and
  1.11e-16 Ha/Bohr². Both binaries completed 6/9 columns with 12/18
  primary-plus-LR DF passes of each kind and 20/43 induced Fock actions.
  Neither used Direct response quartets, host numerical fallbacks, displaced
  SCFs, runtime Python or runtime compilation. These timings qualify this
  fixture and grid, not arbitrary ECP or basis sizes.
- Optimized full matrices agree with the preserved independent Libcint/Libxc
  and VV10 reconverged energy stencils to 8.64e-10 and 6.50e-9 Ha/Bohr². The
  largest change from the old matrices is 1.39e-16 Ha/Bohr². Raw symmetry,
  translations and HVP parity pass the same gates.
- Slurm 1681/1683 passed the 20 selected front-end tests, including native
  CLI/Python parity for the semilocal ECP matrix of supported model choices
  and the independent spherical-g auxiliary energy stencils. Slurm 2108
  repeated both f/g fitted-LR dense fixtures from a relocated `cmake --install`
  prefix, without a Python environment or executable search path. The loader
  selected the installed library via its relative RUNPATH and used separate
  CUDA/BLAS dependencies. Both JSON receipts matched the optimized build-tree
  receipts exactly except for timing. This is an opt-in CMake installation
  check, not a wheel or portable GPU-architecture qualification.
- Raw profiling, builds/cache statistics, timings and replay receipts are
  under `build/hessian-gpu-sm120/fg-ecp-*` and
  `build/hessian-gpu-sm120/fg-lrdf-matrix-evidence/`.

## Revisit when

Profile larger admitted ECP endpoints before optimizing the remaining dense
AO matrix maps or streaming more radial layers together. The compact
pullback does not change their `rows * nao²` work or expand angular coverage.

The physical CsH/def2-SVP/ECP46 B3LYP fixture from the
[CPU qualification](../numerics/2026-11-04-scalar-ecp-hessian-bounded-validation.md)
still rejects its unscaled Cs-x HVP on CUDA, on both matched explicit and
default molecular grids (Slurm 2109). The isolated 29-AO CUDA ECP source admits
the same geometry, basis, direction, and zero weights (Slurm 1684); substituting
independently reconverged Libcint/Libxc density and its centered finite-difference
response alone reproduces the rejection (Slurm 1685). This isolates a weighted
source failure without SCF or CPKS. With the same probe input, the density-only
case rejects and the density-response-only case passes (Slurm 1686), isolating
the failure to fixed-density curvature rather than the mixed density action.
A test-only instrumented probe localized the mismatch to the nonlocal HVP
with maximum coarse/fine difference `3.53e-7` Ha/Bohr² (Slurm 1687). Replacing
the diagnostic pair by 224 × 44 and 320 × 60 passed the unchanged comparison
with both physical weight panels (Slurm 1688). Its refined local and nonlocal
HVPs agree with independent Richardson-extrapolated Libcint gradient
differences to `7.81e-13` and `2.23e-12` Ha/Bohr², respectively. This was not
a change to the production quadrature. A full third grid takes the 29-AO affine bound
from 6,343,375,104 to 18,536,605,824 terms, above the incumbent 2³⁴ budget.
Replaying only the nonlocal map would add 27,572,160 affine terms, but still
requires its AO collocation, projections, reduction and workspace accounting.
Inputs and receipts are under `build/hessian-gpu-sm120/csh-physical-ecp/`.

The selective-validation prototype in that directory replayed only the
channels that exceeded the gate, then published the original 224 × 44 result.
It passed the fixed-density, moving-density and density-response-only probes
(Slurm 1689). In a **test-only** SDK interposition (Slurm 2110), the same
explicit-grid physical B3LYP/CsH endpoint published all six columns in 48.14 s
and the unscaled unit HVP in 11.33 s. The maximum independent error was
`7.95e-8` Ha/Bohr² at the retained 0.0003-Bohr stencil; raw symmetry,
translation and matrix/action consistency passed. This prototype is not
linked in production. Its 17 isolated source regressions also passed
(Slurm 1691), including lifecycle and rejection without publication.
Integration needs to preserve existing two-grid admission when there is no
room beyond mandatory aggregate reservations, and account for attempted
validation separately from baseline work.

The test-only default Treutler level-3 replay (Slurm 2111) took 49.55 s
for six columns and 11.70 s for the unit action. Independently reconverged
Libcint/Libxc energy differences at 0.002 and 0.001 Bohr gave a Richardson
maximum error of `9.50e-8` Ha/Bohr² and center energy agreement to
`2.4e-12` Ha. The subsequent production decision is recorded in
[bounded CUDA ECP confirmation](../numerics/2026-11-21-cuda-ecp-hessian-confirmation.md).
