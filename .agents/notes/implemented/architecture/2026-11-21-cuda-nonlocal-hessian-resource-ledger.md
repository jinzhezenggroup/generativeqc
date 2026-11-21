# Decision: Account for nonlocal KS owners in the public CUDA Hessian ledger

Status: implemented
Date: 2026-11-21

## Problem

The stationary Hessian can reuse accepted CUDA SCF state for canonical
r²SCAN+rVV10 and WB97M-V, but a scalar SCF inventory cannot describe a separate
VV10 grid or a late-built radial Fock owner. Hessian scratch must also fit while
the accepted SCF state remains live.

## Decision

Compose the SCF and response requests before preparing SCF. Request only SCF
energy for the nonlocal Hessian; a force request needs a separate qualified
resource contract. Shared and independent nonlocal grids use the same CUDA VV10
sub-budget authority. The pageable host fleet estimate still bounds host
staging, but does not use the CPU VV10 workspace to accept or reject CUDA.

Charge one Direct Fock owner when the exact radial operators share a plan.
Fitted WB97M-V has a primary provider and a late LR provider, either fitted or
exact. The native preparation policy and capacity query interpret the same
selector and split an explicit fitted sub-budget the same way. The fleet sums
retained owners, then accounts for late LR setup excess while other SCF owners
are live. `auto` takes a component-wise maximum over the dual-fitted attempt
and the fitted-primary/exact-LR retry. This is safe only while the failed
transaction unwinds before retry. The selected policy is part of the plan
identity, not an approximation chosen by the global budget.

## Rejected alternatives

- Reserving a single DF workspace misses late LR setup and independently live
  source data. Reserving both auto transactions at once unnecessarily rejects
  feasible resource budgets.
- Adding response allowance only after SCF can admit an endpoint with no
  space for its terminal direction/output or stationary response arena.
- Passing the general forces default to Hessian preparation would conflate
  the force-consumer lifetime with the second-order consumer contract.

## Evidence

`tests/python/test_ks_device_fleet_resources.py` exercises native shape-only
publication, distinct equal-size grids, an oversized Direct topology, and the
exclusive automatic capacity envelope. The public CUDA matrix/HVP regression in
`tests/python/test_wb97mv_hessian_cli_cuda.py` checks output parity, live-byte
recovery, response-budget rejection, and exact versus fitted derivative work.
The forced auto-retry test obtains the LR cutoff independently from PySCF/libcint.
The SM120 qualification on an RTX 5090 (Slurm job 2239) passed 22 selected
matrix, HVP, and resource tests, including shared and independent Direct rVV10
grids and both outcomes of WB97M-V `auto`.

The follow-up SM120 ECP qualification (job 2242) passed the physical CsH
curvature check and the fitted WB97M-V ECP tests with spherical d/g auxiliaries
and f orbitals. Those runs exercise the native Hessian and its local bounds,
not the public global budget for those topologies. The probe verifies that an
insufficient device bound returns out-of-memory and leaves caller output intact.

The follow-up fitted LDA/PBE/PBE0/B3LYP s/p ECP global budget uses the existing
ECP one-electron setup envelope: that CUDA scratch retires before fitted
source construction. No force allowance is needed for the Hessian's energy
preparation. Slurm job 2244 passed the public budget/lifetime regressions and
all eight fitted scalar-ECP RKS/UKS Hessians checked against independently
reconverged energies. This does not qualify the WB97M-V late LR provider with
ECP, or any higher-angular ECP topology under the global budget.

The all-electron r²SCAN+rVV10 extension uses the existing KS DF value
consumer for auxiliary g shells. Budget queries count the actual expanded
orbital and auxiliary topology; raising the angular limit needs no change to
the native inventory formula or VV10 arena. The value consumer does not grant
generic g derivatives or standalone force admission. Keep shell fixtures
center-indexed when shells differ between atoms of the same element; a
per-element catalog would change the scientific comparison.

On SM120, Slurm job 2248 passed all four
`test_r2scan_rvv10_f_orbital_hessian_budget` cases: Direct and DF, each with
RKS on a shared grid and UKS on an independent VV10 grid. The test checks the
budgeted full Hessian and HVP against independently reconverged energy
stencils at two displacements, verifies the device ledger peak and scratch
retirement, and rejects host numerical fallback. This qualifies spherical f
orbitals and spherical g auxiliaries in the exercised all-electron domain;
it does not extend to other CUDA architectures or r²SCAN+rVV10 ECPs.

The fitted full-range inventory now also covers f/g energy preparation for
LDA/PBE/PBE0/B3LYP and canonical r²SCAN. The same shape-based native value
inventory already accounts for scalar ECP setup and for the restricted
occupied exchange projection. Do not reuse the energy-only limits for forces
or infer auxiliary h support from the first-order SCF value consumer.
On SM120, jobs 2249 and 2252 qualified all ten budgeted dense/HVP cases across
the five methods: all-electron RKS and scalar-ECP UKS. They checked independent
energy differences, peak device bytes and release of response allocations.
Host inventory regressions passed 168 tests, with 10 GPU-only tests skipped.

The UKS B3LYP f/g ECP qualification needed a separate displacement scan.
For the independently reconverged Libxc/libcint energies at the probe
geometry, the maximum matrix error was `1.60e-5` Hartree/Bohr² with the
`0.0005/0.00025` Bohr Richardson pair, fell to `1.45e-7` with
`0.00025/0.000125`, and increased again with `0.000125/0.0000625` as
the energy differences became smaller. Keep the middle pair for this
fixture; the fixed-point and Hessian tolerances did not change. The scan logs
are retained under `build/hessian-gpu-sm120/r2scan-orbital-f/` with prefix
`hessian-fitted-b3lyp-fg-ecp`.

## Revisit when

Expand the resource domain only after qualifying larger angular and ECP
topologies, or force consumers, with their real retained/setup allocations.
Revisit the exact host packing envelope if its conservative shell-quartet
bound rejects workloads that could fit with a tighter native inventory.
