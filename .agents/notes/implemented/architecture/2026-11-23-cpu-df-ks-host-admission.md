# Decision: Model CPU DF-KS host capacity by allocation lifetime

Status: implemented
Date: 2026-11-23

## Problem

Native CPU DF-KS energy execution and global host-budget admission are
separate. PBE-RKS HeH⁺/STO-3G with a same-basis fit converges in five
iterations at −3.0721689595826 Hartree, but the previous CLI and Python
planners both refused the host resource plan. The fitted byte argument forwarded by the KS method
does not cap allocations in the CPU branch of `PreparedFockPlan`. Do not
recommend it as an alternative CPU host budget.

## Boundary

The inventory covers exact linked LDA/PBE CPU FP64 energy, one full-range DF J
provider, no ECP or corrections. Size from actual orbital **and** auxiliary
topology, including defaults and their Cartesian versus public extents.
The CLI and Python use one checked native inventory and one linked program
lookup keyed by the exact MethodIR identity. Python binds replay to the selector
from its own admission check, not a selector supplied by an external plan. It
reuses that check's model signature when reporting basis metadata and refuses
force replay. No promotion for forces, Hessians, hybrid/long-range exchange, or
derivative diagnostics.

## Allocation phases to cover

- **Construction:** CPU `PreparedFockPlan` builds one-electron integrals with
  `include_eri=false` for pure DF. Current KS preparation can still retain
  one-electron derivatives even for energy. Include them unless the admitted
  descriptor explicitly disables that preparation. Bound both basis copies
  and Cartesian/public integral projection; during a real-spherical projection
  the raw Cartesian and public metric/three-center arrays overlap.
- **Metric and whitening:** The native scalar eigensolve and inverse square
  root require separate metric scratch. The inverse root remains live while
  whitening creates the orthonormalized three-center factor and, when selected,
  its auxiliary-major copy. The raw metric and raw three-center tensor survive
  preparation. Factor storage uses the full auxiliary extent, not the effective
  metric rank.
- **Replay:** A retained auxiliary-major copy and a temporary transpose are
  alternative owners, not simultaneously live copies. Charge Coulomb scratch
  in addition to the ordinary KS solver/grid envelope. Compose persistent
  owners over the fleet and serialized peak scratch according to the actual
  replay schedule, including any preliminary-guess allocations that overlap.
- **Diagnostic override:** `GENERATIVEQC_CPU_DF_MATERIALIZED_DERIVATIVES=1`
  can materialize coordinate-scaled raw DF tensors. An energy plan excluding
  these tensors must refuse this override when derivative preparation is live,
  or explicitly bind a preparation mode that does not read it.

Neither the Direct four-index KS envelope nor the CUDA source-backed DF host
envelope describes this provider. Do not coerce either planner into accepting
CPU DF by changing its backend flag or treating a provider byte argument as
an enforced host limit.

## Acceptance and remaining qualification

The CLI and Python quote the same 3,758,168-byte peak on spherical water
PBE/def2-SVP with a def2-TZVP auxiliary at grid level 0 (24 public orbital AOs,
43 public auxiliary AOs). Tests cover exact/one-byte-short admission, a
heterogeneous same-basis fleet, a different spherical auxiliary basis,
unchanged fitted Fock-build counts, independent PySCF energy, selector
tampering and refusal of the materialized derivative override even if set after
estimation. On one local single-threaded water timing, CLI and Python singlepoint
passed the 5% end-to-end gate.

For Python batch replay, the final run used the installed CPU Release library,
one BLAS/OpenMP thread and 40 alternating pairs per grid after warm-up. The
bounded arm includes estimation, preparation with the supplied plan, execution,
and close. The ordinary arm prepares, executes and closes the same energy batch
without a budget. The paired median overhead was +4.13% at grid level 0 and
−0.02% at grid level 3 (28,981,256 modeled bytes). Both arms converged with 13
iterations and 15 Fock builds at each grid, and yielded identical energies
(−76.28565903143046 and −76.30510841399348 Hartree, respectively). Earlier grid-0
measurements before signature reuse crossed the 5% gate. The final measurements
qualify these water cases, not heterogeneous fleets or other hosts.

Related: #2227 (R2/R6), `src/scf/fock_prepared.cpp`,
`src/scf/density_fitting.cpp`, `src/integrals/s_integrals.cpp`.
