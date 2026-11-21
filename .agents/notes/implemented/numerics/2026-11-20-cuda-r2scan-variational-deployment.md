# Decision: Deploy the energy-derived CUDA r²SCAN domain

Status: implemented
Date: 2026-11-20

## Decision

An NVIDIA build with native stationary force support links both CUDA r²SCAN
point programs and the corresponding RKS/UKS force profiles. Before SCF, select
the energy-derived domain `0x52325302` for canonical fused FP64 calculations
without an application precision budget, with three Becke iterations and a
native force orbital inventory. The choice is independent of requested
properties, Hessian packaging, tests, and temporary workspace availability.
The incumbent point program remains the owner of AUTO, budget-selected,
host-unfused and other non-admitted contexts. The internal dispatcher key is
not a new public functional code or MethodIR identity.

The final-state token binds the selected point domain, so force and response
cannot combine the incumbent potential with the derivative of the continued
energy. The legacy Python bridge permits an explicit reference-only export
of the actual arrays, grid, and Fock-provider identity; its point evaluator
and derivative contract reject the variational domain. The Python and CLI
Hessian frontends also require the native AOT package and an admitted final
state. Named r²SCAN-3c stages both D4 and gCP actions before publishing a
Hessian or HVP.

## Rejected alternatives

Do not switch domains when a derivative is requested, after a workspace
refusal, or after SCF. This would attach derivatives to a different energy
or make the same input model depend on what the caller requested. Keeping the
incumbent point policy for analytic second order would require differentiating
its actual fixed-point conditions, not just substituting an `fxc` kernel.
The [development note][work-domain] records the boundary counterexample and
the rationale for the smooth continuation.

## Evidence

- CUDA 12.9, RTX 5090, full default integral AOT inventory, build based on
  `f8e0945d0` plus this working tree. No r²SCAN qualification build or runtime
  flag selected the implementation. `sccache 0.16.0` statistics are in
  `build/hessian-gpu-sm120/r2scan-{deploy,final-routing}-cache-{before,after}.log`.
- Slurm job `19008` passed the eight CLI/Python full-Hessian and HVP parity
  combinations (both spin layouts, Direct/DF, bare and corrected), the
  canonical-basis default-grid water cases, and the domain routing gate.
  The routing gate checks both one-tile and four-tile equal-spin execution,
  AUTO and alternate-partition incumbent selection, and rejection of the
  variational reference snapshot by the legacy derivative contract.
- Slurm job `19010` passed the routing gate and four precision-budget
  regressions, including independent water, ammonia and water-dimer
  energies/forces plus frozen-seed, displacement and failure isolation.
  The direct budgeted native owner and the legacy Fast wrapper retain their
  distinct correction backends; the tests check their agreement without
  requiring CPU and GPU gCP to be bit-identical.
- Jobs `19012` and `19013` passed 11 native public force/reference tests
  and the final capability/point-domain check, respectively. The force
  matrix includes named 3c spherical DF water with warm and displaced
  executions compared to the independent moving-grid DF gradient.
- Unflagged Slurm jobs `1652`, `19011` and `1653` passed 16 distinct
  independently reconverged molecular matrices: the crossed spin/provider
  valence, scalar-ECP and orbital-d bare r²SCAN fixtures plus all four named
  3c combinations. Job `1652` hit its time limit on the last UKS/DF case,
  which passed on the isolated reruns. The Libxc/PySCF, D4 and gCP references
  are test-only. Raw symmetry, translation nulls, full-gradient agreement and
  matrix/HVP agreement remain separate gates.
- Job `19016` passed the incumbent RKS/UKS analytic-reference diagnostics and
  reconverged directional-force checks in both point domains.
- `test_ks_point_batch_resources.py` and relevant compiler codegen/point
  tests passed 365 host checks. The public API, snapshot, stationary r²SCAN
  point and Hessian frontend host run passed 91 tests (16 device/oracle gates
  skipped). The Sphinx warnings-as-errors HTML build also passed. These are
  correctness receipts, not performance claims.

## Revisit when

Widen scientific admission only when the additional force and Hessian
owners have independent endpoint evidence. Qualify mixed precision, wider
basis inventories and arbitrary r²SCAN compositions separately; do not infer
their support from a linked point program.

[work-domain]: ../../proposed/2026-11-20-r2scan-stationary-hessian-work-domain.md
