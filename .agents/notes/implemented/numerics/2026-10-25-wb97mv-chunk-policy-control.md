# Decision: compare WB97M-V histories under the same initialization policy

Status: implemented
Date: 2026-10-25

## Problem

The complete CUDA KS runner compared ordinary and two-iteration chunk histories
while allowing chunk width to select different VV10 initialization algorithms.
The ordinary default starts with a semilocal proposal; explicit chunks start
with the full target. On the retained H2 fixture, ordinary produced three rows
and chunks two, although both final energies were -1.1419425513004247 hartree.
The ordinary first-row XC energy was -0.35037385114601166; the full-target value
was -0.34144310618114448. This is a policy mismatch, not a chunk arithmetic error.

## Decision

The trajectory control explicitly disables both VV10 initializers for both
widths. All existing trajectory, energy, residual, exchange, resident-feature,
host-fence and no-host-XC gates remain unchanged. Every compared row must also
identify the full target. A third solve selects the actual default initializer,
requires a semilocal proposal followed by full-target certification, and checks
its endpoint together with both full-target routes against independent CPU
composition. Environment values are restored on success and exception.

`generativeqc_ks_cuda_tests --wb97mv-chunk` isolates the control. No production
solver, default, scientific expression or tolerance changes.

## Evidence

Release sm120 build with verified sccache 0.16.0 on node1, CUDA 12.9.1, RTX 5090,
finite Slurm main/gpu:5090:1 allocations with assigned visibility preserved:
the isolated control and complete `generativeqc_ks_cuda_tests` pass. Build and
raw output remain in ignored `.artifacts/repair-20261025/remote/` and the matching
node1 `qc-pr2287-repair-20261025/evidence/` directory (`wb97mv-policy-*` and
`full-runner-policy.log`). The earlier `wb97mv-history-red.log` remains retained.

## Rejected alternatives

Comparing only final energies would discard the chunk trajectory guarantee.
Disabling the production initializer globally would remove a separately
qualified policy to accommodate a stale test. Increasing tolerances cannot
make two intentionally different initialization histories equivalent.
