# Decision: align the independent reference's response quadrature

Status: implemented
Date: 2026-10-16

## Problem

The native compiler-AD Hessian agreed with independently reconverged molecular
energies but failed the full GPU4PySCF matrix gate. The
[earlier investigation](2026-10-16-wb97mv-reference-hessian-primal-consistency.md)
retains unsuccessful isolation controls and rejected timing records. Do not
add compensating native terms or assume a missing VV10 kernel from that result.

GPU4PySCF's RKS constructor creates a separate SG1 `cphf_grids` with atomic
rule `(50, 194)`. Its Hessian response selects that object independently of
the primary SCF and NLC grids. The comparison adapter previously configured
only the latter two. Its nominal 8/4/8 contract therefore used 512/768 primary
points for H2/water but 8704/13056 CPHF points. A tighter response residual does
not make these different discretized operators equivalent.

## Single-variable real-device control

Slurm 3239 on node2/pro6000 changed only `engine.cphf_grids = engine.grids`
between two fresh reference calculations. Canonical WB97M-V, self-consistent
VV10, full moving-grid response and the default response tolerance `1e-6`
were retained to isolate the quadrature change. No vendor kernel was patched.
The native matrices are the unchanged retained clean-source84 matrices.

| Case | Default CPHF max matrix error | Primary-grid CPHF max matrix error |
| --- | ---: | ---: |
| H2 | 9.326707032830672e-5 | 1.8346335561858496e-9 |
| Water | 0.001045670973612467 | 3.602089859722213e-8 |

Errors are Eh/Bohr². Both aligned results pass the unchanged `3e-6` gate.
Reference energies are unchanged to rounding. Last-coordinate analytic minus
own-primal `.004/.002` Richardson curvature is `4.2707198888436437e-10` for H2
and `7.211370122855998e-7` for water; the latter energy stencil is noisier and
is not a full-matrix qualification. Raw A/B matrices and all displaced energies
are retained in the source deployment's
`native-default-fg-20261016/.artifacts/slurm-3239/reference-cphf-grid-control.json`.
The driver is retained locally as
`.artifacts/wb97mv-hessian-aot-20261015/reference-cphf-grid-control.py`.

## Decision and qualification contract

Configure CPHF to use the independently reconstructed primary moving quadrature
in the shared reference adapter, for RKS and UKS. Record policy and actual point
counts in the Hessian comparator. Keep its established `1e-11` response tolerance
and all numerical acceptance gates. A host semantic regression checks generated
coordinates, weights and partition policies, not implementation spelling.

Independently qualify every raw entry for both actual benchmark molecules from
values-only molecular energy stencils. Water uses the exact canonical seven-AO
STO-3G decimals, independently read from the pinned basis fixture, and
PSD-preserving occupied/virtual rotations for the reconverged energy oracle.
No finite-difference term enters the production native Hessian.

This control identifies a reference workload/configuration mismatch, not a
reference-kernel defect. It supplies no accepted speed ratio: obtain fresh
complete-endpoint timings with the corrected contract. Preserve old failed
raw measurements and null ratios unchanged. The coarse grid remains a bounded
implementation/latency diagnostic, not evidence of chemical grid convergence.
