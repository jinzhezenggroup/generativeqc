# Decision: Differentiate the accepted fitted long-range Hamiltonian

Status: implemented
Date: 2026-11-20

## Problem

WB97M-V can converge with a fitted full-range J/K provider and a separate
fitted erf exchange provider. Replacing the second provider with exact ERIs
at the Hessian stage would differentiate a different SCF energy, even when
both exchange values are numerically close at the reference geometry.

## Decision and invariants

Borrow the separate range provider through the accepted KS binding. The SCF
range-output event orders its producer stream before the stationary KS stream;
the accepted frame gate drains the latter. Replay immutable values and
whitening on the stationary stream, not mutable J/K scratch on either stream.
Bind the range radial, threshold, owner and generation; charge independent
raw/inverse/weight panels and work for each radial source. Full-rank cutoff
separation remains mandatory.

For induced actions, emit unsigned fitted range K from the exchange-only
energy VJP. The same assembly coefficient is applied once to either unsigned
K provider. For nuclear motion, use the physically weighted exchange-only
energy derivative, add both frozen Fock motions before the same CPKS solve,
then contract both sources with their own moving weights and the solved dD.
No Direct quartets or second Coulomb term describe this model. The Hessian
package includes the needed erf leaves; a custom partial inventory fails
closed if its required radial closure is absent.

## Evidence

RTX 5090, Slurm `node4`, strict FP64, factorized full/erf DF package; independent
libcint/PySCF and Libxc values plus test-only VV10 reconvergence, separate
spherical d auxiliary fixture. Values are in Eh/bohr² where applicable.

| Gate | RKS two atoms | UKS three atoms |
| --- | ---: | ---: |
| HVP max absolute error against reconverged energy stencil | 7.80e-10 | 2.86e-9 |
| HVP max absolute error against displaced native gradients | 1.43e-8 | 6.15e-8 |
| Raw matrix antisymmetry, maximum element | 5.55e-16 | 7.63e-15 |
| Matrix–resident HVP difference | 6.25e-17 | 2.88e-16 |
| Largest Cartesian translation residual | 4.44e-16 | 5.32e-15 |
| Simultaneous Hessian owner bytes | 8,956,184 | 10,488,984 |
| Value and contraction passes per radial (HVP plus matrix) | 7 each | 10 each |
| Raw range workspace bytes | 26,368 | 34,304 |
| Fitted range affine action terms, HVP plus matrix | 89,874 | 402,246 |
| Exact nuclear components / induced Direct quartets | 0 / 0 | 0 / 0 |

Observed matrix calls after preparation and one HVP took 0.142 s and 0.451 s;
these are not fresh endpoint timings or speedup claims. Twelve rejection cases
covered invalid metrics and panels in the combined and exchange-only programs.
Raw logs and detached receipts are retained locally under
`build/hessian-gpu-sm120/lrdf-endpoint-validated-node4.log`,
`build/hessian-gpu-sm120/lrdf-force-priority-node4.log`,
`build/hessian-gpu-sm120/.lrdf-endpoint-receipts/` and
`build/hessian-gpu-sm120/.lrdf-force-receipts/`.

The complete default f/g AOT package also passed all five checks (independent
molecular energy and displaced gradients for both spins, plus CLI/Python
publication). On the CLI H₂ case, automatic selection retained the fitted LR
Hamiltonian. Complete native times were 0.776 s for the HVP and 0.846 s for the
matrix, with 8,911,736 and 8,911,928 Hessian device bytes, respectively. There
were zero exact primitive components or induced Direct quartets, and 2/12
total (1/6 LR) raw value and contraction passes. These are single-fixture
observations, not comparative performance claims. See
`build/hessian-gpu-sm120/lrdf-default-node4.log` and
`build/hessian-gpu-sm120/.lrdf-default-cli-receipts/`.

The independent molecular energy stencil also passed for a spherical g
auxiliary. Both spins with orbital d and the separate d auxiliary passed
displaced native gradient tests. The orbital-d fixture has an unresolved
independent reference Fock discrepancy (about 1.35e-6 Eh for UKS), reproduced
with either exact or fitted LR on identical input; its native fitted HVP
agreed with displaced gradients to 7.69e-8 Eh/bohr² for UKS. Keep the external
reference tolerance unchanged; do not count orbital d as independently
qualified by that Fock comparison. Direct LR plus fitted-primary smoke
checks passed for both spins. See `lrdf-angular-verified-node4.log`,
`lrdf-default-angular-node4.log` and `.lrdf-angular-final-receipts/` in the
same build directory.

The initially unresolved Fock discrepancy was subsequently traced to the
[obsolete VV10 mask in the independent reference](2026-11-21-wb97mv-orbital-d-vv10-reference-domain.md).

At the joint orbital-f/auxiliary-g boundary, the spherical ECP UKS HVP with
fitted LR (RTX 5090 job 2077) agrees with independently reconverged energy
stencils to `4.59e-10 Ha/Bohr²`. The corresponding exact-LR ECP UKS full
matrix from job 2075 agrees to `2.08e-9 Ha/Bohr²`; its Cartesian HVP passes
as well. Direct orbital-f RKS passes the spherical matrix gate to
`8.64e-10 Ha/Bohr²` and the Cartesian HVP gate to `2.12e-10 Ha/Bohr²`.
Receipts and both stencil estimates are under
`build/hessian-gpu-sm120/wb97mv-f-g-receipts/`. The fitted-LR f/g result
qualifies the HVP, not the complete matrix.

### Follow-up: fitted-LR f/g full matrices

The default factorized AOT package passed the joint spherical orbital-f,
auxiliary-g and scalar-ECP full-matrix gate for RKS and UKS on RTX 5090
(Slurm 2103). Neither model used Direct nuclear or induced quartets. The
independent Libcint/Libxc and direct-space VV10 total-energy reference rebuilt
both fitted metrics and reconverged each displacement. Richardson
extrapolation at 0.004 and 0.002 Bohr gave maximum absolute matrix errors of
`8.64e-10` and `6.50e-9 Ha/Bohr²`, respectively. The corresponding native HVPs
agreed with those matrices applied to the probe direction to `1.71e-10` and
`8.82e-10 Ha/Bohr²`. Raw symmetry, Cartesian translation and native
matrix/HVP consistency passed the `3e-9` absolute gates. The repeatable test is
`test_fitted_long_range_f_g_ecp_full_hessian`.

These are numerical gates, not performance qualification. The two dense
calls alone took 158.846 s and 266.007 s after preparation and one HVP. The
HVP-plus-matrix receipts have 7/10 value and 7/10 contraction passes **per
radial source**, at 9/10 AOs and 11/12 auxiliary functions, respectively.
Nsight Systems 2023.4.4 (Slurm 2104) collected no CUDA kernel trace, so the
timings do not attribute the cost to a particular phase. Retain both the GPU
and independent stencil receipts in
`build/hessian-gpu-sm120/fg-lrdf-matrix-evidence/` for phase attribution and
future exact-model comparisons.

## Revisit when

An operator other than the canonical fixed-ω erf source or a rank-truncated
metric obtains its own variational response and independent molecular gate.
