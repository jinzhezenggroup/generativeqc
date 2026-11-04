# CUDA fitted KS cold gauges and physical final-state proof

Status: implemented
Date: 2026-11-05

## Problem

Issue #2984 supplies five pinned all-electron spherical def2-SVP B3LYP-D3(BJ)
DF-J/K failures with successful independent E/F controls. They do not establish
a common electronic root: in particular the original O2+ native and reference
cold solves select different states. Energy-only convergence is not the
requested stationary force endpoint.

Same-density, same-quadrature independent Focks agreed with native CUDA to
approximately 1e-13--3e-12 Eh. The visible failure therefore did not establish an
XC, J/K, or physical-Fock symmetry error. Near-stationary tails have very soft
orbital modes, and small cold-seed changes can select distinct basins. Arbitrary
core eigenvector phases matter when two columns are subsequently mixed; tied
core eigenspaces also lack a unique frame. A separate admission bug disabled
automatic restricted Hückel preparation merely because an unused element in
the basis snapshot carried an ECP.

## Decision

- Preserve automatic preliminary candidates until actual target atoms determine
  ECP admission. CUDA retains its conservative whole-batch fallback for an
  actual unsupported atom; explicit requests keep their existing checks.
- Qualify a bounded cold-gauge policy only for strict-FP64, all-electron,
  spherical, full-range fitted B3LYP UKS with at most 64 AOs. Detect core gaps
  no larger than 1e-10 Eh adjacent to either beta-frontier column. Only on a tie,
  redo the native core eigensolve after a deterministic diagonal perturbation
  smaller than 1e-8 Eh in the orthogonal guess basis. Nondegenerate targets do
  not receive that perturbation. Select each column's sign from the first AO
  within relative 1e-12 of its largest component before the ordinary beta
  rotation. This fixes a cold gauge, not the physical Hamiltonian or a ground
  state. It adds no Fock evaluation, persistent matrix, CPU eigensolve, or
  reference-oracle dependency.
- Before publishing an admitted full-range fitted CUDA KS result, independently
  prove its actual unshifted physical/canonical state. A DIIS proposal can move
  along a soft mode even when that actual state already passes. Successful
  proof avoids a redundant unextrapolated corrective closure; rejection
  resumes the existing DIIS proposal and iteration budget without a token.
- Preserve the separate original limits: the caller's energy/density controls,
  1e-9 physical AO-maximum stationarity, and 1e-8 canonical AO-maximum density
  reconstruction. `FinalStateLimits.stationarity_tolerance` also tightens the
  canonical density threshold, so setting it to the physical residual limit
  indiscriminately would introduce a different, stricter density contract.
- Keep original typed provider errors and execution-local stopping/proof
  products through batch isolation. Partial numerical history is diagnostic
  evidence, never a successful label or stationary-state token.

## Rejected alternatives

Early 0.1-Eh proposal shifting did not repair the five endpoints and changed
roots; the attempted RKS shift generalization is not retained. Simply removing
the beta rotation helps the oxygen ions but fails other inputs. A small change
to its trigonometric rounding is not a robust seed policy. Unconditional core
perturbation also changes nondegenerate AlF2's basin, hence the explicit tie
guard and separate phase selection.

Ordinary same-grid DIIS, default independent Newton, and diagnostic
Newton--Krylov solves can drift along soft modes at thresholds tighter than the
published reference protocol. These controls do not justify a new production
Newton implementation, relaxed native gates, a larger iteration ceiling, or a
PySCF fallback. DIIS weights were modest in the sampled terminal states and
satisfied their affine constraint; large-coefficient cancellation was not
established as the common cause.

## Evidence and invariants

`tests/python/test_df_ks_convergence_cuda.py` retains the original Q/M,
untranslated coordinates, explicit XC/D3 definition, orbital/auxiliary records,
200-iteration limit and numerical controls. Its unused ECP-bearing Sr record
reproduces the package-versus-target admission distinction. The pinned Gist
revision and original input digests are in the fixture provenance.

Each real-GPU qualification runs through finite Slurm allocations: complete
native E/F, two prepared warm replays, independently solved matching-root
PySCF/D3 E/F on the native quadrature with full grid/auxiliary response, and
reconverged two-step native finite differences. The independent SCF uses the
published 3e-7 orbital-gradient control; a separate independent F[D] evaluation
at native D must also satisfy the unchanged 1e-9 AO stationarity bound and agree
with the native Fock. It is not acceptable to compare unrelated cold roots or
infer force correctness from an energy-only probe. Compact measured results
are retained with source/library identities under
`benchmarks/results/ks-scf-2984/qualification.json`.

Compiled CUDA controls exercise tie/no-tie and empty/full occupation guards,
orbital-sign invariance, original exception detail, per-item isolation, stale
detail clearing and strict nonconvergence diagnostics. CPU/native tests retain
existing numerical, lifetime and ABI coverage.

## Revisit when

Expand the cold-gauge domain only with independent endpoint, derivative and
resource evidence. A different occupation/subspace policy, a second-order
solver, other functionals, larger AO spaces and ECPs remain separate scientific
qualifications. This change does not claim to close the general degeneracy or
range-separated final-state work tracked by #1002 and #2840.
