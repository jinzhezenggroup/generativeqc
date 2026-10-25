# Decision: raw Libxc agreement does not clear the empty-spin derivative gate

Status: rejected
Date: 2026-10-25

## Problem

PR #2276's empty-beta TPSS control previously compared interior work derivatives
with physical density-matrix birth directions. Matching the pinned Libxc raw
coefficients does not establish that those coefficients differentiate the
clipped composite energy at an empty spin. Floors and the ferromagnetic branch
make these different contracts.

## Evidence

The retained input is
`tests/data/xc/physical-kinetic-predicate/empty-beta-h2plus.json.gz`; its README
records provenance and the independent host replay command. The replay uses the
current canonical CPU point and PySCF 2.14.0 / Libxc 7.0.0 on identical rounded
features, retaining every stencil and its floor/ferromagnetic branch counts.
Historical native CUDA results remain separately attributed to their source.

| Beta birth | Raw coefficient contraction | Energy secant, h=1e-10 |
|---|---:|---:|
| Occupied orbital | 1.4832041759285985e-5 | -0.03540212077555244 |
| Virtual orbital | -0.08868050921661155 | -0.1213524598089214 |
| Mixed orbital | -0.04434453510066903 | -0.07837861445368341 |

These short secants are not converged derivative limits. Continuing to 1e-16
crosses floor/ferromagnetic regions and introduces cancellation; extrapolating
the short stencil would hide the unresolved branch question.

An independent density-only control stays strictly below the beta work floor:
rho-alpha=1e-10, zero gradients/tau, beta increment=1.25e-16. All work inputs
remain fixed, so the exact derivative of the original-density energy prefactor
is E(0)/rho-alpha = -0.00021366864422640795. Its native secant differs by
1.23e-14; the exported raw beta coefficient is -0.0007225004260050542. This
isolates the contract distinction without extrapolating an AO birth stencil.

## Decision and revisit condition

Do not change floors, smooth branches, promote a default, or declare physical
birth qualified based on raw Libxc derivative parity. No production numerical
expression is changed by the replay. Clearance still requires an explicitly
defined one-sided composite derivative, justified floor/ferromagnetic behavior,
and independent AO directional/endpoint gates. The replay makes the remaining
problem reproducible without a historical binary or a GPU.

The later [admission decision](../implemented/compatibility/2026-10-25-automatic-cuda-empty-spin-admission.md)
retains this negative evidence and explicitly rejects empty-spin automatic CUDA
KS. It does not claim that the physical-birth derivative has been repaired.
