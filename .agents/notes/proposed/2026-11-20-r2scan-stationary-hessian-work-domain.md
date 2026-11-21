# Decision to qualify: r²SCAN energy–potential work domain for CUDA Hessians

Status: proposed
Date: 2026-11-20

## Problem

The native stationary Hessian requires the deployed Fock field to be the
first derivative of the deployed energy on the differentiated directions.
The existing r²SCAN Libxc-style continuation conditions the density, gradient
Gram and kinetic-density inputs component by component. It rescales the
returned energy density to the original total density, but the point potential
is the derivative evaluated on the conditioned work inputs. In a clamped
direction these are not the same chain rule.

This is observable even without the numerical AO or CPKS layers. With the
pinned Libxc 7.0.0 oracle, use a spin-polarized point with densities
`(4e-10, 2e-12)`, x gradients `(1e-16, 3e-17)` and kinetic densities
`(2e-18, 1e-19)`. For r²SCAN exchange plus correlation, the minority potential
is `-0.0023690663`, while central differences of `rho_total * exc` at minority
steps `2e-14`, `1e-14`, and `5e-15` converge to `-0.0031541918`.
With minority density `2e-11`, above the exchange floor, the same comparison
converges to the potential `-0.0020134707`. This is a pointwise work-domain
counterexample, not a bound on error for a molecular quadrature.
Reproduce both branches with pinned PySCF/Libxc using
`PYTHONPATH=python:. python -m pytest tests/python/test_xc_boundary_qualification.py -k work_policy_energy_potential`.

## Required choice

Do not infer a variational r²SCAN Hessian from a point `fxc` implementation or
from the GGA qualification. A new stationary work policy would need to derive
both the Fock potential and its action from the same conditioned energy, and
qualify the changed SCF/force contract. Preserving the incumbent work model
instead requires differentiating the actual fixed-point conditions with the
appropriate left response when energy stationarity cannot be proved. Merely
freezing the clamps, or symmetrizing the published matrix, does not address the
first-derivative mismatch. An interior-only proof is useful as a component
gate but must not claim a molecular model whose tail points fail the proof.

In either case, validate the spin and tau pullbacks independently at interior,
clamped and exactly empty-spin points; then qualify raw molecular matrices
against independently reconverged energies before changing public admission.

## Development candidate

`python/generativeqc_compiler/xc/r2scan_stationary.py` lowers a separate,
energy-derived point model, not an option of the deployed r²SCAN method. A
quintic activation joins exchange per spin and correlation per total density
to vacuum. A quartic transition joins the constant work floor to the physical
coordinate with two continuous derivatives, for correlation spin densities
and both components' kinetic densities. Derive the potential and directional
action after that projection, including the correlation energy's normalization
to physical density. The graph uses physical Cartesian gradients and forms
correlation's invariant as `|grad_a + grad_b|²`; flooring the diagonal Gram
entries can change the curvature at zero gradient and subtracting those
entries can lose the invariant for nearly antiparallel gradients.

The host and RTX 5090 point tests compare to independent pinned Libxc formulas
evaluated on the candidate's work coordinates. They exercise interior, spin
activation, active kinetic continuation, empty spin, flat and antiparallel
gradients, with first and second directional energy differences and spin-block
permutation. Both generated entry points (energy/potential only and energy,
potential and action) agree at every tested point. For the deep correlation
activation window, the suite uses a separate fixture generated from the
upstream C formulas with 113-bit arithmetic before differencing energies.
Regenerate with `python tools/generate_r2scan_tail_reference.py --stationary --archive /path/to/libxc-7.0.0.tar.gz`.

The default-off NVIDIA test build now selects the candidate in the fused FP64
SCF and native force owners. Domain `0x52325302` identifies that continuation
in the converged-state token and force binding. The force weak form puts the
factor of one half on `V_tau`; the Hessian feature map instead puts the half
on `tau` before the point action, and includes its tangent in the moving-weight
energy cotangent. Both arrangements differentiate the same physical energy.

On RTX 5090, independently reconverged all-electron RKS/UKS × Direct/DF test
matrices passed raw symmetry, translations, HVP agreement and full gradient
and matrix comparisons. Across the four fixtures the largest entrywise
Hessian error was `6.44e-9 Eh/bohr²`. The four orbital-d Direct/DF tests and
four scalar-ECP Direct/DF tests also passed both force and Hessian gates (worst
entrywise matrix error `3.22e-7 Eh/bohr²` for the contracted UKS ECP Direct case).
The ECP/DF probes originally hit an incumbent SCF admission check before
response. The trial geometry therefore shares the same canonical graph gate
between ECP/DF preparation and Hessian state binding, without widening the
default production build. See the developer Hessian guide for the reproduction
entry point. The qualification build also exposes the new domain through the
Python and CLI entry points. Their parity gates cover bare r²SCAN and the
complete basis-bound r²SCAN-3c for both spin layouts and both integral
providers. Python legacy snapshots still reject the new domain instead of
re-evaluating its points with the incumbent work policy. These checks do not
establish continuity across backends, broad inventory coverage, or performance.
Production r²SCAN still uses the incumbent policy.

The default-off experiment described here was superseded by the
[production deployment decision](../implemented/numerics/2026-11-20-cuda-r2scan-variational-deployment.md).
