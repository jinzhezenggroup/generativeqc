# Decision: use complete reference Fock rebuilds for strict PBE0 endpoints

Status: implemented
Date: 2026-10-02

## Problem

The clean-master PBE0 README rerun at `b9c626d15` could not qualify 24/48/96
atoms: GPU4PySCF 1.8.1 failed its 100-cycle cold SCF at the recorded `1e-12 Eh`
energy and `1e-10` orbital-gradient tolerances. This prevented independent
force qualification of the native endpoint; it was not evidence of a native
SCF failure.

At 24 atoms the default incremental run drifts to `-610.3996843421548 Eh`
after 100 cycles, while a fresh potential at its final density gives
`-610.3996843393957 Eh`. Its last reported gradient norm is `2.53e-10`, but
the fresh-potential norm is `2.33e-9`. GPU4PySCF's RKS `get_veff` consumes
`dm_last`/`vhf_last` whenever the latter carries `vj`, independently of the
public `direct_scf` boolean. Setting that boolean alone does not remove the
incremental arithmetic.

## Decision

Configure PBE0 reference engines to rebuild their complete potential from the
current density on every `get_veff` call. The instance adapter suppresses both
incremental arguments and preserves the molecule, density and Hermiticity
request. GPU4PySCF still owns all reference integrals, XC, SCF updates and
analytic grid-response forces. No reference density enters native execution.

Record the policy in the scientific protocol and retain final reference SCF
residuals with every endpoint. Rebuild work remains inside every measured
endpoint and is counted through the existing `get_veff` counter. This changes
reference execution policy, so newly measured native/reference pairs must use
matching new protocols; older measurements keep their historical settings.

## Rejected alternatives

- Increasing the DIIS subspace to 16 does not repair the incremental error:
  the 24-atom diagnostic still fails after 100 cycles with a fresh gradient
  norm of `2.29e-9`.
- Relaxing the energy, gradient, force or screening gates would conceal the
  disagreement between the incremental and full physical potentials.
- Disabling incremental SCF only through `direct_scf=False` is ineffective for
  this GPU4PySCF RKS implementation.

## Evidence

All GPU work uses finite `srun` allocations on `main` with one RTX 5090.
Standalone full-rebuild diagnostics converge at 24/48/96 atoms in 43/43/37
cycles, with fresh physical orbital-gradient norms below `1e-10`. Complete
reference cold, moved and all five original/moved warm endpoints subsequently
pass at every README size, retaining `1e-12 Eh`, `1e-10`, 100 SCF iterations,
the full spherical def2-SVP basis, and the original moving quadrature.

The native baseline also converges independently at 24 atoms; its cold/warm
energy and force discrepancies against the new original-geometry oracle stay
below `8e-13 Eh` and `1.2e-11 Eh/Bohr`. This establishes that the original
reference failure must be diagnosed separately from native force performance.

Exact iteration journals and endpoint records are retained locally under
`.artifacts/pbe0-large-20261002/` in the benchmark checkout. In particular,
`reference-24-{default,full,diis16,full-diis16}.json` distinguishes the rejected
and accepted policies, and `full-fock-reference/` preserves every complete
reference energy/force sample for 3/6/12/24/48/96 atoms.

## Revisit when

Reconsider incremental reference execution only with a newer reference backend
that passes fresh-potential residual checks and complete cold/warm/moved force
gates at all required sizes. Do not transfer this benchmark adapter into the
native production solver or imply that the two engines perform equal work.
