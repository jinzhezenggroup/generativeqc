# Reconsider the routine CBS convergence default

Status: proposed; public stopping defaults remain unchanged
Date: 2026-10-25

Follow-up: [routine Standard is implemented](../implemented/numerics/2026-10-25-cbs-routine-standard.md),
with a retained strict reference-density boundary and expanded independent
qualification. This document preserves the preceding assessment and sources.

## Problem and assessment

The user asked whether current Fast convergence should become Standard.
The current CBS Standard is a conservative numerical reference, not an
established optimum for routine energy and analytic-force calculations.
Its CCSD energy-change criterion is four to six orders tighter than common
energy/gradient defaults. The independent
[preconvergence optimization](../implemented/performance/2026-10-25-cbs-standard-preconvergence.md)
does not settle this separate choice of stopping criteria.

## External defaults checked

Inspected on 2026-10-25. Defaults below concern CCSD iteration; ordinary
canonical `(T)` is noniterative. They are not total-energy error guarantees.

| Implementation | CC energy-change threshold (Eh) | Additional criterion |
| --- | ---: | --- |
| PySCF 2.14.0 CCSD object | `1e-7` | `conv_tol_normt=1e-5`, Euclidean norm of the packed amplitude-update vector before DIIS |
| Psi4 CCENERGY, energy driver | `1e-6` | `R_CONVERGENCE=1e-7`, module-defined wavefunction-change criterion |
| Psi4 CCENERGY, analytic-gradient/optimization driver | `1e-8` | The driver tightens energy convergence; the module owns its residual criterion |
| ORCA 6.1 MDCI manual | Not inferred from this source | Lists `STol=1e-5`, maximum element of the residual vector |
| This project's CBS Standard | `1e-12` | Physical residual maximum `1e-10` |
| This project's default DZ/TZ Fast | `1e-8` | Physical residual maximum about `2.38e-7` after component-budget allocation |

PySCF's installed 2.14.0 `CCSDBase` and `kernel` were inspected directly.
The low-level function's default arguments differ from the public object's
defaults; normal calls pass the latter. `solve_lambda` forwards
`conv_tol_normt`, but its update norm is not our Lambda physical residual.
Psi4's current manual identifies itself as `1.12a1.dev55`; its driver table
and `negotiate_convergence_criterion` agree on energy `1e-6`, gradients
`1e-8`, and energy-finite-difference frequencies `1e-10`. Do not quote only
the CCENERGY option page and overlook driver tightening.

The named residuals are not interchangeable. An amplitude update includes
denominator/preconditioning effects; a vector 2-norm also scales differently
with dimension from a maximum norm. Even equal physical residuals do not
imply equal observable errors when the Jacobian is differently conditioned.
ORCA's PNO truncation settings and iterative DLPNO triples tolerances are
separate approximations and are not part of this comparison.

## Proposed direction and required evidence

Prefer a routine Standard based on empirically validated observable accuracy,
with today's strict settings available as Tight/reference. Current Fast is a
credible starting candidate: its energy-change threshold already matches
Psi4's gradient-driver scale. Do not introduce still looser tolerances merely
to manufacture a distinct Fast product or graph.

Before changing the default, qualify the complete SCF/CCSD/Lambda/response
policy, not just `ccsd_energy_tolerance`. Extend the current independent water
and C2 evidence to supported closed-shell heteroatom, ionic, stretched-bond,
small-gap and noncovalent cases, with physically meaningful basis pairs.
Include energies, relative energies and full Cartesian forces under declared
independent-reference gates; retain failures, resource refusals and complete
endpoint time/work. Near-degenerate systems must retain visible failures when
the single-reference solver or existing gap safeguards are not applicable.

The default DZ/TZ coefficient L1 sum is about 4.2; it motivates accounting for
extrapolation amplification but does not by itself justify tightening CCSD
energy change from `1e-8` to `1e-12`. Nor does an allocated error budget prove
a residual-to-observable bound. Current Fast agrees very closely with the
strict controls on water/alkanes; that narrow population is insufficient for
global default promotion. The new timing/accuracy evidence is retained in
[the preconvergence campaign](../../../benchmarks/results/cbs-standard-preconvergence-20261025/README.md).

HF/DFT defaults need their own observable qualification. Post-HF reference
orbitals and force response have additional canonical-frame requirements;
do not copy the DFT density tolerance into CBS when unifying profile names.

## Sources

- [PySCF CCSD implementation](https://pyscf.org/_modules/pyscf/cc/ccsd.html)
- [Psi4 post-SCF driver defaults](https://psicode.org/psi4manual/master/scf.html#table-conv-corl)
- [Psi4 CCENERGY energy option](https://psicode.org/psi4manual/master/autodir_options_c/ccenergy__e_convergence.html)
- [Psi4 CCENERGY wavefunction option](https://psicode.org/psi4manual/master/autodir_options_c/ccenergy__r_convergence.html)
- [Psi4 driver negotiation](https://github.com/psi4/psi4/blob/master/psi4/driver/driver_util.py)
- [ORCA 6.1 MDCI manual](https://www.faccts.de/docs/orca/6.1/manual/contents/modelchemistries/mdci.html)

Raw downloaded documentation and the inspected source are local research
artifacts under `.artifacts/convergence-defaults/`; no production dependency
on an external package or document is introduced.
