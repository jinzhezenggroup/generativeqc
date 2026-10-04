# Decision: compose complete native correlation-only DF-CCSD(T) forces

Status: implemented
Date: 2026-10-04

## Problem

Separately qualified triples, Lambda, factor/source and reference response maps
still left the user without complete DF-CCSD(T) forces. Their source identities,
overlapping budgets, denominator-source conventions and final force signs had
to be composed on a single physical state. Rebuilding a source for response or
reusing conventional dense-Hamiltonian forces would undermine those contracts.

## Decision

`methods::detail::run_df_ccsdt_native` performs the cold geometry-to-result chain:
exact native CUDA RHF, physical DF source, native DF-CCSD, occupied triples,
corrected Lambda/parameter response, retained-factor VJP, original source/metric
reverse with the persistent nuclear sink, exact matrix-free orbital/Z/Pulay,
and the ionic nuclear gradient. It negates the final gradient exactly once.
The optional CCSD-only and energy-only modes share the same Hamiltonian.

The internal reference builder can now retain the original source/frame owner.
Its source token and storage survive the CC solve and response. Source retention
is opt-in, so ordinary energy calls preserve their previous release policy.
The sink is constructed from the exact same immutable normalized orbital and
auxiliary arguments used to create that source. A mismatched nuclear geometry
is rejected. Source reverse verifies the nonzero lifetime identity before work.
The sink dies before its source stream owner; host coefficient cotangents remain
alive until producer completion. No result escapes on any exception.

Full triples oo/vv Fock response replaces the epsilon-diagonal sources. Direct
triples factor/Fock/retained-block cotangents are added to corrected-Lambda
parameter outputs once. No `ovvv/vvvv`, full MO ERI or orbital Hessian is rebuilt.
All electronic equations and nuclear integral contractions use existing native
CUDA/generated owners. Nuclear repulsion reuses the identical small host ionic
pair assembly previously local to conventional MP2; it is now shared under the
molecule owner, with no electronic reference fallback.

## Resource and work invariants

Each phase charges all simultaneously retained numeric owners, removing only
explicit borrowed overlaps already counted by its callee. Triples and corrected
Lambda include the retained source, original problem/reference and prior response
outputs. Frame outputs are admitted before allocation. Source reverse counts
exactly N^2 Q raw cotangents and Q^2 metric cotangents. After its successful drain,
CC amplitudes/blocks/factors and the source owner are released before allocating
RHF response storage. This keeps separate phase peaks from becoming one giant
simultaneous endpoint peak.

Matrix-free orbital diagnostics explicitly report zero full-Hessian elements;
local residual/stationarity do not certify global RHF stability. The exact
reference nuclear branch retains the three-pass bounded polarization fallback.
That avoids storage amplification but can still be expensive; large endpoint
work and timing, not bounded allocation alone, determine further optimization.
An opt-in JSONL progress journal marks scientific phases without publishing
partial numerical results.

## Independent complete-force evidence

RTX 5090 Slurm job 12219 passed the initial 11 endpoint tests in 18.47 seconds.
After adding every water coordinate, job 12220 passed all 12 tests under
compute-sanitizer memcheck (`--target-processes all`) in 184.83 seconds, zero
errors. Frozen library SHA256:
`37d301d1d73c607af8b8a248f1441409afa68d1466bfc1d328dc61148c638ea5`.
Probe SHA256:
`63201edf57369359cb55a792fcdd0d5d93e51a1c7374142a34feb2edf00b39f9`.

The tests construct independent libcint three-center/metric values and NumPy
fixed-rank inverse roots, then solve PySCF CCSD/(T) with those fitted interactions
while preserving the exact RHF Fock/orbitals. They do not run DF-SCF. Dense ERIs
exist only in these tiny oracles. Tests cover H2/water/LiH, with and without
triples, auxiliary g in Cartesian/spherical representations, duplicate auxiliary
shells, translation, native energy/force FD consistency and budget failure without
publication. The declared force FD gate is 3e-7 Eh/bohr at steps 1e-4 and 3e-5 bohr.

| System | E_CCSD(T), Eh | E_(T), Eh | independent Lambda norm | full stationarity |
| --- | ---: | ---: | ---: | ---: |
| Water | -74.98859067839796 | -3.861645459414955e-5 | 8.77e-14 | 1.18e-13 |
| LiH | -7.875201060875552 | -4.794613796904107e-6 | 1.56e-13 | 2.57e-13 |

For all nine water coordinates, maximum errors against independent energy finite
differences were **1.24e-9** and **2.51e-9 Eh/bohr**, respectively. Initial warm-GPU,
cold-SCF native method calls were 1.015 s for water and 0.578 s for LiH; those are
small fixtures, not large performance claims. The water numeric bound was
108,090,255 bytes and source weight counts were 343 + 49.

Files were independently copied to n2. Slurm job 2191 on one RTX PRO 6000 reproduced
the complete water energy/force endpoint in 1.922 s including native CUDA/reference
setup in a fresh process, with energy -74.98859067839797 Eh, Lambda norm 8.78e-14
and full stationarity 1.18e-13. Executable SHA256:
`8b78650579e62fae5485bcbabbe00de6f7f9868799699ab99614f6b0c6693fcc`.
This remote run used the same frozen library above. Ignored local evidence is
under `.artifacts/complete-force/`; remote copied artifacts are under
`/data/jzzeng/qc-df-complete-force-20261004-run` on n2.

## Remaining qualification

The #1792 bounded quartet reference route and its independent 230/264-AO energy
evidence are integrated before larger cold runs. A 230-AO/488-auxiliary ethane
complete-force trial was submitted as n2 Slurm job 2192. At this note's creation
it is running; it is **not** completed force or timing evidence.

Public capability registration is unchanged. Hundreds-AO complete forces still
need independent gates and timing. Prior strict large-factor gates
`atol=rtol=3e-10` remain unqualified; neither the force tests nor the endpoint
composition relax them. Fusion/mixed precision/pipelining in #1763–#1765 remain
subsequent, independently qualified schedule changes.

References: #158, #1792, #1799, #1802, #1805, #1806, #1808 and
[the RHF response decision](2026-10-04-matrix-free-rhf-frame-response.md).

## Qualification checkpoint at the user's requested pause

The complete owner is submitted in PR #1809. Its conventional prepared-source
lifetime test seam was updated for the new opt-in DF response argument; all four
host lifetime checks pass. This is a fixture-interface correction, not a change
to the frozen numerical library above.

The user requested wrapping up and pausing. n2 job 2192 was therefore cancelled
at 44:18 elapsed, while still in corrected Lambda. It had completed cold
RHF/source/CCSD in 388.298 s and triples response in 110.614 s. Approximately
36 minutes in Lambda had not produced a complete result. This interruption is
neither a convergence failure nor successful hundreds-AO force qualification.
The copied binary and complete progress journal remain preserved.

Independent C--C stretch energy finite differences are now ready for the future
force comparison. The direction is +z/sqrt(2) on atom 0 and -z/sqrt(2) on atom 4;
other components are zero. n1 jobs 5706/5707 ran all four native displaced energy
endpoints using the same frozen library; local CPU jobs 12221/12222 ran pinned
PySCF 2.14 with the same correlation-only DF Hamiltonian and tighter convergence.

| Step, bohr | Native directional derivative | Independent directional derivative | Maximum energy error, Eh |
| ---: | ---: | ---: | ---: |
| 1e-4 | 0.0180858134513 | 0.0180858074827 | 1.052e-12 |
| 3e-5 | 0.0180858404993 | 0.0180858179988 | 1.066e-12 |

Derivative units are Eh/bohr. Native displaced endpoints took 459--500 s on RTX
5090. These are energy/FD-consistency observations; no analytic force comparison
has passed at 230 AO. Reproduction inputs, direction, reference script, logs and
comparison JSON remain in `.artifacts/complete-force/`. The native energy probe
SHA256 is `9fe322662b50c9dd22441b39a891abaee8d48d4d6ca75ca4ab2acada4872cec7`.
