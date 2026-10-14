# Experiment: bounded DIIS tail from authentic VV10 target history

Status: rejected for default promotion (retained opt-in research source)
Date: 2026-10-14

The nested frozen-VV10 initializer avoids pair sweeps in its inner solves, but
successive anchors repeatedly rebuild a cheap-operator DIIS history. An inexact
inner threshold and a once-per-solve outer secant both retain the 96-atom
whole-point timeout; neither is a qualified default improvement.

`GENERATIVEQC_VV10_TARGET_DIIS_TAIL=1` keeps the previous genuine target Fock
and its genuine AO commutator in two explicitly charged device matrices. After
a second target has a maximum AO residual below 1e-5 and at least twenty-fold
contraction, it can seed the existing ordinary DIIS updater with those two
physical pairs. No inner Fock/residual enters this history. The regular GPU
eigensolver projects the Hermitian proposal to an occupied density; density is
never extrapolated or copied to a host oracle.

The tail is attempted once per solve and can consume at most three unsuccessful
full-target iterations before the existing nested policy resumes with cleared
history. A four-fold residual increase also restores that policy. All work
uses the original total iteration budget, the original exact physical energy,
density and maximum-residual gates, and the existing final-state validator.
Warm calls, failed attempts and new geometries reset the optional trajectory.
Device allocation refusal retains the smaller ordinary arena; host allocation
and non-allocation device failures still propagate. Mixed, incremental,
frozen-range and the other outer-history experiments do not compose here.

This differs from approximate inner-DIIS rebasing: each stored pair is an actual
evaluation of the same complete nonlinear target operator at its own density.
It still might cost more pair sweeps than it saves inner Focks. Qualify small
complete endpoints and independent molecular E/F/energy derivatives first; do
not promote it merely because its proposal is scientifically admissible. Keep
negative observations and source/library identities.

## Measured rejection

Exact source `231663cd9533b35e7a14ec7ecfe3f676b9a88fb2`, native library
`138a6ef8e75cc7342d5fa6119e4645f8c94d515ed3e0e85a2166a8b8f0aa5215`, passes
24 device cases on n2 Slurm 2965, including six independent molecular E/F and
reconverged finite differences. n5 Slurm 1549 completes both original-protocol
3/12-atom populations with all 24 E/F gates passing. At 12 atoms the cold,
warm, moved and moved-warm times are 32.018/1.718/27.520/1.789 seconds.
The cold and moved trajectories each require six pair sweeps, against four in
the incumbent nested solver. The 3-atom first cold call is 41.074 seconds and
is retained without subtracting setup. Do not call fewer inner Focks a win:
this replaces linear-cost inner work with more expensive pair work. No 96-atom
population is launched for this candidate, and no default selector is changed.
See the frozen target-tail source branch and `vv10-range-followups-20261014`.
