# Experiment: bounded DIIS tail from authentic VV10 target history

Status: proposed (implemented opt-in candidate; no performance claim)
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
