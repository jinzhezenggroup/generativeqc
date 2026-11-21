# Decision: Bound I28 def2-ECP saddle steps in the CLI

Status: implemented
Date: 2026-11-20

The installed CLI's first I28 trial at `0.05068 Bohr` failed the occupied
overlap gate of `0.99`. Independently restarting that same trial cold did not
fix its continuity (`0.981397`). Rather than lowering the threshold, rerun
with initial/maximum trust radii `0.015/0.03 Bohr`, accepted-frame warm starts,
and no Cs-style cold damping. Keep the canonical B3LYP-D3(BJ) RKS model and
spherical scalar iodine def2-ECP operator fixed. Admit only this bounded I28
protocol in the CLI, with equal or tighter bounds, and do not mix the I28 and
Cs46 ECP families. Python does not yet have matching I28 saddle evidence.

The installed trial accepted 4/4 attempts with one unscaled imaginary mode
`−256.185940558 cm⁻¹`, final occupied-space overlap `0.9999999999999735`, and
maximum raw Hessian error `5.27e-7 Eh/Bohr²` over 108 independent displacements.
Its `3:05:41` native wall time and `470840 KiB` peak RSS include the frozen
terminal re-evaluation; work was 165 SCF iterations, 205 Fock builds, and
108 response solves. See `installed-ecp-saddle-candidate/i28-bounded-radius/`
under `/data/jzzeng/qualify-2844-2848-20261120`. Those results are proof for
the candidate artifact, not for every input geometry or a later rebuilt wheel.

The bounded protocol was subsequently applied to Python after a distinct
[installed candidate qualification](2026-11-20-admit-bounded-python-i28-saddles.md).
