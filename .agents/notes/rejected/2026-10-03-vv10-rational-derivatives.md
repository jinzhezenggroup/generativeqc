# Rejection: arbitrary-quotient VV10 partials

Status: rejected
Date: 2026-10-03

A serious candidate combined `1/gi + 1/gsum` into
`(gi + gsum)/(gi * gsum)`. It preserved the energy root and reduced pair
FP64 division nodes from 3/6 to 2/3 in feature/geometry mode, retaining the
original ordered exceptional-domain fallback. Seventy host tests and six
independent complete molecular GPU tests passed.

Slurm node1 job 5410 compared geometry and this candidate at water-24 (24
atoms), def2-SVP, grid 48 x 16 x 32. Warm median was 57.0146 -> 56.7041 s
(candidate repeats 56.7041, 56.3765, 56.8331), with one SCF iteration each.
Cold was 413.355 -> 362.539 s but 23 -> 18 iterations, so work differed.
Every sample passed: max energy 2.297e-11 Eh, force 4.166e-10 Eh/Bohr.
The warm gain is about 0.5%; it does not establish a useful endpoint benefit.
Fewer IR divisions alone is insufficient evidence. An arbitrary quotient can
compile differently from a constant-numerator reciprocal; that explanation
remains a hypothesis, not a measured hardware attribution.

The archived candidate binary is
`0f4558e83707bff7acb66925dad149d494668bd1eba613783f7a43d6e11fdfa1`, with source
patch/archive, register inventory and complete results in ignored
`.artifacts/wb97m-vv10-rational/`. These pre-#1716 measurements are historical.
The [energy-denominator candidate](../proposed/2026-10-03-vv10-energy-denominator.md)
reuses the energy value itself and has a larger measured benefit. Revisit this
arbitrary-quotient formulation only with new complete endpoint evidence.
