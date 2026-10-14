# Candidate: select bounded native VV10 initialization by default

Status: candidate default; endpoint qualification pending, not yet promoted
Date: 2026-10-14

The explicit initialized/incremental experiment `7194d6ea6` passes complete
3/12/48-atom populations and independent molecular force gates. At 48 atoms,
cold/moved are 387.850/290.871 seconds, against 575.813/405.620 for the existing
default (`948fb1058`). At 96 it completes cold in 1328.238 seconds and five warm
calls (median 58.358 seconds), then reaches the original 1800-second whole-point
limit during moved/0. It does not resolve #2426.

The candidate selects the existing bounded frozen-VV10 initializer for its
qualified strict-FP64 resident restricted VV10 domain, without enabling
incremental J/K, range freezing, either DIIS continuation, longer delta chains,
or optional device final validation. An explicit
`GENERATIVEQC_VV10_FROZEN_POTENTIAL_SEED=0` keeps the ordinary full-target solve.
A caller-selected device chunk keeps its existing solver. Unsupported spin,
precision, provider or iteration-budget cases retain their bounded fallback.

This is an initialization policy only. All provisional and target work shares
the original total budget; only complete physical target evaluations can
satisfy the unchanged energy/density/RMS/maximum residual gates and authorize
forces or retained seeds. Already stationary initialized densities receive an
unchanged-density physical energy confirmation. Geometry transport starts at
the target instead of moving a good seed toward a semilocal-only solution.

The published curves cannot use the earlier opt-in population as evidence for
this candidate. Rerun default independent molecular E/F, budget/replay tests,
sanitizers and original 3/12/48/96-atom populations, with every experimental
environment selection absent. Preserve all superseded slower points and any
remaining 96-atom timeout. No #2426 closure without its complete population.
