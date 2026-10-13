# Experiment: bounded delta chains only inside provisional VV10 initialization

Status: default-off prototype; independent qualification pending
Date: 2026-10-14

The one-delta/full-refresh frozen-VV10 experiment (`7194d6ea6`) completes all
twelve 48-atom calls within the original 1800-second point limit, with cold
387.850 seconds, warm median 15.304, moved 290.871 and moved-warm median 15.397.
Every original external E/F gate passes. Cold still performs 19 anchor full J/K
builds, 17 deltas and one final full audit across 37 total Focks. The 96-atom
population has not yet completed; these observations do not close #2426.

With the additional explicit `GENERATIVEQC_VV10_PROVISIONAL_DELTA_CHAIN=1`,
an already enabled frozen-VV10/incremental initializer may perform at most
eight screened delta updates between full J/K refreshes in its provisional
semilocal and frozen-potential stages. Accumulated screening omissions can
affect a proposal, but cannot certify a physical target: **every** full-target
VV10 anchor forces a fresh full-density J/K build. The existing consecutive
full-density final energy audit, density/RMS/maximum residual gates, total
iteration budget, FP64 and grid remain unchanged. Failed provisional states
never become a last-good seed, converged result or force source.

The shared incremental policy and all ordinary solves retain their original
screened cadence. Warm resident replay retains its ordinary one-Fock path.
There is no extra storage, CPU/oracle work, or range-matrix freezing. The
eight-step bound is an experimental proposal policy, not an accuracy claim;
independent molecules, changed geometries, finite-budget rejection, memory and
synchronization checks and complete endpoint measurements are required before
any promotion. A regression or remaining timeout must be retained.
