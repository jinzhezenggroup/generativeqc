# Experiment: approximate secant continuation between frozen-VV10 anchors

Status: default-off prototype; qualification pending
Date: 2026-10-14

Keeping old inner DIIS vectors unchanged (`fead57a4e`) did not reduce the small
endpoint work: at 12 atoms, cold rose from 33.317 to 35.086 seconds and moved
from 20.116 to 23.685 seconds against `7194d6ea6`. All 24 calls passed the
original E/F gates, but the experiment was not expanded to large points.
Provisional eight-delta chains (`9c9731aa5`) reduce full J/K builds substantially
but give little 48-atom cold improvement, 387.850 to 382.227 seconds. The latter
population is still running; neither observation is a default-policy claim.

The explicit `GENERATIVEQC_VV10_REBASE_INNER_DIIS=1` proposal retains the ring
between nearby fixed-potential stages and translates its live Fock/residual
vectors by the difference between the new physical anchor and the latest inner
observation. Those observations may have slightly different densities, so this
is an approximate secant continuation, **not** a reconstruction of exact past
physical Focks. The proposal history is never acceptance evidence. One worker
owns each matrix element across every live slot; no extra arena or cross-block
synchronization is needed.

Every changed residual invalidates its Gram cache. This option therefore uses
the incumbent full Gram reduction throughout the solve, while keeping any
already admitted cache bytes accounted. The first semilocal transition, bounded
fallback and exit to the final physical target tail clear the history. Real
target Focks and the unchanged energy/density/RMS/maximum residual gates still
decide acceptance, within the original total iteration budget. A provisional
or failed state cannot become a successful seed or publish forces.

The proposal can worsen convergence or produce a rejected nonfinite history.
Require independent molecular E/F, changed geometry, finite-budget rejection,
real-device memory/synchronization checks and full endpoint evidence before
promotion. This experiment is separate from unshifted DIIS continuation and
from provisional delta-chain cadence; keep their flags and measurements distinct.
