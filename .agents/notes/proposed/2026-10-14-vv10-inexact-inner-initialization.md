# Experiment: avoid oversolving provisional VV10 inner problems

Status: default-off experiment; endpoint qualification pending
Date: 2026-10-14

The first frozen-potential prototype (`003e5c7b0`) reduced the 48-atom cold
endpoint from 574.811 to 463.084 seconds and physical pair sweeps from 19 to 6.
It also increased total Focks to 41, including 16 semilocal-only and 19 frozen
inner iterations. The 12-atom moved endpoint regressed from 32.510 to 35.425
seconds. Its transported near-target density was first pulled toward a
semilocal-only solution, then driven back to the full VV10 target.

This follow-up keeps the same opt-in flag and scientific acceptance gates. A
transported or external seed starts with a fresh full-target anchor. Native cold
guesses still begin with a semilocal/RSH solve, but refresh at maximum residual
below 1e-5. A frozen inner solve refreshes at 1e-3 of its anchor residual, bounded
above by 1e-7 and below by the stricter of 1e-12 and one tenth of the original
target residual gate. These thresholds only choose when to recompute the
expensive operator. No provisional state can be accepted or publish forces.

The ordinary consecutive full-target tail begins only when the physical maximum
residual already passes the original target gate. This avoids starting multiple
expensive tail sweeps while a cheap fixed-potential solve still has useful work.
The 16-step inner cap, original 100-step total budget, reserved 20-step exact
tail, history scopes, precision, quadrature and bounded fallbacks are retained.
The earlier semilocal-only diagnostic flag keeps its original threshold.

Source-matched independent molecular energy/forces, changed-geometry behavior,
finite-budget rejection, sanitizers and complete 12-call populations are required
before any default promotion. Earlier measurements remain attached to their
original source, including any timeout. This experiment does not close #2426.

See `2026-10-14-vv10-frozen-potential-initialization.md` for the original design.
