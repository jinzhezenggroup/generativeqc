# Experiment: inexact VV10 inner stopping

Status: proposed
Date: 2026-10-14

The original 96-atom 48×16×32 grid trial at source `68f4c5f70` still
misses the 1800-second complete-population limit. Its late frozen-potential
inner solve spends eight iterations reducing a provisional maximum residual
far below the final target gate. Explicit `GENERATIVEQC_VV10_INEXACT_INNER=1`
tries a one-percent forcing term, bounded below by one tenth of the actual
physical maximum-residual gate and above by 1e-6. It changes when the expensive
VV10 operator is refreshed, never the physical acceptance thresholds.

The original total iteration budget, fresh-target energy epoch, two physical
observations, maximum-residual test and independent final validator remain.
The old policy is the default and explicit-off fallback. No matrix storage,
reference-oracle dependency or lower precision is introduced.

The benefit is unproven: weaker inner solves may increase expensive pair
sweeps and make the complete endpoint slower. Qualify independent energy,
force and reconverged finite differences, then retain every original-protocol
sample before considering default selection. This experiment does not close
#2426.
