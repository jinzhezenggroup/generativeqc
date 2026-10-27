# Decision: do not promote the native multisecant diagnostic candidate

Status: rejected (default promotion of this candidate)
Date: 2026-10-27

## Problem

The [host screen](../implemented/numerics/2026-10-27-lambda-multisecant-screen.md)
saved about one Lambda action on tiny STO-3G cases. The user authorized enabling
CCSD-history multisecant preconditioning if it accelerates production. That
requires measuring complete native CBS energy + force with the existing FP32
preconvergence, history lifetime and memory budget, rather than extrapolating
from the earlier all-FP64 `1e-12` host solve.

## Decision and evidence

Retain the production diagonal default. Archive the diagnostic patches and
qualification without adding the temporary callback ABI to native code.
All measurements use the current budget-scaled Standard/auto controls unchanged,
32 GiB numeric budget, one RTX 5090, pinned QM7 inputs and native CUDA operators.
The [evidence bundle](../../../benchmarks/results/cc-lambda-multisecant-native-20261027/README.md)
contains all 29 complete comparison endpoints and an offline numerical verifier.

Using recovered FP64 CC secants only in the final FP64 Lambda stage saved no
actions on C2/C4. With the same inverse in warmup as well, C2 exchanged one FP32
action for an additional FP64 action; C4 saved one FP32 action but capture/fit
overhead exceeded that benefit. C4 complete medians were 66.606 s diagonal,
66.810 s rank 4 and 67.029 s rank 6. C2/C4 each have two reversed repetitions.
The C6 pilot, one observation per arm, took 464.611/467.602/469.321 s with
identical 12 FP32 and 7 FP64 actions. No population demonstrated a net gain.

Every arm preserved Q32. All component/composite energies match diagonal
controls, maximum full-vector force change is `3.66918e-11` Ha/Bohr, and maximum
independent candidate Lambda residual is `2.03217e-9` within unchanged limits.
The pinned independent C2 PySCF oracle passes energy and two-step directional
force gates after exact input identity matching. C4/C6 are qualified against
their diagonal controls; no separate independent oracle was generated for them.

## Implementation lessons and limits

- Correctly pair `(t, R(t))`, recovering t from native `(G(t), R(t))` only within
  the same fixed-denominator/damping FP64 epoch. Export the accepted final point
  as well as the ring to support six independent adjacent secants without an
  extra residual evaluation.
- Lambda requires its own representative ordering and sqrt weights. Pack the
  denominators without weights; a sqrt2 multiply/divide roundtrip can break exact
  diagonal identity checks. The diagnostic deliberately refuses packed rings.
- Host ownership does not make storage free. Charge retained factors against
  subsequent native phases, and account for capture and fit workspace. C6 rank
  4/6 retained approximately 375/541 MB, with estimated construction peaks of
  2.75/3.91 GB. These are numeric estimates, not measured RSS or VRAM peaks.
- Current preconvergence leaves very few FP64 actions. An inverse that saves an
  iteration in a tight all-FP64 screen may save none at this production endpoint.

The temporary synchronous Python/NumPy callback includes conversion, host fit
and application overhead in complete timings. This result rejects default
promotion of that measured implementation; it does not rule out a cheaper native
implementation, different history selection or harder systems. C7, tight-budget
native fallback and measured peak memory remain unqualified. Historical
`strict-*` archive names denote final-FP64 placement, not stricter policy gates.

## Invariants and revisit conditions

Keep exact generated physical actions and final independent residual audits.
Do not loosen energy, force or Lambda gates to obtain an apparent speedup. Keep
detached ownership, epoch/coordinate validation, stable rank filtering and a
bounded diagonal fallback when native integration is reconsidered.

Revisit after identifying a way to save materially more expensive actions or
reduce capture/fit cost. Require repeated complete CBS E+F comparisons including
C4/C6/C7, unchanged accuracy gates, and no Q-batch regression before enabling it.
The present evidence does not justify implementing or defaulting a production
history-transfer API merely because the numerical approximation is correct.
