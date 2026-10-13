# Experiment: continue approximate inner DIIS across nearby VV10 anchors

Status: default-off prototype; qualification pending
Date: 2026-10-14

The frozen-VV10 96-atom history spends 9 semilocal/RSH initialization steps,
22 frozen-potential inner steps and 7 physical target steps. Each anchor resets
the inner DIIS history, so the small nearby nonlinear subproblem repeatedly
relearns similar response directions. Integral work persists even when VV10
pair work is absent; all of these iterations remain in the shared budget.

`GENERATIVEQC_VV10_CONTINUE_INNER_DIIS=1` retains this history across provisional
frozen-VV10 anchor changes solely as an approximate proposal accelerator. Old
residuals belong to the previous fixed-potential problem; they are not claimed
to be current physical residuals. The current complete physical Fock always
supplies the actual target energy, density and maximum-residual acceptance.

The initial semilocal-to-VV10 transition still clears history. So do the bounded
fallback into the reserved exact tail and the transition from a sufficiently
stationary outer anchor to final physical confirmation. No provisional history
may replace a physical acceptance gate, classify approximate work as target
work, or extend the original iteration budget. The option changes no storage,
equations of the accepted endpoint, or default policy.

This continuation can worsen proposals or stall. It requires independent E/F,
changed-geometry, finite-budget, sanitizer and full endpoint measurements before
any selection. Preserve failed/slower trials and do not close #2426 without the
original complete 96-atom population fitting its original 1800-second deadline.
