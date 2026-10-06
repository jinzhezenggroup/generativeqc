# Decision: stationary DFT consumes the common resident force scheduler

Status: implemented behind existing qualification controls
Date: 2026-10-06

## Decision

When `GENERATIVEQC_BOUNDED_ANGULAR_FORCE=angular` and
`GENERATIVEQC_PSSS_RESIDENT_BRA=1` are selected before preparation, the generated
Direct exchange plan builds the common psss host inventory within its remaining
device lease. It uploads both complete views and retains a
`DirectForceResidentBraSchedule` with stream-ordered ownership. Host staging
capacities and device allocations are both included in resource accounting.

The full-range angular scheduler substitutes the shared resident launch for its
order-one bounded pass. It does not add a traversal or invoke HF twice. The same
resident kernel consumes Combined or Separate source weights through the common
task owner. DFT uses Separate J'/K' arrays and retains the caller's signed
Coulomb/exchange coefficients and native force convention. The publisher alone
converts to energy derivatives. Long-range and other angular passes retain their
qualified execution, and neither opt-in is promoted by this extraction.

## Resource and fallback invariants

- Zero/missing views, a task-count overflow, unsupported primitive capacity or
  insufficient metadata budget retain the bounded psss pass.
- Views publish only after both allocations and uploads have been enqueued.
- Host metadata OOM retains the existing bounded owner.
- Device metadata OOM fences pending uploads, frees only resident allocations,
  restores both actual and expected device accounting, and retains required
  bounded buffers. Execution/fence errors propagate instead of selecting a
  successful fallback.
- Separate channels share shell/AO and primitive geometry traversal while keeping
  independent activity; cancellation of J and K cannot screen either channel.

## Evidence

The node1 host campaign passes 34 checks: independent Combined/Separate force
controls, full/partial resident primitive views, exact inventory ownership,
capacity limits, missing views, metadata allocation failures at either stage,
and fence/unknown CUDA error propagation. The force controls compare 19,540,224
coordinates bitwise against the historical independent HF consumers.

Real-device Libcint qualification independently covers bounded, angular and
resident schedules, RKS/UKS, Cartesian/spherical representations and disabled
exchange. Clean GPU/end-to-end receipts are retained separately; this design
note does not claim a measured endpoint speedup.

## Rejected alternatives

Silently using empty matrix-packer resident fields would never execute the
candidate. Skipping the original psss pass before complete admission could drop
force work. Retaining its pass after a resident launch would double count work.
Each is avoided by selecting the complete class's owner once at the shared
schedule boundary.

## Revisit when

Complete #2007-based endpoint evidence justifies a default/resource selection
policy, or a long-range resident consumer is independently qualified. Neither
change is implied by this shared execution extraction.

## References

- `2026-10-06-direct-force-output-policy.md`
- `2026-10-06-shared-resident-force-inventory.md`
- #1892 and #2007
