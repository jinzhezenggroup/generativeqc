# Decision: independently lower value and force consumers

Status: implemented
Date: 2026-10-30

## Problem

The portable profile supplies twenty value classes but no generated force
entrypoints. Filling the force inventory by copying a legal tuned row also
changes the value lowering: mixed derivative/value Rys plans historically
specialize Fock into subset/Wick. A100 experiments improved HF Direct 96 from
11.68 to 4.45 seconds but regressed PBE0 Direct 48 from 2.27 to 2.59 seconds.
These are complete default-grid energy/force endpoints with frozen-engine cold
density warm reuse, not isolated kernel times or universally qualified rankings.

An earlier attempt to retain the portable value schedule on that joint plan
failed compilation. Scalar/subgroup force support does not define the addressed
primitive/HRR symbols owned by component-lane force support. It must not be
borrowed by an implicit component-lane Fock emitter. After correcting that
admission, the unintended subset/Wick value lowering could still exceed static
shared-memory limits. Changing launch dimensions alone does not fix ownership.

## Decision

Allow an explicit `fock_recurrence` beside `fock_schedule` on dual-consumer
selections. Specialize one IntegralIR into separate force and value requests,
preserving the scientific operator and retained contraction definitions. The
value request owns its derivative-free Rys root count, support symbols and
schedule. The force request retains the derivative recurrence and schedule.

Primary Fock, mixed/streaming Fock and work-bucket ABI names remain primary.
Generated force and resident ppps force use a separate `_force` namespace and
registry route. The legacy single-target bundle follows the same separation.
Selections without the new field retain their existing emission path. An
independent force selection owns no optional value capabilities.

The profile parser validates both plans against the actual build target; an
A100-derived seed is not a claim that another GPU has been tuned. Missing force
classes retain the native bounded fallback. Production promotion requires real
device numerical and complete-endpoint qualification; target legality alone is
insufficient. No GPU model-name whitelist is introduced.

## Invariants

- One scientific IR owns operator, root counts and output pruning.
- Private force support may only be shared when that emitter actually defines
  the addressed symbols (currently the component-lane implementation).
- Neither consumer may silently substitute the other's recurrence or schedule.
- Force registry routing includes resident wrappers, not only ordinary launches.
- Independent value configuration must fail before emission for wrong root
  counts, absent consumers, unsupported recurrence or target block overflow.
- Existing optional alternative kernels and native resource fallbacks remain
  explicit; portable metadata does not falsely advertise measured tuning.
- The schedule-only tuner cannot promote a subset/Wick measurement under a
  retained independent Rys recurrence. Its manifest writer refuses the update
  without mutation until promotion carries the actual qualified IR and baseline.

## Evidence

The one-off `independent-fock-preservation.json` diagnostic in ignored
`.artifacts/ecnu-stationary-parity-20261030/` compares all twenty primary value
classes on SM80 and SM120. They are byte-identical to the original portable
value kernels. This is a migration diagnostic, not a permanent source-hash test.

Executable tests cover scientific output splitting, target validation and
actual CUDA compilation/linking of both legacy and profile registries, including
streaming, work-bucket and resident force symbols. The narrow implicit-Fock
support regression is compiled for scalar and subgroup force schedules.

## Rejected alternatives and consequences

Copying an entire tuned profile changes several independent algorithms and can
regress value consumers. Globally changing mixed-plan Fock specialization would
also rewrite already qualified tuned kernels. Separate lowering makes the
choice explicit and costs some generated source/object size. Revisit sharing
only when compatible support ownership is explicit and complete endpoint
measurements justify it.
