# Decision: retain the pending evidence within an 85 MiB aggregate cap

Status: implemented in this change
Date: 2026-11-07

## Reason and decision

The user explicitly approved the 84-to-85 MiB increase to retain all evidence
and continue normal checks and merging for #3038 and #3040. Set only the
aggregate policy value from 88,080,384 to 89,128,960 bytes, with matching
current-state documentation.

At master `6572be1904d370bd156a754a149acce982ab0ced`, indexed results use
88,047,938 bytes. Integrating #3038 retains 88,451,743 bytes, exceeding the old
limit by 371,359 bytes and leaving 677,217 bytes under the approved limit.
Lossless compression limited to its own evidence was insufficient at the old
limit. Migrating unrelated historical capsules would expand the repair scope;
that alternative was researched but not implemented.

## Invariants

Keep every raw record, unfavorable observation, historical source identity,
checksum binding and scientific gate unchanged. Preserve the hard 1 MiB
per-file cap, 2 MiB change-review budget, exact publication checks, and normal
repository checks and merge queue. This finite storage allowance is not a
scientific qualification or authorization to increase future budgets.

## Verification scope

Existing aggregate-overflow and invalid-policy tests continue to exercise the
same checker. Check the actual integrated index and its change-review budget;
CI and real-device qualification remain separate obligations. No historical
payload migration, new archive, external backup, Release, or GPU run is part
of this policy change.
