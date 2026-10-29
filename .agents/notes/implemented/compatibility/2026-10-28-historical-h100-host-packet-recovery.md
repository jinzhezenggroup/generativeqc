# Historical H100 HOST packet recovery

Status: implemented
Date: 2026-10-28

The H100 provider-owner branch passed its own evidence limit, but later default
branch evidence additions made the prospective merged tree exceed the unchanged
80 MiB limit. Branch-only storage checks therefore cannot admit a merge.

Five superseded small-CBS HOST producer/test packets move to the existing Git
snapshot recovery contract. Their durable anchor is the already-merged #2763
commit `9a0c072e279b29413085b6fa588821267478a9d9`, independently of the pending
provider-owner PR. All original compressed bytes are recovered and verified
before retirement. Publication and review inventories refer only to retained
members, with the historical snapshot manifest giving exact recovery identities.

The actual runtime capsule, all scientific vectors and independent reference
inputs, the qualified source patch, prepared plans and numerical replay remain
offline inputs in the current checkout. No numerical tolerance or outcome
changes: the original 32-slot CBS matrix is still incomplete. This migration
neither retries those slots nor promotes the new provider SDK.

Do not replace the pinned published anchor with an unmerged branch head, raise
the storage cap, or require normal numerical tests to fetch historical HOST
diagnostics. Recompute the prospective merge-tree budget after upstream changes.
