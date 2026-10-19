# Decision: Explicit qz allocation proof for residency qualification

Status: implemented
Date: 2026-10-19

## Problem

The prepared CUPTI collector and optional DF compatibility test admitted only
Slurm. qz Jobs use a different scheduler; inventing SLURM_JOB_ID would destroy
allocation provenance, while accepting any visible GPU would weaken admission.

## Decision

Preserve Slurm admission and add an explicit, externally reviewed qz receipt.
A read-only operator-side snapshot joins live API identities to the selected
container. The native collector independently resolves the actual visible GPU
and binds the reviewed receipt hash, container hostname, assigned UUID, versions,
and relevant environment at pinning and completion. The adapter is intentionally
limited to one running Job instance with one non-MIG GPU and CUDA 12.9.

## Rejected alternatives

Neither fake Slurm variables nor a free-form scheduler label establish qz
allocation identity. A file checksum alone cannot authenticate the scheduler;
the API observation and operator evidence remain external review obligations.
No production runtime ABI or source ownership tags change.

## Invariants

- Missing, stale, conflicting, old-schema, or drifted allocation proof fails.
- Ordinary CUDA history checks instrumentation noninterference; native CPU
  history remains the independent scientific comparison.
- Successful collection remains INCOMPLETE / exit 2. Payload lineage and
  complete implicit-wait coverage are separate unresolved contracts.
- Historical rigid-translation ratchets cannot be learned from or applied to
  the current internally displaced history.

## Evidence

Executable host admission controls are in test_cupti_qz_receipts.py, with the
existing prepared and CUPTI protocol/error/lifetime tests retained. Device
qualification must retain its own source/build/artifact and platform receipts;
host doubles do not qualify a device or the whole parent issue.

## References

Refs #1629; docs/maintainer/residency_receipts.md.
