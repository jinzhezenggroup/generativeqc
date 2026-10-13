# Decision: retain native KS force ownership per prepared item

Status: implemented
Date: 2026-10-13

## Problem

Python used a whole-fleet capability predicate: one unqualified item moved all
DFT force consumers back to Python. The legacy C batch ABI also interpreted any
force buffer as a whole-fleet work request. Passing mixed output buffers through
that ABI could therefore request a native force for an unqualified neighbor.

## Decision

Add a private, versioned selective-force bridge using the existing result
descriptors: NULL/zero force storage requests energy only for that input slot.
It reuses the complete ABI invalidation/diagnostic/publication transaction.
The native prepared lifecycle accepts a borrowed boolean selection span through
one shared execution gate; invalid sizes/values revoke publication, and owners
without selective execution reject it. KS implements it through the existing
scheduler, changing only per-item force admission and derivative consumption.

Python queries the native prepared capability separately for each slot. A mixed
fleet executes one native SCF dispatch: qualified slots receive complete native
forces; only remaining successful slots enter the existing Python compatibility
consumer. Uniform fleets keep their prior native entry. An older library without
the additive bridge keeps the old whole-fleet compatibility route. Native query
or execution errors propagate and never masquerade as missing capability.

## Invariants

- No new scientific equations, force qualification or precision policy.
- No split/reprepared fleet, duplicate SCF, detached D/W or cross-owner token.
- Cold/warm/moved geometry, bounded retries, failed-neighbor isolation and final
  state invalidation remain in the existing native scheduler/lifecycle.
- Legacy public batch ABI behavior and layouts remain unchanged.
- Resource planning remains conservative for the requested whole fleet; this is
  not a new complete resource-admission/accounting claim.
- The selection span is consumed synchronously and never retained by an owner.

## Rejected alternatives

Splitting the Python fleet into separate prepared native and compatibility
owners would duplicate lifecycle/resource ownership and complicate index,
geometry and checkpoint identity. Reinterpreting the legacy ABI would change
its historical mixed-buffer behavior. Letting native failures fall back to
Python would hide real allocation/numerical faults.

## Evidence

`test_dft_mixed_native_forces.py` exercises genuine native and compatibility
consumers, cold/warm/moved replays, native failure isolation, an actually
unqualified large neighbor, malformed-ABI token invalidation, old-library
behavior and error propagation. Selected UKS Direct/DF consumers retain
independent PySCF energy/force/D/W gates. Native lifecycle tests reject malformed
selection spans and unsupported owners and verify recovery. Existing uniform
batch, force-error, CLI and native API checks remain applicable.

CPU Release validation used GCC 11.4, ccache 4.5.1 and the isolated PySCF 2.14.0
oracle. Results:

- Full native build and 94/94 CTests passed, including a final rebuild/CTest run.
- Batch routing, force-error, property, lifecycle and CLI regression tests:
  58 passed, 11 skipped; existing UKS independent E/F/D/W, ragged replay and
  nonconvergence gates: 16 passed.
- After refining routing assertions to test consumer eligibility rather than
  exact internal call counts, the installed library passed all 11 new mixed
  force cases and all 7 Python/CLI endpoint cases. The native CLI child had an
  empty runtime PATH/PYTHONPATH. An installed-header C++ SDK smoke also passed
  with the runtime interpreter/toolchain paths removed.
- Native API, runtime ownership, electronic-structure boundary, docstring,
  Ruff and formatting checks passed. CUDA-enabled host syntax compilation of
  the shared KS scheduler passed; it is not device execution evidence.

Ignored local receipts are under `.artifacts/per-item-forces/` in the validation
worktree. The final changed-source manifest SHA-256 is
`9461ca43db9fe12364a4577a0ffcc3c6e7a4615862e10fad943393ef16597cd5`,
based on parent `5298ab7f289910ff3a456307573295891ca761e6`; its per-file hashes
and installed/build artifact identities are retained alongside the logs. The
final built and installed shared library SHA-256 is
`a059166fe75f1c573c6b39c1307ab767f0ef6ea6d223f4a051c9c98db7444652`.

Slurm exposed only the maintenance node `node3`, in drained state; no real-GPU
work was run. CUDA runtime and complete endpoint performance qualification
remain outstanding, so this follow-up remains Draft. This advances #2227's
scoped force/runtime retirement work; it does not complete the tracker or
admit any additional force domain.

## References

- #2227 and PR #2400.
- [Shared native KS scheduler](2026-10-13-unified-prepared-ks-execution.md).
