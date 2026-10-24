# Decision: bound the through-d long-range force call graph

Status: implemented in candidate; complete moved-endpoint qualification ongoing
Date: 2026-10-23

## Problem

Lowering the requested stack to 16 KiB in the earlier
[headroom change](2026-10-23-through-d-stack-headroom.md) did not bound CUDA's
automatic stack growth. The linked generic long-range force dispatcher includes
unreachable f-order callees even for a proved through-d basis. On CUDA 12.9.1,
sm120, its restricted and unrestricted force entry points report a 52,352-byte
stack. A real WB97M-V DF96 cold/warm diagnostic observes that same context limit.
The next geometry preparation sees only 19,222,888,448 free bytes and selects
streamed J, requiring approximately 25 seconds per application. A 50-minute
finite job expires without finishing the moved endpoint.

Closing only the old Python stationary-response owner before the move frees
308,281,344 bytes (4,255,514,624 to 4,563,795,968), and a separate 40-minute job
still expires. That diagnostic workaround is not a production fix. The larger
persistent cost belongs to driver stack backing.

## Decision and invariants

When host-owned basis metadata proves maximum shell angular momentum <= 2 and
the operator is long-range exchange, instantiate the existing one-pass force
queue with fixed LR and total angular ceiling eight. Compile-time dispatch
omits orders nine through twelve. Preserve every screening predicate, queue
item, subtile, custom omega, packaged-omega source, spin coefficient and output
owner. F/unknown metadata and short-range exchange retain the complete generic
call graph. Do not lower a shared context's stack limit.

This is not the rejected
[thirteen-scan angular schedule](../../rejected/2026-10-04-tzvpd-angular-force-schedule.md):
there is still one traversal of the original candidate domain. No integral,
precision, convergence, grid or basis contract changes.

## Evidence and limitations

The source inventory contains 5,984 authenticated inputs from base
8661c5cd4d24b90399aef0c884988ef10c2c5f0c plus the retained candidate patch.
The core and five stationary manifest targets were rebuilt with verified
sccache 0.16.0. Linked entry-point resource reports show a 23,016-byte stack for
both new LR specializations; full/f fallback entry points remain unchanged.
These are linked resources, not an inference from unlinked PTXAS frames.

622 host tests pass, including execution of the actual production selector
against launch seams covering spin, exact/custom omega, density/bounds/cursor,
block domains and f/unknown/SR fallback. Slurm job 13479 passes 24 Direct
streaming/high-angular GPU cases, native restricted/unrestricted fitted WB
composition against independent CPU oracles, cold/warm occupied-factor proof,
and complete exact/DF Fock/force/replay/geometry composition.

The traced WB96 moved run now observes 26,833,453,056 free bytes and keeps the
packed resident value layout. Its budget nevertheless drops optional complete
occupied U, causing repeated fitted-panel force projections. This remaining
endpoint work is not solved by a smaller force call graph. The cold traced
365.264-second observation is diagnostic, not a repeated speedup claim.
Final moved numerical checks and untraced repeated endpoints remain required.
All original logs, source inventory, build commands, compiler statistics,
failed build-directory-cache attempt and linked resource dump remain in the
ignored stage-eight artifacts locally and on n1.

## Revisit when

New angular domains, recurrence implementations or compilers change linked
frames. Keep higher-angular and optional-storage fallbacks explicit, and qualify
complete original/moved endpoints after any new resource-policy change.
