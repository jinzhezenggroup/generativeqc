# Decision: bind CUDA qualification to the complete selected profile package

Status: implemented
Date: 2026-11-02

## Problem

The focused WB97M-V production builds selected only `wb97mv_rks` and
`wb97mv_uks`. A100 molecular job 224764 passed the WB/mixed-range controls but
failed fourteen PBE-D4 cases because the qualification package lacked PBE.
Adding only the PBE s/p and s/p/d shared libraries in 224788 removed the missing
sidecar artifact but did not admit the native D4 resource owner.

The stationary profile selection also determines the static native registry
inside the core. A sidecar cannot add a native profile to an already linked
core. The global D4 planner correctly refuses an electronic owner without a
native stationary capacity receipt; falling back to compatibility execution
would not validate that resource contract.

## Decision

Use a separate OFF D4 qualification package built with PBE in the native
registry as well as its sidecars. Preserve the original OFF WB production core
and its timing receipts. Record source identity, CMake profile selection and
every core/sidecar hash for both packages. A common scientific source identity
does not make different binary packages interchangeable.

The D4 retry exercises the original complete resource tests, including the
independent energy/force reference and minimum-budget failure/recovery. Preserve
both failed runs and map recovered test IDs to the later retry. Do not skip the
unavailable method or count its refusal as successful D4 qualification.

## Consequences

Future narrow qualification builds must include the profiles used by adjacent
regression tests. Building a missing shared library alone is insufficient when
the test requires native ownership. Production timing claims stay attached to
their measured binary; an expanded registry needs its own applicable controls.
The normal complete build's profile policy is unchanged.

## Evidence

A100 job 224790 passes all 38 D4 controls on the expanded OFF package.
Together with the unaffected controls from 224764, the replayable unique
population has 146 passes and one skip, with fourteen explicitly mapped
recoveries. Both earlier failure logs remain in
`benchmarks/results/wb97mv-review-a100-20261102/`.
