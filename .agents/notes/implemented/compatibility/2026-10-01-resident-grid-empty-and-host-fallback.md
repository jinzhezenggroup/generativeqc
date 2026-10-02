# Decision: publish empty density leases and preserve the explicit host fallback

Status: implemented
Date: 2026-10-01

## Problem and repair

An empty CUDA feature tile advanced the generation and published its view but
left contracted density jets unavailable. Publish that zero-length lease when
features are requested in density mode; orbital/non-feature tiles remain
ineligible, and the getter continues to reject stale generations.

An intentionally HostUnfused KS owner has no resident molecular grid. After the
exact final-density token and binding are validated, return NOT_IMPLEMENTED for
that specific absent-grid case so the checked host-grid route can run. An absent
DeviceFused grid, invalid binding or wrong device remains an error. Stale-token
failures must be returned before considering fallback.

The device regression also needs to drain an injected asynchronous failure while
its grid lease is alive. A replay callback must request energy only to invalidate
the old owner without recursively calling its own force-completion hook. Route
labels follow the actual prepared stationary source, and the budget rejection
constrains executed grid work rather than an elided AO primitive page.

## Evidence and boundary

Host-compiled tests execute the production publication/getter and resident-grid
admission bodies with CUDA/owner stand-ins. They cover empty density versus
orbital leases, generation revocation, and the host/fused, present/absent,
valid/invalid-density, current/stale-token and correct/wrong-device combinations.
They establish control-flow contracts, not NVIDIA arithmetic or synchronization.

The external receipt on PR #1659 reports 40 passing tests only after its local
production/test patch, and retains an earlier teardown failure as a failure.
Those patch bytes are not available here. This independently implemented repair
is not relabeled as byte-identical or a new-head device pass. Keep the unchanged
independent force/FD gates and rerun them on the published head before merge.

Reference: https://github.com/jinzhezenggroup/generativeqc/pull/1659#issuecomment-5941984406

Agent: dot
