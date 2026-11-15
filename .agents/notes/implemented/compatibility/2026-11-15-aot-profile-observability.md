# Decision: separate AOT profile identity from generated coverage

Status: implemented
Date: 2026-11-15

## Problem

Issue #3086 identifies a native profile test that requires tuned SM120 solely
from the physical compute capability. A documented forced-portable SM120 build
therefore fails despite resolving correctly. Meanwhile, an unmatched device and
a matched portable bundle both report `portable=true`, even though only the
latter may have generated force/Fock classes.

## Decision

Keep selection, kernels and portable packaging unchanged. Derive native test
expectations from the generated build inventory, not a GPU model assumption.
Test CUDA builds with shell AOT disabled as well as enabled.

Extend the existing tuning-device descriptor rather than introduce another
public probe. Keep actual SM and profile flags separate from registry state
(`matched`, `no_matching_kernel_set`, `aot_disabled`) and environment-enabled
force, Fock and mixed-Fock exact-class masks. A matched empty inventory remains
distinguishable from no matching inventory; environment masking does not change
registry identity. Masks describe generated availability, not final per-workload
resource/precision admission or numerical qualification.

Update the Python ctypes consumer together with the pre-release C descriptor.
Existing full-size descriptor validation rejects obsolete clients before a
device probe. Do not add a compatibility shim. Exclude all selection diagnostics
from user-local hardware compatibility identity, as was already done for the
official profile and portable flag; retain actual device limits and source ABI
identity checks.

## Rejected alternatives

- Force `sm_120` whenever the GPU is SM120: contradicts explicit portable intent.
- Interpret `portable=true` as generated coverage: loses unmatched/disabled and
  environment-masked distinctions.
- Enable optional portable variants to align metadata: changes first-use/code
  size policy without endpoint qualification and is outside this repair.
- Rewrite the dispatcher or copy tuned schedules to other architectures: not
  needed to repair observability, and would require independent device gates.

## Evidence and scope

The host-compiled regression reproduces the original SM120 forced-portable
failure (one failing case and three passing cases). Executable tests use actual
generated registries and the production C API with deterministic CUDA discovery
and launch-body stubs. They distinguish automatic SM120, forced portable SM120,
portable SM80/SM90, multiarch selection, unmatched devices, an empty generated
registry, and shell-AOT-disabled builds. Public diagnostics are checked against
dispatch and environment masks, including caller-device restoration.

These are host contract tests, not new on-device energy/force or endpoint
performance evidence. No launch route, mathematics, numerical acceptance gate,
build default, or retained measured evidence is promoted by this change.
P1b policy separation, P1c cross-architecture capability qualification and P1d
global-default promotion remain separate work under #3086; #3084's benchmark
review blockers are not resolved by this scoped P1a repair.

The derived Libxc snapshot is refreshed only for its two live source identity
fields. Original measured identities, receipts, stale flags and qualification
outcomes are preserved byte-for-byte at the decoded-field level; this refresh
does not establish new scientific acceptance.

## Master metadata transition

Master `c99e6543e1fe0f4ea6832d58381d30e002e04c67` adds scoped Libxc
dependency inventories and strict predecessor/acknowledgement handling. Merge
it without rewriting the published P1a history. Preserve its exact current
summary in `native-qualification-previous-c99e6543e1.json.xz` before regenerating
the live snapshot with the upstream owner. Its zero dependency-current products
and 55 blockers remain unchanged. The preceding 15 explicit demotions and the
older 7d7dc89b5c predecessor remain in their original published Git history and
retained file; stale acknowledgements are not reused for a new transition.
This is not new Libxc science or permission to relabel earlier AOT/GPU evidence.
The original two-field refresh described above belongs to the earlier 8d2ad48
P1a revision, not a bypass of the new scoped provenance contract.

## Revisit when

Revisit the diagnostic inventory when additional independently admitted runtime
routes need observability. New global defaults still require source/device-matched
cold, warm, moved-geometry, numerical and resource qualification, not just a
successful registry match or local warm tuning result.
