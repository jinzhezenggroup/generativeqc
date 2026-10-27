# H100 owner lifecycle after allocation-exception repair

This model-free run measures the production input closure at
`f0cb1eabd36153b37a1763925d162d1a9360445a`, including the exception-safe
pending-owner attachment. The source manifest SHA-256 is
`32fa9681327f988e9280d26625ef73575ac52529a44dab1427d9b3f02b7d216b`;
the standalone binary SHA-256 is
`e07162852dfece9bc630505eefcc853312178df7eddf73dffed5f84bef56182a`.
Later evidence-only additions do not change those measured production inputs.

All 53 actual CUDA/vendor returns succeeded. The retained inventory has 21
slots and 24 close operations. The owner closed, its bucket was absent, the
source lease was inactive and its pin count was zero. The native process was
reaped, its group was absent and post-close context/library identities matched.

The original 45/90/900-second Root-grant/native/global limits were unchanged.
The conditional exact-library check and grant happened in one control command.
The 512-MiB reservation and two 2-GiB envelopes were unchanged. The observed
synchronized free-memory difference was 375,390,208 bytes; it is not a process
or device peak, default allowance or universal H100 limit. The old v5 expired
grant remains unaccepted, without a deadline extension or replacement grant.

The separate HOST regression compiles the actual pending owner, resources,
graphs and eigen-handle implementation with the driver's attach rollback.
Its control discovers the allocation population, then every allocation is
fault-injected with CUDA/vendor seams that abort if reached. It failed before
the repair because the bucket remained published. After repair, 57 focused
tests pass; the formatted regression passes separately with its 221 staged LF
inputs matched. These are HOST/source results, separate from device execution.

Original remote bytes were persisted and rehashed before collection ended.
This public capsule redacts personal paths and transport-redacted boot IDs.
Its exported text is not the original byte stream: original digest witnesses
remain, while transformed original bytes are **UNAVAILABLE** here.

The full linker map is unused by replay and is compacted to its original
digest/length and linked-input excerpt. Its earlier exported text is recoverable
from the same capsule path at Git commit `e944b34082d1856d9a50dc235a5c9cfadf751d10`.
The current capsule does not claim to contain that complete map. All status,
failure, source and build/loaded-library identity records remain retained.

The source archive stores the complete V7 manifest and changed source texts
against the immutable `../v6/source-closure.json.xz`. Replay first verifies that
ancestor's byte length and SHA-256, reconstructs all 102 measured inputs, then
verifies every input against the V7 manifest. Both archives are required for
offline replay; no source, digest or qualification is silently substituted.

Optional capability bits remain zero. Installed SDK acceptance is
**INCOMPLETE** and scientific qualification is **NOT_RUN**. No molecule, SCF,
correlated-method, energy or force endpoint was requested. Parent scientific
and performance matrices remain open; the earlier v6 source identity is kept.

Replay the retained identities, outcomes, budgets and grant checks offline:

```console
python benchmarks/results/h100-provider-owner-20261027/v7/replay.py
```
