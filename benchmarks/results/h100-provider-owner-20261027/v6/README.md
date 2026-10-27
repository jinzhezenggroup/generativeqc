# Follow-up H100 provider-owner engineering run

This run exercises the lifecycle fixes at source base
`a298c803a9a3f933f81ba26f9abda16247780e48`. Its complete measured input
manifest is `dcf84602a429e70a2030c86c7b6d73333b4fb3448275526004d0674e96c8d97f`;
the standalone binary SHA-256 is
`40c3dc6a57d88b14a5645c61a399ceebad01f52bfa11c4fc7f239a4dcd23732c`.
It requests no molecular, SCF, correlated-method, energy or force endpoint.

All 53 actual CUDA/vendor calls returned success. The 21-slot inventory and
24 close operations confirm that the native owner, bucket, source lease and
library pins closed. The native process was reaped, its group was absent,
and the post-close current context and loaded-library identities matched.
The explicit 512-MiB reservation and two 2-GiB envelopes remain unchanged;
the observed synchronized free-memory difference was 375,390,208 bytes.
That observation is not a device/process peak or a universal allowance.

The preceding v5 attempt missed its original 45-second Root grant window.
It remains unaccepted, and its original deadline was not extended. In this
separately bound follow-up, Root's already reviewed four-library hash/build-ID
policy is checked against the current native identity in the same bounded
control command that writes the grant. This removes the intervening UI/model
round trip while retaining the original 45/90/900-second limits. The policy,
actual grant clock and failure distinction are retained in the receipt.

The attached 221-case HOST receipt covers the subsequent upstream named-KS
integration. It is separate evidence: this device run does not qualify that
later entire head or an installed SDK. Optional capabilities remain zero,
installed-SDK acceptance remains **INCOMPLETE**, and science is **NOT_RUN**.
The full #2727/#2728 scientific and performance scopes remain open.

Inspire redacted boot IDs before export; personal storage/launcher paths are
also redacted from this public capsule. Exported text is not the original
remote byte stream. Original digest witnesses remain, while transformed
original bytes are **UNAVAILABLE** here. No missing status or telemetry is
zero-filled, and private operational paths are not published.

Replay the source identities, outcomes, budgets and conditional grant checks
without a GPU, compiler or network:

```console
python benchmarks/results/h100-provider-owner-20261027/v6/replay.py
```
