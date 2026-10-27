# H100 provider-owner engineering qualification

This is one model-free run of the pending native provider owner on an H100 80GB,
driver 595.58.03, CUDA runtime 12.9. It qualifies initialization, bounded resources,
source/event/graph ownership, and explicit teardown. It requests no geometry,
matrix solve, SCF, energy, forces or independent scientific reference.

The explicit initialization reservation is 512 MiB inside separate 2 GiB
reference and downstream envelopes. The synchronized free-memory difference was
375,390,208 bytes (358 MiB). This observation includes runtime/context effects;
it is neither a process peak nor a universal H100 allowance. The legacy 96 MiB
per-provider default remains unchanged.

The actual run returned 53 successful CUDA/vendor status records, a 21-slot
inventory and 24 explicit close operations. The native owner, bucket, access
lease and library pins closed; the native process was reaped and its group was
absent. Post-close current context and loaded-library identities matched their
original observations. The original 90-second native window includes setup,
controller acknowledgment, operation, close and verification; no new cleanup
window was added. Existing compiler caches were retained and the task's
foreground compiler service was closed. Eight source objects were compiled;
the whole SDK was not built by this probe.

`source-closure.json.xz` contains the exact engineering source closure and
its manifest, including the actual bridge and real API forwarders. The manifest
is the authoritative source inventory if its count changes. It records the
repository base and the preceding engineering source identity separately.
The report is tied to the retained source and binary hashes, not automatically
to another build or later branch head.

`receipt.json.xz` retains the exported generated build/native records and the
36-test HOST C/C++/Python contract transport. Inspire redacted host boot IDs and
temporary connection URLs before local export. **Redacted exports are not the
remote originals**: the inventory retains original SHA/mode witnesses, but the
redacted original bytes are unavailable in this public capsule. API status,
allocation, close and numeric budget fields remain present. No scientific
arrays exist for this engineering run. Unknown local launcher/helper counts are
not zero-filled or promoted to complete process-cost accounting.

Earlier EOF/exit-race and staging/compile failures remain separate local
engineering records; they are not relabeled GPU or scientific failures. This
accepted current-source run does not replace their outcomes, qualify previous
PBE0/CBS attempts, enable an SDK capability, or complete #2727/#2728/#2731.
Installed-SDK and scientific acceptance remain **INCOMPLETE / NOT_RUN**.

Replay without a GPU, compiler, network or new model call:

```console
python benchmarks/results/h100-provider-owner-20261027/replay.py
```
