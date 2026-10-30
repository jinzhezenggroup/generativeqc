# H100 provider-owner engineering qualification

## Historical snapshot and offline replay

This is a **historical model-free engineering report**, not a current-source
molecular or performance qualification. Original summary JSON bytes stay in this
checkout. The complete source/build/export receipts, failed control observations
and original replay programs are pinned by `historical.manifest.json` to published
commit `15ab66a68a061355fda833a93e88484dbfee6360` (20 files, 946365 bytes).
No independent molecular reference or active numerical-test input is moved.

Restore the exact complete snapshot before using any archive paths or replay
commands below. The existing restorer verifies every size/SHA-256 before creating
a new output directory. It uses only existing local Git objects: no implicit
fetch, upload or execution. A shallow/source-only checkout must explicitly obtain
the pinned commit; normal unit tests do not require this historical snapshot.

```console
python tools/restore_retained_evidence.py --all \
  --manifest benchmarks/results/h100-provider-owner-20261027/historical.manifest.json \
  --output .artifacts/h100-provider-owner-20261027-original
```

With Python 3.10+, all four original replays run from that restored copy:

```console
python -I -S -B .artifacts/h100-provider-owner-20261027-original/benchmarks/results/h100-provider-owner-20261027/replay.py
python -I -S -B .artifacts/h100-provider-owner-20261027-original/benchmarks/results/h100-provider-owner-20261027/v6/replay.py
python -I -S -B .artifacts/h100-provider-owner-20261027-original/benchmarks/results/h100-provider-owner-20261027/v7/replay.py
python -I -S -B .artifacts/h100-provider-owner-20261027-original/benchmarks/results/h100-provider-owner-20261027/v9/replay.py
```

This migration was checked by recovering all 20 original files byte for byte,
running all four original replays, and reconstructing the three earlier V4/V6
compressed archives at their original SHA-256/lengths. All report
`SDK_INCOMPLETE; SCIENCE_NOT_RUN`; redacted private originals remain unavailable.
The original `restore_original_archives.py` is in the same restored snapshot.
The unchanged summaries below refer to those historical archive bytes, not to
missing live publication inputs. Retention limits and scientific gates do not
change. This uses existing Git history, not a Release or external backup.

## Original engineering report

This is one model-free run of the pending native provider owner on an H100 80GB,
driver 595.58.03, CUDA runtime 12.9. It qualifies initialization, bounded resources,
source/event/graph ownership, and explicit teardown. It requests no geometry,
molecular SCF/correlated solve, energy, forces or independent scientific reference.

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
python .artifacts/h100-provider-owner-20261027-original/benchmarks/results/h100-provider-owner-20261027/replay.py
```

The [v6 follow-up](v6/README.md) retains the post-fix resource-lifecycle run
and its unchanged finite grant/native limits. Each capsule keeps its own
measured source and binary identity; neither qualifies a later installed SDK.

## Lossless shared storage

The earlier published V4/V6 receipt JSON bytes now share `v4-v6-receipts.tar.xz` in the family root. The V4 source closure is a three-file delta against the unchanged V6 source archive, which V7 also requires. Both historical replays verify recovery of the exact original compressed archive bytes before checking their unchanged engineering outcomes. Original archive SHA256/length fields remain in each summary. This storage change does not recover private redacted remote bytes or establish SDK/scientific acceptance. Recover the exact earlier compressed archives with the original restorer in the
verified snapshot, using a fresh output directory:

```console
python -I -S -B .artifacts/h100-provider-owner-20261027-original/benchmarks/results/h100-provider-owner-20261027/restore_original_archives.py \
  --output .artifacts/h100-provider-original-compressed
```

The [V9 current-input receipt](v9/README.md) retains the integrated `964faab6`
production closure, full generated text and separately bounded Root observation.
V8 path refusal and the original V9 collector limit remain failures. This is
model-free engineering evidence; installed SDK/science remain unqualified.
