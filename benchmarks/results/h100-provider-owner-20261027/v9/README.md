# Current-input H100 owner engineering receipt

This describes the recorded historical run. First restore the complete family
using the [offline recovery instructions](../README.md#historical-snapshot-and-offline-replay).
Archive paths and the original replay program refer to that verified snapshot;
summary JSON bytes, original source identities and scope remain unchanged.

The measured production inputs are from `964faab623f6bf9fb2a99f39a772924f382a41f4`:
96 repository inputs and nine separately identified generated engineering inputs.
This model-free run returned 53 successful CUDA/vendor statuses, 21 resource slots
and 24 close operations. Owner, bucket, lease, pins, process groups and the private
compiler service closed explicitly; post-close device/context/library identities
matched. Optional capabilities remain zero. **Installed SDK acceptance is
INCOMPLETE; no molecule, solver, SCF or E+F was executed.**

Run `python -I -S -B .artifacts/h100-provider-owner-20261027-original/benchmarks/results/h100-provider-owner-20261027/v9/replay.py`
with Python 3.10+. It validates all 105 source inputs through the immutable V7/V6
anchors, every public generated-text hash, native statuses, close inventory,
identity matching, capacity arithmetic, original deadline/seal witnesses and
explicit scope. It neither loads native libraries nor executes a GPU/oracle.

The new packet retains the original A=512 MiB, B_ref=B_endpoint=2 GiB,
240-second build, 45-second grant, 90-second native and 900-second global limits.
These are explicit engineering limits, not defaults or universal H100 allowances.
The observed free-memory delta is a diagnostic, not an allocation ledger.

Two control-layer failures remain in the payload. V8 refused a 145-byte Unix
socket path before cache-server, compiler or native startup. The changed V9
harness places only its short, exclusively owned IPC endpoint in `/tmp`; source,
data and results stay in the project. Its socket/directory are removed only after
confirmed owned-actor closure. V9 compiled and linked, but its unchanged collector
could not retain a 4.6-MiB linker map within its 4-MiB text cap. No failed partial
copy was repaired or overwritten. A separate Root observer, bounded to 8 MiB of
original text and 4 MiB compressed transport, retained all 95 generated text
members (5,115,835 original bytes), including the complete linker map. It reused
the original terminal authentication checks and rechecked all 499 dependencies.
The same completed build was used; no build/native retry or deadline extension
occurred for this observation. Original and partial evidence remain separate.

`receipt.json.xz` contains every generated text member, not a map excerpt. Private
paths, boot/device IDs and nonce have public aliases. Public hashes cover these
actual exported bytes; separate original SHA/length/mode witnesses cover the
private originals. Redacted original bytes are unavailable from this public
capsule. Exact originals and the full original compressed archive remain in the
user's persistent project storage. Their hashes do not make transformed text
byte-identical to an original. The auxiliary Root observer is outside the compiled
105-input closure and its source is retained separately in the receipt.

The exact combined production tree passed 226 HOST tests and 25 subtests using
verified sccache 0.16.0. This engineering receipt is separate from the `aa6e05ea`
SDK/small/48-atom science campaign, historical V4/V6/V7 engineering, the closed CBS
pilot and all positive-performance goals. It does not close their parent issues.
