# Decision: publish CUDA KS Fock counts only with complete execution evidence

Status: implemented
Date: 2026-10-05

The PBE0 endpoint campaign retained `null` Fock counts even though the legacy
CUDA KS executor increments an actual build counter. The public batch adapter
still rejected every CUDA counter under an older CPU-only contract.

The existing getter now admits CUDA KS results only when both completeness
flags of the owner's operator census are set. It copies the recorded build
count; it does not derive work from iterations or from final-audit events.
CPU admission is unchanged. No ABI field or production GPU operation is added.

Chunk/replay can submit speculative bodies after the terminal iteration, and
warm-to-cold retries can discard an earlier attempt. Their partially observed
counts must remain unavailable. A complete final attempt never establishes a
complete retry total. Unexecuted and malformed requests still invalidate the
previous counter before returning.

The native public-batch regression compares the getter with actual
Fock-assembly operator counts. Host tests cover incomplete census flags,
non-KS CUDA owners, retries, and stale-result invalidation independently of
solver iterations. This change does not populate historical timing records:
new source-matched endpoint runs are required for new public counts.

Revisit the conservative exclusions when those execution owners instrument
all submitted physical work and all discarded attempts. Refs #1895, #1912.
