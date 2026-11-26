# Decision: Reserve CPU saddle snapshots separately from native seed restore

Status: implemented
Date: 2026-11-24

This extends [the CPU saddle host envelope](2026-11-24-cpu-saddle-host-envelope.md).
The last accepted snapshot can outlive its batch and remains needed while the
trial is differentiated and its own snapshot is captured. Charge both snapshot
payloads outside the prepared batch, even for the first geometry. Check each
pre-SCF estimate against the actual prepared seed shape; the independent root
cap still applies to capture.

Restore copies a candidate density and validates it against its **source** AO
metric, not the moved trial's overlap. The native lease reserves both normalized
source copies (including the raw AO expansion), integral scratch, and the dense
validator's matrix products and Jacobi storage. Restore and capture share the
lease workspace lock with the other electronic stages. The modeled restore
workspace is serialized with primal work; the detached snapshots are not.
Neither validation nor seed publication rules change. Cold Fock damping and
ECP remain outside the aggregate saddle contract.

The damping exclusion above was superseded by
[`2026-11-24-saddle-cold-damping-host-admission.md`](2026-11-24-saddle-cold-damping-host-admission.md).
