# Decision: Admit cold Fock proposals in the same saddle SCF lease

Status: implemented
Date: 2026-11-24

This extends [warm-seed admission](2026-11-24-saddle-warm-seed-host-admission.md).
Cold Fock damping owns one additional AO-square FP64 proposal while ordinary
primal SCF workspace is live. Call the existing policy's `proposal_bytes` on
the admitted native basis before provider setup and add that reservation to
the primal phase, not to the detached warm seed or to every moved frame.
The aggregate cap does not replace the independent proposal ceiling and does
not reinterpret or skip an explicit damping prefix when storage is short.
Model/method validation, warm-root handoff, and ECP refusal remain unchanged.

The ECP exclusion above was superseded for the two existing saddle protocols by
[`2026-11-24-saddle-ecp-host-admission.md`](2026-11-24-saddle-ecp-host-admission.md).
