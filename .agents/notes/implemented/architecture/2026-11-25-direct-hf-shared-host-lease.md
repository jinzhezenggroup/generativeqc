# Decision: Attach the shared native host lease to Direct CPU HF calculations

Status: implemented
Date: 2026-11-25

The shape-only CPU HF fleet can quote a peak but cannot prevent simultaneous
consumers on separate contexts from exceeding one shared cap. Reserve retained
and workspace bytes from the actual validated system and solver controls during
native preparation. Reuse the existing lease, calculation handle lifetime,
and native workspace lock rather than adding a second policy or allocation path.

Ordinary CPU HF reconstructs the Fock and integral owner inside each execution.
The execution holds the shared lease's workspace lock across that reconstruction,
SCF, force finalization, and native output publication. The cached CUDA Fock
branch is not covered. Numeric phases do not move into retained storage when an
energy call is followed by a force call. System/configuration copies and the
bound remain live until destruction; a failed prepare releases its lease.

Keep this API opt-in and limited to all-electron Direct FP64 RHF/UHF without
preliminary providers. The legacy SDK remains unchanged. Never reuse this
partition for a persistent integral owner, an initial-guess phase, or a fitted
provider without qualifying its additional lifetimes. The cap is modeled native
capacity, not memory allocator interception or process RSS enforcement.
