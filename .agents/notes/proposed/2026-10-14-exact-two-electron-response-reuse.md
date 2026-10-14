# Proposal: exact density-keyed Direct RSH integral response reuse

Status: experimental, default-off; 3/12/48 endpoints qualified, 96 acceptance pending
Date: 2026-10-14

The expanded 96-atom range-cache experiment still times out at moved/0 under
the original 1800-second protocol. Its fixed-density warm endpoints spend about
10.8 seconds in native integral derivatives. A separate exact-source device
probe at d4f0ba62a (n2 Slurm 2975) shows repeated density bytes but changing Fock, orbitals and
weighted-density bytes. Reusing the whole integral response would therefore be
incorrect: Pulay response depends on the current W.

Retain only the already contracted two-electron response within the immutable
prepared Direct RSH operator/geometry owner. Each lookup first authenticates the
current final density and stationary-weight leases, then compares every bit of
the already exported snapshot density. No new host matrix export, CPU oracle,
tolerance key, pointer-only key or epoch-only key is introduced. The one-electron
and Pulay pair runs anew. Geometry rebuild creates another owner; disabling the
experiment clears retention. An explicit shell-derivative fallback also bypasses
retention, preserving that diagnostic route. A failed or nonfinite response never populates it.

The single exactly sized host slab holds all spin density blocks and contracted
response. It is bounded by 64 MiB, a quarter of the caller allowance and the
remaining observed host envelope; its capacity is reported in retained owner
and derivative work accounting. Existing one-electron work reserves this live
capacity. An allocation refusal discards retention; an OOM with a live key
retries once after releasing that key. Default execution remains unchanged.

Host tests execute exact repeats, one-ULP differences, signed zero, spin/shape
changes, finite-result gates, retained-copy ownership, clear and byte ceilings.
Real-device independent molecular E/F, actual-hit traces, changed geometry,
failures, resource pressure and original full populations remain mandatory.
The trace reports exact key hits and actual two-electron response evaluations;
capacity is not called executed work. No performance or issue-resolution claim
is implied by this implementation.

Qualification at `b5ca4306e` (native inputs identical to compiled `78961f12d`)
passes all 72 same-library off/on original-protocol E/F calls. The 48-atom warm
median is 11.40746 s off and 7.49369 s on; cold/moved remain unchanged within
these observations. Six independent molecular and displaced-energy gates,
stale-state recovery, actual RKS/UKS exact-snapshot hits, memcheck and synccheck
pass. The initial UKS warm-hit assumption failed because its density bytes
changed; retain that failure and require a frozen authenticated snapshot for
its exact-repeat test. All samples, failures and identities are preserved in
`benchmarks/results/exact-integral-response-20261014/`. No 96-atom completion or
default-promotion claim follows from the smaller populations.
