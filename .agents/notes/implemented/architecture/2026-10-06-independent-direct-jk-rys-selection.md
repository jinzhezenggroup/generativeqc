# Independent prepared Direct J/K Rys lowering

Status: implemented, qualification pending
Date: 2026-10-06

## Decision

Compile derivative-free Rys alternatives alongside each incumbent streaming
Fock class in the same stable architecture build unit. The compiler intersects
the production streaming inventory with value IR/root/decoder/target legality.
The optional inventory is not a measured preference or a new coverage contract.
Absent classes, including one-root ssss/psss and the unsupported dddd decoder,
retain their exact incumbent/native paths. No method or molecule list selects
recurrence mathematics in the runtime.

Native prepared J and K owners freeze independent masks through
`GENERATIVEQC_DIRECT_J_FOCK_LOWERING=incumbent|rys` and
`GENERATIVEQC_DIRECT_K_FOCK_LOWERING=incumbent|rys`. The default is incumbent.
Disabled AOT classes remain disabled in the alternative inventory. Range K and
mixed-J contracts keep their existing fallbacks. Launch errors propagate rather
than retry into partially accumulated output.

HF also consumes these choices. A complete strict-FP64 bounded streaming owner
splits into J-only and HF-weighted K-only passes when either choice requests a
Rys inventory, or `GENERATIVEQC_DIRECT_HF_SEPARATE_JK=1`. Both immutable views
borrow the original topology and accumulate into existing Fock scratch. K
scatter applies the original RHF -1/2 or UHF -1 factor exactly once. The task
consumer bits together encode this weighted K contract; ordinary J-only,
raw-K-only and fused HF retain their established meanings. Native dddd uses the
same compiler-owned scatter contract. Partial/higher-l and mixed owners retain
the qualified fused route. Prepared replay freezes the choice; rebuilt geometry
creates a new owner.

## Timing and qualification

The shared optional CUDA component ledger records complete generated Direct-J/K
builds (density transforms, screening metadata, traversal, projection) and HF
streaming passes (queue reset and generated/native class work). HF's density
preparation/projection remain common work. Capture receipts are explicitly not
GPU timings. Traced endpoints are diagnostic; clean wall time is measured with
tracing disabled. Candidate selection does not establish profitability.

Independent Libcint gates must cover the complete compiler candidate inventory,
RHF/UHF raw J/K and HF-weighted K matrices, repeated and changed geometry E/F,
and empty/component tails. Complete 48/96-atom PBE0/HF warm/moved-warm comparisons
must retain numerical gates, semantic work, binary/source identities and
per-class resource/timing evidence. No default promotion is justified before
that evidence exists; in particular resource/spill costs may reject high roots.

Refs #2015, #2017, #1892. This builds on #2007 and the shared force scheduler;
force output layout does not force value J/K to choose the same lowering.
