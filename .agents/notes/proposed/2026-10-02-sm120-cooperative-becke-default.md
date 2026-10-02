# Qualification: automatic cooperative Becke scheduling on sm_120

Status: candidate implemented; complete endpoint qualification pending
Date: 2026-10-02

The existing cooperative kernels remain scientifically unchanged. Device tests
have exercised the independent complete force oracle and the 96/128-atom tiled
geometry probes. The candidate changes only automatic selection: sm_120 owners
with 12–128 atoms request cooperation, subject to the existing shared-memory,
thread, allocation and native-device checks. Smaller systems and unqualified
architectures retain the generic route. Explicit `False` still forces generic
execution; explicit `True` retains the previous diagnostic opt-in semantics.

The choice belongs to the shared compiler resource planner, so planned evidence
and actual source-owner configuration agree. No extra persistent memory, point
screening, arithmetic precision change, or grid partition change is introduced.
The existing 256-point public force and SCF tile sizes remain unchanged.

Exploratory 1024-point force tiles improved 12/24/48-atom endpoints, but the
96-atom dense grid plan exceeds the existing 512 MiB additional-consumer cap.
Do not promote that unconditional tile size or raise the cap to hide the cliff.
The 512-point 96-atom diagnostic is separate from the public-default campaign.

All experiments use finite Slurm RTX 5090 allocations. n4 could not execute
the tests: its loaded kernel module is 580.173.02 while system libcuda is
580.178.04. These environment failures are not device qualification. n1 runs
the independent derivative, cooperative geometry and Compute Sanitizer gates.

Before moving this note to implemented, require clean memcheck, initcheck and
synccheck plus all cold, moved and ten warm endpoints at each README size.
Preserve timing comparisons on one host and actual source/binary identities;
partial diagnostics and cross-host comparisons are not default-speedup claims.
