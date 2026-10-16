# Decision: Share native CUDA KS host inventory and fleet lifetimes

Status: implemented
Date: 2026-10-16

## Problem

Python owned CUDA KS host sizing and fleet composition while the native CLI
carried separate arithmetic. Its decomposition omitted the matrix and structured
product binding reservations already retained by the prepared CUDA owner.
Other conservative metadata bounds mean this omission alone does not prove an
endpoint RSS underbound. Host-unfused diagnostics also omitted retained XC
staging from their item subtotal, although the public plan already charged it.

## Decision

Generalize the existing CPU host owner to `scf::KsHostResourceShape`,
`ks_host_resources` and `KsHostFleetResources`, preserving CPU aliases and the
old CPU ABIs. Execution storage distinguishes CPU, CUDA fused and CUDA
host-unfused XC. CUDA solver bounds use the existing solver threshold and
workspace authority. CUDA shapes never evaluate CPU integral fourth powers or
inherit CPU angular admission. Pure tensor constants are shared with actual
bindings, without including CUDA runtime headers in CPU shape queries.

The additive private CUDA host fleet ABI has eighteen scalar fields per item:
the existing seventeen CPU fields and one canonical host-unfused boolean. It
returns fifteen aggregate and sixteen per-item diagnostic values. Two scalar
passes validate the complete fleet before publication with O(1) scratch.
Malformed later items, overlapping buffers and signed-int64 overflow preserve
the full output. Successful composition does not allocate, discover providers,
construct CUDA contexts or execute chemistry.

## Consumer lifetimes and compatibility

A retained consumer allowance coexists once with every item and replay. Python
supplies its existing snapshot-grid and force capacities; the ordinary native
CLI supplies zero. A transient floor instead shares the serialized workspace
maximum, preserving the old CPU force contract. Neither input admits a force
consumer or changes its eligibility policy. Multiplying a CUDA KS CLI fleet a
second time would double-charge retained owners, so generic batch multiplication
applies only to paths that still return a per-item host total.

Only an absent additive symbol permits the explicit old-library portable host
path. Selected native refusal and incoherent diagnostics fail closed. Python
checks wire consistency but does not repeat capacity formulas on the native
path. CUDA host, device and ECP provenance remain distinct. The legacy
host-unfused retained diagnostic now includes already-charged XC staging.

This extends the preceding
[CPU fleet decision](2026-10-15-native-cpu-ks-fleet-sizing.md), not CUDA device
fleet ownership, full provider allocation/lease authority or complete #2227 R2.

## Validation boundary

Native tests independently pin a small component vector, solver thresholds,
host-unfused RKS/UKS storage, large CUDA shapes, retained versus transient
allowances, exact limits, transactional append and malformed ABI transport.
Python compares ragged portable bounds, compiled publication, exact budgets,
missing-symbol compatibility and coherent corruption without provider loading.
CLI tests interpose only synthetic device leaves and retain the real host
owner, checking homogeneous batching, exact/below limits, binding reservations
and the absence of Python consumer allowances. These host-only controls are
not physical GPU or scientific endpoint evidence; those require separate
source-matched CPU/CUDA qualification and finite Slurm GPU jobs.

## Rejected alternatives

Keeping separate CLI formulas would leave two authorities. Treating retained
consumers as a workspace floor would lose their coexistence with preparation.
Reusing CPU formulas for CUDA would impose irrelevant fourth-power overflow and
angular restrictions. Removing portable metadata planning would break older
SDKs; using it after native failure would hide malformed capacities.
