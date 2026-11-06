# Decision: admit Direct nonlocal force batching independently of SCF panels

Status: implemented; H100 source-bound selector and frozen default integration qualified
Date: 2026-11-05

## Problem and decision

The nonlocal preparation branch requires a fitted strict RKS owner before
considering either SCF or force tiles. Large Direct WB97M-V consequently retains
256 force points, although force batching already has independent capacity,
geometry and stationary-provider admission.

Permit that branch for an automatic force request from a strict fused RKS owner.
Keep fitted admission for nonlocal SCF panels: the measured Direct owner still
uses 256 SCF points. Pass the fused strict RKS capability to the existing force
policy. Preserve its all-electron, through-d, 48–128 atom, 512-AO, 524288-point,
device, free-memory and native stationary guards. Explicit force bounds keep
their previous preparation path; resource promises and bounded drained fallback
remain authoritative. No grid, recurrence, precision or stopping rule changes.

## Source-bound selector evidence

The trial uses source `87bbc0b4f20f7696cfd22974bbe5b207e7d69c38`, native identity
`d60aea30bf2891a8d5c6976f59ee7f8a3505cd83af5c123dde48093e4b3ea604`, core SHA256
`4702ac35b5fada834aa15d0569a4f750d1dc756dbca731090ed5b5c6f49d64a9` and one
H100 80 GB, GPU UUID `GPU-4b9a1102-bbb3-758f-43b5-a81e9594e892`.

Three rotating fresh-process OFF/1024/GPU4PySCF rounds retain all nine processes
and 36 complete energy/analytic-force calls. Actual force tiles are 256/1024;
SCF tiles remain 256. Every output passes the original independent dense/CPU-spot
gates and the stronger matched checks. Cold target work remains 18 iterations /
19 Focks, warm work one iteration / one Fock. Larger force tiles may change local
AO work; equal SCF work is not a force-work census.

The paired candidate/OFF cold ratios are 0.975913844, 0.979996023 and
0.975474375: median cold reduction **2.4086%**. Warm ratios are 0.885823212,
0.882235373 and 0.881085545: median warm reduction **11.7765%**. The original
2% cold and 5% warm-regression gates pass without rounding the decision. The
median paired GPU4PySCF/candidate cold ratio is 0.519732; this does not establish
WB97M-V cold parity with GPU4PySCF.

Cold includes constructor/preparation, SCF, forces and host export after CUDA
context initialization, with verified compiled artifacts retained. Installation,
AOT compilation and process startup are excluded. First-call outliers remain.
These static original-matrix water-96/def2-SVP results are not a general molecular
or cross-device ranking; the separate default integration is recorded below.

## Rejected companion and verification boundary

A separate default-16/32-GiB optional range-residency pilot passes its numerical
gates but has cold ratio 1.001562 and warm ratio 0.998499. It fails the predeclared
continuation gate; no follow-up rounds run. A requested ceiling does not prove
the amount of storage actually retained. Do not promote it or combine its
timings with the force-tile study.

The complete work capsule retains all twelve processes / 48 E/F calls and the
rejected pilot: SHA256
`6e606531c948fad76ccb2ee2d81bb454393fd404d3437d6b2788e29d0b00c441`.
Authenticated independent offline readback recomputes every scientific gate,
actual force tile, unchanged scientific setting and paired statistic. Its
combined performance flag is false because the residency study was rejected;
the independently predeclared force study passes. Preserve both outcomes.

Eight focused compiled CPU policy, capacity, failure and retirement cases pass
on the existing CPU environment using verified sccache 0.16.0. Initial local
packaging and zero-test import-path failures remain retained. No GPU acceptance
is inferred from those host probes. The independent selector evidence does not substitute for changed-default device admission.

## Frozen default integration and later main

The source `db0788b2032ba99029d1e7a7a995380777d4981f`, native identity
`0ee40c0ab15addfe978f93ae293c6ea5fd0cacec4147161b283ee5159134a73e`, was built
and qualified on one H100 `GPU-a21ed680-6184-b931-377d-81ff7cf9f52e`. All180
fresh processes /720 complete E/F calls in the unchanged original30 matrix pass
the original40 independent dense/CPU-spot numerical references. WB96 Direct
observes automatic prepared force1024 /SCF256 in every cold/warm sample, without
an explicit tile override. Cold wins25/30, complete-set geometric mean
GPU4PySCF/native1.646099582645; warm wins26/30, geometric mean1.438302760540.
The five cold nonwins and all adverse process ratios remain retained.

This frozen cohort precedes upstream #3006. The later normal main merge changes
only `src/dft/cuda_ks.cpp` among native inputs; it keeps this force policy intact.
Its necessary H100 integration is restricted to original WB48/96 Direct, with
eager-anchor OFF and unset/default arms on the same combined source. Do not pool
those later-source times into a hybrid latest30 aggregate or multiply their gains
by the force-selector result. See the source-qualified publication under
`benchmarks/results/h100-cold-force-fixed30-20261105/` and issue #3004.

## Revisit when

New representative geometries or actual resource failures show a regression in
complete E/F. Retain explicit force requests, strict scientific settings, independent
numerical references, the drained fallback and the existing resource promises.
A100/5090, UKS/CBS and displaced-geometry acceptance require their own evidence;
this H100 stage does not extend those claims.

## Postqualification upstream integration

The normal merge of #2991 updates CPU UKS final closure and ECP qualification.
A bounded source/byte review confirms unchanged RKS force/default/timer paths and
prior scientific packet bytes; only current checked Libxc identities regenerate.
Reuse the source-qualified H100 evidence within that scope. Do not relabel the
later head as a measured matrix or repeat original720/72 reductions/device gates
for this unrelated CPU integration. The public review supplement retains the
actual parent/source hashes and original WARN limitations.

The following normal #3011 merge updates CPU full-range Hessian admission and
second-order diagnostic exports. CUDA short-circuits the new CPU eligibility arm;
first-E/F routing and native RKS force/default/timer inputs remain unchanged.
The second bounded public review supplement verifies unchanged scientific payload
and original/v1 review bytes. Extra Python bookkeeping carries no new timing
claim. Retain the same measured-source boundaries and original WARN; unrelated
interface integration does not require repeating completed numerical/device work.

The next normal main merge retains #3009 CI maintenance and #2985 low-level CPU
UKS response admission. Its only executable runtime edit is in the explicit
second-order binder; ordinary CUDA cold/warm E/F routing and native/default/timer
inputs remain unchanged. The third bounded supplement retains source/byte checks
and the original WARN. It does not qualify upstream UKS science or a later-head
timing population and requires no original numerical/device replay.
