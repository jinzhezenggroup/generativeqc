# Proposal: admit Direct nonlocal force batching independently of SCF panels

Status: proposed; selector performance qualified, changed-default device integration pending
Date: 2026-11-05

## Problem and proposed decision

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
or cross-device ranking, and are not measurements of this proposed default.

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
is inferred from those host probes. The changed default still requires its
source-bound native build and actual device admission; publication and broader
original30 results remain pending under issue #3004.
