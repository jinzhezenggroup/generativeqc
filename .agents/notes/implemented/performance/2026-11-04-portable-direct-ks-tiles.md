# Decision: portable preferred Direct SCF and force tiles

Status: implemented
Date: 2026-11-04

## Problem

PR #2948 initially replaced the A100-name profitability gate with an SM80 gate,
then required exact local profiles before enlarging any native Direct tile.
Neither choice follows from a portable cost model: hardware identity does not
establish the best batch size, but missing exact-workload evidence does not
establish that 256 is the better default either. The user approved preferring
1024 across GPU models while preserving resource and workload boundaries.

## Decision

Automatic native Direct SCF and force owners prefer 1024 on every supported CUDA
device within the existing full-range, device-fused, gradient-dependent,
strict-FP64 RKS envelope. Preserve zero application arithmetic budgets,
all-electron shells through d, 48–128 atoms, at least 512 AOs and 524288 grid
points, at least 1 GiB free device memory, and absence of numeric capacity
promises, active resource ledgers or reference-memory budgets. Force admission
still requires a native stationary binding. Direct tiling does not query device
names, compute capability or FP32/FP64 throughput; fitted scheduling is unchanged.

Explicit SCF/force bounds and their inheritance remain authoritative. Automatic
SCF preparation and force-controller allocation failures retain the existing
single drained retry at 256; non-allocation failures and a second failure still
propagate. Unsupported workload/resource domains retain 256. Local DFT profiles
remain optional and exact-identity checked, not a prerequisite for the default.
No implicit cold-start benchmark, kernel, grid, precision or screening change is
introduced.

## Evidence and limits

The [original A100 qualification](2026-11-03-a100-direct-automatic-tiles.md)
retains complete E+F measurements, independent numerical gates and semantic work
counts for the original measured source. Larger batches improve its six measured
Direct methods despite increasing local AO work: PBE0 force projection FMA-pair
counts grow from 207471842560 to 269510873344. These are historical semantic
counters, not hardware FLOPs or new-source qualification.

The [three-device DF experiment](../../../../benchmarks/results/portable-integration-20261101/README.md)
is a counterexample to universal optimality, not a current Direct regression:
r2SCAN-3c warm E+F medians increase about 0.8–1.0% when only the SCF tile grows
from 256 to 1024, while B3LYP improves on the same devices. Larger local AO unions
and tile-proportional scratch can outweigh launch amortization. This is why the
decision is a default preference with bounded fallbacks, not an all-workload
performance claim or permission to bypass resource admission.

Native host probes cover preference, exact boundary admission, explicit overrides
and conservative resource/workload fallbacks without a CUDA device. Existing
fitted capability-policy and public SCF/force transport tests remain independent.
No new GPU endpoint performance or numerical qualification is claimed here;
fresh exact-head CI and representative real-device endpoints remain the evidence
needed for new-source/device performance claims.

## Revisit when

Use stable complete-endpoint regressions and semantic work/resource observations
to refine workload-based admission. Do not reintroduce GPU-family allowlists or
require an exact local profile merely because universal optimality is unproven.
