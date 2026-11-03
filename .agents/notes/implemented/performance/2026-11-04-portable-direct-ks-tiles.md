# Decision: portable preferred Direct SCF and force tiles

Status: implemented
Date: 2026-11-04

## Problem

PR #2948 initially replaced the A100-name profitability gate with an SM80 gate,
then required exact local profiles before enlarging any native Direct tile.
Neither choice follows from a portable cost model: hardware identity does not
establish the best batch size, but missing exact-workload evidence does not
establish that 256 is the better default either. The user approved preferring
1024 across GPU models with bounded resource fallbacks. During integration,
master merged #2949, which already applies the broader portable CUDA semilocal
default. Preserve that implementation rather than restoring older workload gates.

## Decision

Keep #2949's automatic native CUDA semilocal SCF and force preference of 1024,
including Direct/global-hybrid and unrestricted owners, without a device-name,
generation, FP64-throughput, minimum-size or local-profile gate. Remove the
obsolete SM80 Direct predicate and its unreachable caller: full-range Direct
owners already enter the ordinary semilocal branch before the fitted nonlocal
capability branch. Do not add an alternative helper or retain a compatibility
alias for a predicate that no longer owns any execution choice.

Explicit SCF/force bounds and their inheritance remain authoritative. Automatic
SCF preparation and force-controller allocation failures retain the existing
single drained retry at 256; non-allocation failures and a second failure still
propagate. CPU and numeric resource promises retain 256; nonlocal CUDA keeps its
independent fitted capability, size, free-memory and stationary-admission policy.
Local DFT profiles
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

Native host probes cover unknown, older, SM80, Hopper and consumer-GPU capability
values, small and large Direct shapes, explicit overrides and promised-resource
fallbacks without a CUDA device. Existing nonlocal fitted capability-policy and
public SCF/force transport tests remain independent. The separately retained
[default-path qualification](../../../../benchmarks/results/default-cuda-xc-1024-20261103/README.md)
belongs to #2949's measured source, not a new GPU run of this integration. Fresh
exact-head CI and representative real-device endpoints remain the evidence
needed for new-source/device performance claims.

## Revisit when

Use stable complete-endpoint regressions and semantic work/resource observations
to refine workload-based admission. Do not reintroduce GPU-family allowlists or
require an exact local profile merely because universal optimality is unproven.
