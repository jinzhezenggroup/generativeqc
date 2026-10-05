# Decision: calibrate CUDA static diagnosis with retained NCU execution evidence

Status: proposed diagnostic layer; no production selector change.

Agent: ChatGPT
Model: GPT-5.6 Sol

## Motivation

The compiler already has useful GPU-free PTXAS/resource screening and a separately
calibrated homogeneous-kernel timing estimator. Recent source-matched Nsight
Compute captures show that the resource layer is often accurate while the
mechanism ranking needs a second execution-efficiency layer.

The clearest example is the generic stationary Direct force producer tracked by
#1965/#1892: 255 registers/thread at 256 threads/block predicts a one-block
register limit and 16.67% theoretical occupancy, and NCU measures 16.67%
achieved occupancy. That validates the resource bound. It does not establish
that register count alone is the root cause: the same capture has about 17.9/32
executed threads per warp instruction, very low eligible/issue-active warp
fractions, a roughly 44% Barrier share of displayed warp latency, substantial
dynamic local-memory requests and low DRAM busy.

Other retained captures distinguish different mechanisms:

- #1502 and #1763: generated CC scalar reductions execute about one lane per warp,
  have Barrier=0 and zero local requests. Their actionable defect is underexposed
  reduction parallelism, not a register/spill prescription.
- #1972: a shared exact J/K value producer has low lane utilization, strong long
  scoreboard latency and large local request counts but Barrier=0. It is distinct
  from the generic derivative barrier-tail mechanism.
- #1855: WB97M-V full/LR generic derivative producers reproduce the low-occupancy,
  partial-lane, barrier-tail and local-state pattern, strengthening the
  cross-functional classification without proving one particular optimization.
- #1879/#1882: GFN2 currently exposes only whole device-launchable SCC-graph NCU
  aggregates. Those counters must not be assigned to density or Broyden nodes.
- #1876/#1884: BLAS eligibility is not a speed certificate; real local/indexed
  materialization can lose even when dense synthetic algebra favors a vendor
  primitive.

## Decision

Add a diagnostic-only NCU execution evidence schema and mechanism classifier.
Attach it to `tools/analyze_cuda_cost.py` with `--ncu-evidence`.

The static/PTXAS screening priority remains byte-for-byte independent of NCU
evidence. NCU cannot promote a candidate, rewrite scientific precision, or turn
profiler replay duration into endpoint timing.

The report separately compares the static/PTXAS per-SM occupancy upper bound
with the source-matched NCU theoretical occupancy. This checks whether resource
prediction itself is calibrated before interpreting achieved execution.

## Initial coarse regimes

These are deliberately broad advisory thresholds, not fitted speedup weights:

- issue-starved: issue-active < 15% or eligible warps/scheduler < 0.25;
- lane-underutilized: executed lane fraction < 75%;
- serial/underexposed candidate: lane fraction <= 12.5%, Barrier <= 5%, zero
  measured local requests;
- barrier-tail: Barrier >= 25% of displayed warp cycles/instruction;
- long-scoreboard latency: Long Scoreboard >= 25% of displayed warp
  cycles/instruction;
- low-occupancy resource bound confirmed: theoretical occupancy <= 25% and
  achieved/theoretical >= 90%;
- DRAM not saturated: DRAM busy < 50%.

The combined classifications are intentionally mechanism-specific. In
particular, dynamic local-memory request counts are never relabelled as spill
bytes.

## Calibration examples

The first regression set covers three intentionally different retained regimes:

1. generic PBE0 force: resource-bound occupancy confirmed + issue starvation +
   partial lanes + barrier tail + local-state traffic;
2. CC scalar reduction: one-lane underexposure with no barrier/local traffic;
3. exact J/K value producer: partial lanes + long scoreboard + local-state
   traffic without barrier tail.

A future threshold change should add held-out evidence first. Do not tune a
threshold merely to make a desired optimization rank higher.

## Follow-up

After this diagnostic layer survives CI and additional retained NCU samples,
consider whether specific *structural* facts (for example compiler-proven serial
reductions or heterogeneous task padding) should affect pre-benchmark shortlist
priority. Keep measured NCU counters out of scientific identities and default
dispatch policies.
