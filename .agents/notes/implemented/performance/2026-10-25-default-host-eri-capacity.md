# Decision: Admit host ERI backing automatically within available job RAM

Status: implemented
Date: 2026-10-25

## Decision

Following the requested default behavior, admitted cold strict LR sources use
host backing for buckets omitted from GPU residency. GPU selection remains
first. Unset host ceiling uses at most 32 GiB and one quarter of currently
available RAM, constrained by Linux cgroup memory limits and current usage at
every ancestor (including enclosing Slurm jobs). Missing or malformed capacity
information disables automatic backing. Explicit zero disables it; an explicit
byte ceiling retains the existing caller-controlled contract.

The incumbent complete numeric budget and device staging guards still apply.
Full GPU caches allocate no host values. Warm/external seed eligibility, strict
precision, source lifetime, GPU priority and Direct refusal behavior do not
change. This policy estimates capacity, not the profitability of each bucket.
The available-memory observation is inherently racy; retaining three quarters
of the observed headroom reduces, but cannot eliminate, competing allocations.

## Evidence and limits

The source-bound initial screen and independent energy/force gates are retained
in `benchmarks/results/host-lr-eri-20261025/`. Those timings predate this default
and current-master integration and must not be relabeled. The 108-atom omitted
LR recurrences were much more expensive than host replay; the 96-atom formal
cohort refused host admission. Executable capacity tests cover machine, job and
ancestor limits, exhausted/malformed observations and cgroup v1/v2. GPU and
endpoint checks for the default are recorded separately with their new source.

## Alternatives and follow-ups

Do not infer available job RAM from machine totals or an iteration limit.
Pinning the entire host cache would remove reclaimable headroom. SSD temporary
spill is a separate follow-up and must retain local space bounds, cleanup and
complete-endpoint evidence. Full-range KS consumers are qualified separately.
