# Decision: bound larger Hessian exploration by actual device usage and endpoint time

Status: implemented
Date: 2026-10-16

## Problem

The earlier H2/water direction-reuse benchmark establishes only tiny, coarse-grid
latency. The requested larger-system followup must not exhaust GPU memory or
spend unbounded time exploring a method whose practical scaling is unknown.
Hessian-owner arena bytes alone are not a full-device memory measurement.

## Decision

Keep larger benchmark molecules opt-in and preserve the existing tiny default
set. Extend the comparator with a conservative sampled total-device memory
watchdog and failed-attempt journaling. The resource callback shares the finite
endpoint deadline and existing private-process-group cleanup; sampling must not
leave a timed-out numerical child running. Memory rejection, timeout and strict
SCF convergence failures remain failed samples, never filtered speed wins.

The October 16 campaign uses 180 seconds per engine and a 24 GiB watchdog, finite
single-GPU Slurm steps, unchanged direct RKS/spherical STO-3G science and identical
primary/NLC/CPHF quadrature. The node is shared, and the 50 ms device-memory
samples include comparator contexts. Neither exact allocation enforcement,
isolated process peak, chemical grid convergence nor statistical performance
promotion is claimed. Native production binaries remain unchanged and Python/JIT
free; the Python driver belongs only to independent benchmark orchestration.

## Evidence

See `benchmarks/results/wb97mv-hessian-bounded-scaling-20261016/`. All completed
paired matrices pass the unchanged 3e-6 absolute gate; the maximum discrepancy is
3.6514e-8 hartree/bohr^2. The two primed formaldehyde observations have medians
33.4996 s native versus 24.4927 s reference. A single six-atom water-dimer endpoint
takes 93.1989 s versus 11.7443 s. Native methanol completes in 102.6281 s, but its
reference SCF fails; native ethane times out and its reference SCF also fails.
These failed cases retain their raw outputs, not fabricated matrices or ratios.

Sampled native device usage is approximately 16.75--16.79 GiB versus reference
2.38--2.68 GiB. H2 already has approximately 16.75 GiB usage, whereas the dimer's
Hessian-owner arena counter is only 22,386,768 bytes. Do not substitute that
counter for device-wide footprint or infer the footprint's cause without a
separate profile. The ethane timeout does not expose completed counters, so it
cannot establish whether SCF or Hessian evaluation consumed the deadline.

## Rejected alternatives and invariants

- Do not extrapolate the water speed win or call these 2--16-AO workloads genuine
  large-system scalability evidence.
- Do not loosen SCF/matrix gates, replace reference kernels, discard failed cases
  or change response grids to manufacture a comparison.
- Do not raise conservative admission limits merely to make a larger molecule
  appear supported. At this source, STO-3G ethanol's 101250 ordered exact groups
  already exceed the default 100000 cap; that is a static bound, not a timed run.
- Do not treat sampled usage as an exact peak; short-lived spikes can be missed.

## Revisit when

Profile the complete native dimer endpoint before selecting another optimization.
Revisit larger molecules after reducing actual endpoint work, qualifying useful
quadrature/convergence behavior and retaining bounded admission/fallbacks. Use
exclusive timing allocations and more samples before general performance claims.
