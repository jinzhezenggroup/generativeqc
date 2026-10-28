# Decision: default XC sharing with an ordinary-General parameter ABI

Status: implemented
Date: 2026-10-28

## Problem

The [paired-batch repair](2026-10-28-xc-shared-point-abi.md) removed the TH/BKL
RKS regressions, but ordinary UKS APBE remained about 1.5% slower than literal
specialization. Broad [structural discovery](../architecture/2026-10-28-xc-structural-parameter-discovery.md)
therefore remained opt-in. The same storage annotation cannot be assumed to
improve all four execution ABIs: an earlier all-entry grid-constant experiment
made batched APBE worse.

## Decision

Use `const __grid_constant__ ParameterRow` only for the ordinary General entry.
CUDA gives this immutable kernel parameter one grid-wide address and avoids the
per-thread-copy requirement when its address is taken. The annotation does not
change the mathematical coefficients, derivative graph or physical traversal.
Restricted ordinary, restricted batched and general batched entries keep their
previous storage contracts. The restricted batched adapter continues to expose
the equal physical spin channels proved by the host.

The common emitter accepts a kernel name so both storage declarations share one
traversal implementation. Only the ordinary General launcher selects the new
template; each group still instantiates four kernels. There is no functional-name,
GPU-model or point-count exception.

Structural discovery becomes the default policy in the compiler package,
standalone generator and CMake. This is a default inside enabled automatic XC
AOT. The AOT master switch, public method admission, precision and derivative
capabilities are unchanged. `qualified` retains the legacy prepared-product
selection; sharing OFF retains literal specialization. Incompatible mathematical
bodies or execution ABIs, oversized launch data and singleton selections also
retain literal specialization.

## Rejected alternatives

- Deduplicating opposite coefficient columns did not repair APBE and introduced
  a GGA_C_APBE RKS regression: 1.0160 versus literal and 1.0274 versus the previous
  shared worker in the pilot. Keep identical-column deduplication only.
- Combining sign deduplication with ordinary grid-constant storage passed that
  pilot, but adds an unnecessary mathematical-emission change and a known bad
  isolated transformation. Use the smaller storage-only change.
- Applying grid-constant storage to every entry repeats the earlier batched APBE
  regression. The boundary is execution ABI, not membership in a family.
- A functional whitelist would hide the execution issue and require ongoing
  catalog maintenance. Discovery continues to compare both complete spin bodies,
  physical domains, coefficient order and output layouts.

## Evidence and acceptance

The [retained campaign](../../../../benchmarks/results/xc-structural-default-20261028/README.md)
contains raw observations, failed pilot alternatives, source reconstruction,
compiler-cache receipts, independent Libxc/PySCF checks and full endpoint work
ledgers. It qualifies NVCC 12.9.86 / GCC 11.4 / SM120 on a Slurm PRO6000 allocation.
Cache-aware compile times are not cache-miss speedup claims. Static resource
observations do not establish dynamic stalls or measured local-copy removal;
hardware counter access was unavailable in the earlier campaign.

The pilot has 80 cases, five arms and three randomized rounds. Ordinary-General
storage stays within 1% of literal and the previous shared worker in every pilot
case. Its APBE UKS ordinary 16K ratio is 1.00721, versus 1.01056 before. Final
qualification rebuilds both CPU and CUDA identities from the same frozen producer
and the native library from current integrated sources. It covers all 73 shared
registrations, both spins, ordinary/tile-127 traversal and 16K/262K points; the
per-case point boundary is 1.01. Complete energy endpoints require independent
reference gates, equal semantic work and at most 1.02 per cold/warm/moved cohort,
with at least five paired observations and a larger four-water case.

Direct-event spans initially leave four cases over 1.01. A predeclared five-round
confirmation plus fixed controls leaves APBE at 1.011221. These observations are
retained; they do not establish a pure GPU-kernel regression because their timed
interval includes host submission after recording the start event. The short
warmup is also not duration-normalized. A one-node Graph with at least 100 ms of
warmup isolates execution through a common submission path: all 32 pilot cases
pass, with APBE at 0.999643. Final qualification uses this protocol for every
ordinary case, while keeping the original batched and complete-endpoint evidence.
Do not turn these measurements into a claim of zero host-launch overhead. The
endpoint gate includes the production submission paths. The tested 64/256-thread
alternatives do not justify changing the retained 128-thread schedule.

The existing MVSB UKS H4 convergence limitation remains an explicit exclusion from
complete-endpoint qualification, with its original literal/shared failures retained
in the preceding campaign. Its point and fixed-density numerical checks remain
required. Sharing does not change SCF convergence policy or certify every method,
geometry or architecture. Four-water UKS also fails the same 400-iteration limit
in both literal and shared probes and is excluded from positive qualification;
its failed states and work histories remain retained. Four-water RKS passes the
independent cold/warm/moved oracles.

The final 34 converged cohorts retain 1,700 complete calls, equal semantic work,
maximum paired energy difference 2.27e-13 Eh and maximum phase ratio 1.008677.
The declared GPU point protocols peak at 1.008197. All previous and current timing
observations remain available: exact duration-table interning reduces storage
without rounding, and canonical population hashes authenticate reconstruction.
Both preceding reducers remain byte-identical after lossless compaction, keeping
the evidence tree within the existing 80 MiB cap without retiring more evidence.

## Revisit when

A new compiler, device, discovered contract or workload shows a reproducible local
regression. Retain its raw paired observations and use the explicit policy/literal
fallback while investigating the relevant ABI. Preserve the complete endpoint,
independent oracle and source identity gates when changing storage again.
