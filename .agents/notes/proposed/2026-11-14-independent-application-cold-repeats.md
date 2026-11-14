# Proposal: independent application-cold process repeats

Status: proposed for figure adoption; tooling implemented and device acquisition complete; user review pending
Date: 2026-11-14

## Problem

The paper HF/DFT figures use one application-cold call and three same-owner
warm calls per population. A single cold observation cannot quantify variation
across independent process launches. A nonmonotonic cold point alone does not
establish random noise, compiler overhead or a solver defect.

## Decision

Keep the established complete E+F timing boundary: constructor, preparation,
SCF, analytic forces and host export. Imports, CUDA-context startup and an
explicitly separate complete cache-prewarming pass remain outside timing.
Persistent compiler caches are retained, never cleared for a convenient result.

For every engine/method/mode/size, run three fresh processes. Each owns one cold
call and three warm calls from its own immutable cold density. Plot the median
and min/max of the three cold values. For warm, plot the median and min/max of
the three within-process medians; retain all nine original warm values. These
are descriptive ranges, not confidence intervals or nine independent processes.

Balance first-engine order in each round and reverse the middle traversal.
Do not replace, smooth, omit or retry scientific samples to remove a peak.
Failed/incomplete populations prevent a complete aggregate. Preserve their raw
receipts separately. Repeat counts, work, source, actual native image, physical
device and scientific inputs are checked before aggregation.

## Finite supervision and authenticated build reuse

The process timeout covers the whole population, including its cold and three
warm calls, imports, context startup and metadata. A timeout suitable for one
call can truncate a correct population. The historical RTX 5090 default
GPU4PySCF DF omegaB97M-V96 population takes about 754 seconds in total; the
initial 600-second supervisor setting was therefore too short. The corrected
RTX 5090 attempt uses 1800 seconds per process within a finite parent/allocation
deadline. H100's historical slowest four-call population is about 77 seconds;
its 600-second bound remains sufficient for this campaign. These bounds were
checked against the original complete populations in
`readme-rtx5090-composed-20261111/samples.json.xz` and
`readme-h100-20261112/samples.json.xz`, under `benchmarks/results/` at
`efeaa3c7107a759e95c088d366e990d66c551c27`.

The initial RTX attempt was cancelled during prewarm, before any formal
population. Its original observations remain separate. The replacement
performs a fresh full prewarm and all three formal rounds; no earlier timing
is substituted. This is a diagnosed operator correction of a supervisor budget,
not an automatic scientific retry or an exclusion chosen from formal timings.

A completed source-matched build may be reused without recompiling. Preserve
the original build receipt and loaded-image/source hashes, rerun the AOT audit,
and separately bind the actual execution GPU and Slurm allocation. Compiling
on a GPU does not by itself authenticate a later allocation, and an explicit
execution binding must never allow a different native image or source.

## Allocation and ownership

Slurm/LSF retain their actual scheduler identities. Qz Jobs have UUID-based
worker hostnames, unlike Notebook names. Qz metadata admission records the
declared job/project and worker host without fabricating Slurm fields. An
authoritative platform receipt is still required: hostname/environment matching
alone is not independent allocation authentication or scientific acceptance.

Production source stays frozen across the paired campaign. Platform adapter
hashes and actual environments remain explicit; a later recorder fix must not
silently rewrite an active process's dependencies. Do not infer a hardware-only
speedup from two different machines, drivers or package environments.

## Validation and acceptance

Host tests exercise retention of slow outliers, rejection of missing/duplicate
repeats, mixed sources/devices, invalid full-force exports and invalid scheduler
metadata. These host results do not qualify CUDA execution or scientific
accuracy. Full-vector errors against the unchanged independent dense references
are reported separately and never used to select favorable timing samples.

The acquisition evidence and figures must be reviewed before this proposal is
treated as accepted publication behavior. No historical cohort is rewritten.

## Completed evidence and retained negative results

The paired campaign measures production `efeaa3c7107a759e95c088d366e990d66c551c27`,
actual native source identity
`20d337f7ffdb1c75eb89a3ba6b0c307518e21388a4d4fda7df6b54248b4337e7`.
Both H100 and RTX 5090 complete 384 formal processes and 1536 full E+F calls.
All cold-median curves are monotonic in this data, with every original sample
and min/max range retained. This is not a guarantee of monotonic future timings.
The H100 middle-round Direct HF48-to96 decrease remains in its individual data.

H100 native-faster median counts are 63/64 cold and 62/64 warm. RTX 5090 has
60/64 cold and 64/64 warm. Its four cold exceptions are Direct HF96, DF HF96,
Direct PBE096 and DF r2SCAN-3c96; the old RTX cold-all-faster statement does not
carry over. Three exceptions are slower in all three pairs; PBE0 is near parity.
Old/new simple SCF/Fock/preliminary counts match, while source and physical
machines differ. The current CPU affinity is GPU-local. No same-device old/new
source A/B or causal runtime attribution is claimed.

Each GPU retains 348 qualified and 36 failed process results against the
unchanged independent dense Direct references. The failed keys match the
historical DF-HF approximation and three-atom M06-2X grid/force cases. Timing
figures include them, without changing scientific gates or claiming overall
numerical acceptance.

The replay bundle under `benchmarks/results/paper-cold3-20261114/` retains all
full-precision scientific sample fields and original archive/member identities
as declared numeric projections; it does not reconstruct original process JSON.
Acquisition filesystem paths and routine logs remain outside Git. Source patches
reconstruct the actual CRLF/LF recorder bytes. LF attributes preserve the hashed
portable manifests/scripts, while the literal mixed-EOL patches remain unmodified.
The aggregate storage allowance grows by this exact 2,053,507-byte selected
cohort, including its manifest. The hard 1 MiB file and 2 MiB incoming limits
remain unchanged; no archive or classification waiver is added.
