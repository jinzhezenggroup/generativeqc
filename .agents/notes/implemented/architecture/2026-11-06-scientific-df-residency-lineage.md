# Decision: explicit DF host-transform generations and named replay boundaries

Status: implemented
Date: 2026-11-06

## Problem

Issue #1629 already had a recorder and strict receipt consumer. Aggregate copy
budgets cannot distinguish an allowed publication from an equally sized new
scientific host intermediate. Adjacent D2H/H2D bytes also cannot establish
scientific data dependence. The earlier n4 compatibility captures reached a
host MP2 consumer but had unknown subsequent scientific H2D dependencies.

## Decision

Extend the existing private source observer, not the recorder or public ABI.
The DF setup owner tags its actual GPU eigenvalue download, existing completion
fence, checked CPU inverse-square-root transformation and H2D scales consumed by
the existing device metric-scaling kernel. A `host_transform` marker uses the
unchanged 14-word source layout. DF catalogue extensions stay in DF-owned headers
instead of recompiling all common-header CUDA consumers.

Only explicit unique source generations, successful API/activity joins, correct
ordering, owner/execution/role/payload and compatible sizes qualify this narrow
scientific chain. No pointer, equal-byte or timestamp inference is accepted.
Observation adds no CUDA transfer/fence and changes no scientific cutoff/math.

Version-3 work policies add named boundary contracts to existing aggregate
owner/role maxima. Region/owner/role/site/payload/kind/direction must match;
unreviewed work has zero allowance even if aggregate bytes remain identical.
A scoped zero-*unexpected*-round-trip assertion covers the declared HF replay
owner's observed explicit boundaries only. Failed/lost/incomplete observations
cannot grant it. Global residency, scientific dependency and implicit/graph-wait
coverage remain `INCOMPLETE`; the global zero-round-trip assertion stays unset.

## Evidence and rejected alternatives

[n4 qualification](../../../../benchmarks/results/residency-qualified-n4-20261106/README.md)
retains Slurm job 1511's raw journals and independently pinned policies.
H2/LiH/water public CUDA DF-MP2 captures have actual metric host-transform links
and pass ordinary-CUDA/native-CPU/independent-fixture energies. Prepared RHF/UHF
histories pass the original 1e-10/1e-9 energy/force gates, ordinary CUDA controls
and the new named-boundary ratchets. CPU CI recomputes both from raw records and
rejects same-count scientific-payload substitution and broken lineage.

Budgets were declared from historical job 1506, not learned from the candidate.
Solver controls 1e-12 energy / 1e-11 density are pinned and identical for all
three compared histories. Tighter convergence fixes this qualification's prior
moved-RHF gate discrepancy without relaxing acceptance or claiming a universal
fix to default convergence. The original failed evidence remains unchanged.

Job 1510's additional F-HeH full-SCF capture failed with a CUDA launch error;
it is not qualified by the supported three-case campaign. Its log is retained.
This differs from historical F-HeH generated-provider compatibility, which
passed independent energy/amplitude fixtures. A failure-path improvement now
retains raw journals and original errors even when a sticky CUDA error also
breaks the observer fence. No unsupported case is called a passing capture.

## Invariants and revisit conditions

Preserve Slurm visibility, finite jobs, source/build/device provenance,
independent numerical gates, legal preparation/compatibility/publication roles
and all unqualified coverage flags. Do not infer global residency or graph work
from this scoped assertion. Expand owner/graph/implicit-wait contracts only with
their own scientific lineage, clean endpoints and independent actual evidence.

## Merge-base storage follow-up

The first finished head passed local retention against its original base, but
the authenticated CI test merge included newer base evidence: 85,973,776 bytes,
leaving only 9,456 bytes below the unchanged 82 MiB cap. Do not assume the event
base is the actual tested merge base; `check_evidence_change.sh` deliberately
checks the latter. Raising the cap or deleting unrelated scientific evidence
would conceal that integration constraint.

Complete qualified benchmark snapshots remain in existing Git commit
`e1922f36c0e449a8dbd4509a096f3d469c35087b`, with per-file hashes/recovery in
`manifests/evidence_migrations/mp2-residency-qualified-snapshot-20261106.json`.
Current benchmark directories retain small recovery indexes. Every live raw
scientific, calibration and hot-loop negative replay input remains in permanent
`tests/reference_data/cupti/`; no test or independent gate is removed. Full local
source/build/profiler artifacts also remain ignored. This is storage compaction,
not changed measurements, a new publication/Release, or a residency promotion.
