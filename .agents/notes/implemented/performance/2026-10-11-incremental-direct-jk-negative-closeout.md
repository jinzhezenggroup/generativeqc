# Decision: keep Incremental Direct-J/K opt-in; negative closeout for default promotion

Status: implemented
Date: 2026-10-11

## Problem

Issue #1598 DoD item 1 required a durable decision for the exact incremental
Direct-J/K route merged by #990/#1579 (native CUDA ΔD compaction, fixed-topology
work counters): either a guarded default promotion under the promotion contract,
or retained negative evidence keeping the route opt-in. The prior PBE0-domain
prototype evidence (#1803 lineage) was cautionary but did not settle the HF
domain on current master, and #1579 explicitly claimed no GPU endpoint win.

## Decision

`incremental_direct_jk` remains **opt-in** (`GENERATIVEQC_INCREMENTAL_DIRECT_JK`
benchmark selector and the public option), and the inventory entry
`scf.incremental-direct-jk` is reclassified from `needs-qualification` to
`negative-evidence`. No default/auto routing is introduced, no scientific gate
changes, and no runtime behavior changes on master.

The route is not retired: the shared policy still serves the narrower KS/PBE0
owner issue #2053, and a future guarded promotion can reuse the retained
matrices below.

## Evidence

The historical capsule in
`benchmarks/results/issue1598-incremental-direct-jk-20261011/` retains successful
RHF nested-water endpoints on one RTX 4090 and unsuccessful final retry
outcomes. Its README records the exact measured identities, timings and limits.

The measured harness checkouts are `96570fd06243de418e2eec4efb570d5510506d03`
(dirty, old) and `d83aa86585a2f72ed852872c533723648b2799c3` (clean, new).
Previously reported `637466c…`/`143df078c…` master lineage is not a substitute
for those measured identities. Native probe source-content fingerprints and
library hashes are retained separately; the old dirty-source patch is absent.
The capsule audits historical receipts, not exact source reconstruction.

The primary RHF population is 96/192/384/768 AOs; the second is
192/384/768 AOs. Both 384-AO rounds show cold and warm regressions in both
campaigns (round 1: old +42% cold/+94% warm, new +41% cold/+103% warm).
The old 96-AO cold endpoints are 3.1% and 5.8% faster, while their warm
endpoints are about 67% slower. Therefore the data support no warm win or
established cold-and-warm promotion domain, not a blanket absence of any
endpoint improvement.

Both 192-AO incremental points have final exit_code=1, attempt=2,
retried=true in each campaign. Old 768-AO incremental round 1 succeeds
(30 cold iterations); round 2 has the same failed-retry disposition. Both new
768-AO incremental points also have failed-retry dispositions. Baseline points
succeed. Failure exception logs and failed RHF comparator records were not
retained. Earlier reports attributed the failures to nonconvergence, but the
cause and individual first-attempt results cannot be re-established from this
capsule, so this is evidence of unsuccessful retried points only.

Every retained successful RHF warm sample passes the independent GPU4PySCF
reference and all available same-round baseline/incremental comparisons at
|ΔE| ≤ 1e-8 Eh and |ΔF| ≤ 1e-7 Eh/Bohr. Cold native result arrays are not
retained, so offline numerical rechecking is limited to warm samples. Cold
endpoint timings and convergence records are retained. The reported protocol
was fresh-process A-B-B-A baseline/incremental points, one cold endpoint and
three fixed-density warm replays, with one retry for a failed point.

KS/PBE0 control records were attempted but are not accepted independent
numerical evidence; the recorded campaign report attributed the oracle
mismatch to intervening DFT force changes (#2246). This RHF negative decision
does not depend on that explanation. The UHF-cation and moved-geometry
populations were not completed. Refusing default promotion does not require
exhausting every population; any future promotion still needs them. The
narrower KS/PBE0 owner issue #2053 retains its own qualification burden.

The capsule contains compressed comparator records, duplicate selected/successful
attempt records, final retry outcomes, the old device/library/source receipts,
and a checksum inventory. It does not retain per-point exception logs, progress
journals, a resource time-series, reducer summaries, or a dirty-source patch.
Per-record after-measurement device snapshots do not certify isolation throughout
the campaign. Incremental diagnostics show activation, but bounded-streaming
work counters are invalid and cannot certify a resource guard.

## Why no promotion

The RHF evidence shows warm regressions, large 384-AO cold regressions and
unsuccessful retried incremental points at 192/768 AOs. The small 96-AO cold
gain does not establish a domain with wins across cold and warm endpoints.
The remaining provider-owned bounded-streaming work accounting (`quartet_work_counters_valid` excludes bounded-streaming runs) also
means no fail-closed workload guard can currently be certified across the
bounded resource domain the promotion contract requires. Keeping the route
opt-in avoids manufacturing a guard that only covers the unbounded
fixed-topology domain.

## Rejected alternatives

- Guarded promotion restricted to the unbounded fixed-topology domain: no
  retained population establishes wins across both cold and warm endpoints.
- Retiring the option: the shared policy remains in active use by the narrower
  KS/PBE0 owner issue #2053; retirement would churn that work without endpoint
  benefit.
- Relabelling the older PBE0/KS rejected evidence as this decision: the RHF
  domain has its own historical measurements and explicitly scoped limitations.

## Invariants

- `incremental_direct_jk` stays off by default; exact-only, strict-FP64-only
  admission and the strict post-SCF full physical rebuild remain unchanged.
- The final physical state and independent numerical gates are never weakened
  to make incremental timing favorable.
- The inventory entry remains the authoritative classification source.

## Consequences

No runtime change. The retained benchmark matrix and runner
(`benchmarks/run_issue1598_incremental_qualification.sh`,
`tools/reduce_issue1598_incremental_qualification.py`) support future
qualification; the capsule verifier audits the retained historical subset without claiming missing inputs exist.

## Revisit when

Reopen promotion only with: (a) provider-owned bounded-streaming
admitted/rejected work accounting; (b) a candidate guard domain with matched
complete-endpoint wins across cold, warm unchanged-geometry and moved-geometry
RHF/UHF populations at 96-768 AOs with unchanged numerical gates, repeated
interleaved timing, exact source/library/device identity, and compile/artifact
reuse cost charged to admission; or (c) a materially different late-SCF work
distribution (e.g., after a J-side density-screening change) that reopens the
economics.

## References

- #1598 (promotion tracker), #990 (implementation), #1579 (merged CUDA route)
- `benchmarks/results/issue1598-incremental-direct-jk-20261011/`
- `manifests/maintenance/default_promotion_inventory.json` entry
  `scf.incremental-direct-jk`
- Supersedes the PBE0-domain cautionary note
  `.agents/notes/rejected/2026-10-09-pbe0-incremental-endpoint-promotion.md`
  for the HF domain only.
