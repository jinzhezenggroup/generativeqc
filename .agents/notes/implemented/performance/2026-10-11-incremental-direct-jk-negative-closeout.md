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

Two independent campaigns on one NVIDIA GeForce RTX 4090
(UUID `GPU-0639f74d-c42b-3c2b-4a57-3795decc0888`, driver 570.124.06, CUDA 12.9
toolchain, gcc 13.3, AOT portable_cuda sm_89, `-Xptxas
--disable-optimizer-constants` required by nvlink, sccache 0.16.0) in finite
inspire allocation `issue-1598-dijk-20261011`. Every point is a fresh process
running the audited `compare_gpu4pyscf_batch` comparator with an independent
GPU4PySCF reference engine; per size the order is A-B-B-A (baseline /
incremental / incremental / baseline), each with one cold execution plus three
fixed-density warm energy+force replays, and one retry per failed point. All
rebuild/finalization/audit cost is inside the measured endpoints.

Primary campaign, master `637466c624c4c4c4dd4eb7c511a2c88675744325`, library
`2998014a18547b813149521c51193cd8913602b5d7e4551e326a4d5c91053a77`:

| RHF AOs | baseline cold / warm | incremental cold / warm | result |
| --- | --- | --- | --- |
| 96 | 2.34 / 0.23 s | 2.22 / 0.38 s | both rounds pass; no delta opportunity (1-iteration replays) |
| 192 | converges | fails | incremental fails all 4 attempts ("SCF did not converge"); baseline converges |
| 384 | 71.0 / 7.09 s | 100.5 / 13.77 s | passes; warm +94% slower, cold +42% slower |
| 768 | converges | round 1 converges (30 it), round 2 fails 2/2 | run-to-run convergence instability; warm replay 52 s vs baseline 1-iteration solve |

Post-#2236/#2241 re-verification campaign, rebased master `143df078c…`
(base `98263e06c…`), library `30734a0fc741da60389fb817b7429acd4a5164c840728308822fe215eae90a63`:

| RHF AOs | baseline cold / warm | incremental cold / warm | result |
| --- | --- | --- | --- |
| 192 | converges | fails | incremental fails all 4 attempts; baseline converges |
| 384 | 71.1 / 6.83 s | 100.2 / 13.86 s | passes; warm +103% slower, cold +41% slower |
| 768 | converges | fails 4/4 attempts | reproducible convergence failure |

The KS/PBE0 control (96 atoms, strict FP64, off/on/on/off) was attempted on the
new master: cold off-96 endpoints measured (279.0 s, 17 iterations, incremental
inactive) with the same protocol, but independent acceptance against the
retained composed-baseline oracle could not be re-established because that
oracle predates the merged DFT force changes (#2246), so its gates no longer
bound the current master. The KS/PBE0 domain remains covered by its own
retained rejected evidence (`.agents/notes/rejected/2026-10-09-pbe0-incremental-endpoint-promotion.md`)
and the narrower owner issue #2053; the HF decision above does not depend on
it. The UHF-cation and moved-geometry populations of the primary matrix were
also not completed: the campaign was superseded mid-run when #2236/#2241
merged, and the re-verification matrix was scoped to the decisive RHF sizes
plus the KS control. That is acceptable for a negative decision — refusal does
not require exhausting every population — and the harness retains those cases
for any future revisit.

All paired numerical gates that could be compared pass (maximum |ΔE| ≤ 1e-8 Eh,
|ΔF| ≤ 1e-7 Eh/Bohr, plus the comparator's independent reference gates). The
incremental diagnostics confirm the mechanism activates and removes real
fixed-topology quartet work where counters are valid. No complete-endpoint win
exists at any population of 192 AOs or larger on either revision: the 384-AO
warm endpoints are 1.94x/2.03x slower and the cold endpoints 1.42x/1.41x
slower; 192-AO and 768-AO incremental cold runs reproducibly fail to converge
while their baselines converge. Only the 96-AO no-opportunity points slightly
favor the incremental route on cold samples (2.29/2.22 s versus 2.36/2.36 s),
which does not generalize. The in-repo receipts retain the failure outcomes
from both campaigns, the matched 384-AO round-1 pair on the new master, and
the nvidia-smi resource ledger, device/library/source identities, per-run
progress journals, retry outcomes and the reducer summaries; the compact
publication deliberately omits the remaining successful records, which are
cited from this note.

## Why no promotion

Task-count reduction did not translate into complete-endpoint wins at any
population of 192 AOs or larger on either measured revision, consistent with
the PBE0-domain pattern (warm parity after audit reuse; moved-geometry
regression), and the route reproduces convergence failures at 192 and 768 AOs
on two successive master revisions. The 96-AO cold no-opportunity samples
slightly favor the incremental route and are reported as such rather than
generalized. The remaining provider-owned bounded-streaming work
accounting (`quartet_work_counters_valid` excludes bounded-streaming runs) also
means no fail-closed workload guard can currently be certified across the
bounded resource domain the promotion contract requires. Keeping the route
opt-in avoids manufacturing a guard that only covers the unbounded
fixed-topology domain.

## Rejected alternatives

- Guarded promotion restricted to the unbounded fixed-topology domain: no
  retained endpoint population shows a reproducible win there, so a guard would
  gate on a domain with measured negative or neutral economics.
- Retiring the option: the shared policy remains in active use by the narrower
  KS/PBE0 owner issue #2053; retirement would churn that work without endpoint
  benefit.
- Relabelling the older PBE0/KS rejected evidence as this decision: the HF
  domain was measured independently on current master as required by #1598.

## Invariants

- `incremental_direct_jk` stays off by default; exact-only, strict-FP64-only
  admission and the strict post-SCF full physical rebuild remain unchanged.
- The final physical state and independent numerical gates are never weakened
  to make incremental timing favorable.
- The inventory entry remains the authoritative classification source.

## Consequences

No runtime change. The retained benchmark matrix and runner
(`benchmarks/run_issue1598_incremental_qualification.sh`,
`tools/reduce_issue1598_incremental_qualification.py`) make the closeout
reproducible and auditable.

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
