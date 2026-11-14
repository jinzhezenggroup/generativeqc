# Decision: Start the complete public ECP force file before cheaper files

Status: implemented
Date: 2026-11-11

## Problem and evidence

PR #3062 CI run `42965903275`, job `130343104062`, hit the unchanged
30-minute Python test-step deadline. Its last console progress line was 88%
at 05:40:01 UTC, but that did not mean a test was hung. Debug artifact
`13832437333` contains per-test receipts showing uninterrupted progress:

- All 24 physical Stuttgart CPU cases completed by 05:36:54, and all 26
  halogen CPU cases completed by 05:36:50. Neither was the long tail.
- The first public CPU receipt appeared at 05:37:30, almost four minutes
  after the first Stuttgart/halogen receipts at 05:33:38. Xdist's default
  descending-test-count order delayed this file despite its being first in
  the selected target list.
- The twelve new Cartesian/spherical, RKS/UKS, LDA/PBE/B3LYP f-force cases
  progressed from 05:40:20 through 06:03:08. The six spherical cases each
  took about 172 seconds including independent references and replay;
  their recorded public endpoint times were 82.7–83.4 seconds. These are
  distinct timings, not interchangeable performance claims.
- The capability control completed at 06:03:12; the final resource-plan
  control had no receipt when the external deadline expired at 06:03:37.
  Three other workers had reported orderly completion 381.6, 606.1 and
  657.5 seconds after the controller's session record. Their records show
  no worker error or OOM event. All four timed traceback files were empty:
  individual tests continued finishing before the 900-second watchdog.

The twelve numerical f-force cases were added in `b734dc887d`; an earlier
same-base PR's roughly 699-second ECP shard is not a matched timing baseline
because it lacked this new qualification matrix. This diagnosis identifies a
serial scheduling tail, not a newly demonstrated production speed regression.
It does not dismiss future credible matched performance evidence.

## Decision and invariants

For routine PR/push/merge-group ECP shards, pass `--no-loadscope-reorder` to
the existing `loadfile` scheduler. The existing target list already puts
`test_ecp_public_cpu.py` first. No file-order helper, shard transfer, or
scientific test change is needed. Preserve all selected files, node IDs,
skips, numerical gates, compiler caches and finite time limits. Nightly and
manual selection remain unchanged.

Whole-file ownership is essential, especially for DF source-emission and
Hessian fixtures. Do not replace `loadfile` with item-level work stealing to
redistribute this tail. Do not remove f-force or physical ECP cases, mark
them nightly, or increase deadlines merely to obtain a green check.

## Verification and limits

Shell-selection regression coverage checks all five event types and all
shards. A real xdist subprocess comparison uses the actual ECP selection
with cheap synthetic module fixtures: the public CPU file starts on an
initial worker, every selected node/outcome is retained, every module stays
on one worker, and every module fixture executes once. The synthetic check
does not qualify molecular numerics or predict a hosted wall time.

The existing CI run is left untouched. A fresh hosted run must determine
whether starting this file earlier provides enough headroom; the prior
deadline prevented a complete public-file duration measurement. Revisit
scheduling only from complete timing evidence if the unchanged budget still
fails. Keep the original failed run and its incomplete result visible.

## References

- https://github.com/jinzhezenggroup/generativeqc/actions/runs/42965903275/job/130343104062
- https://github.com/jinzhezenggroup/generativeqc/actions/runs/42965903275/artifacts/13832437333
- https://github.com/jinzhezenggroup/generativeqc/commit/b734dc887dcef0821237e85fb9ba5f66e6949328
