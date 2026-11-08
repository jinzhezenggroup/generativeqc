# Decision: retire the rejected frozen-VV10 history opt-in

Status: rejected experiment; implementation retired
Date: 2026-11-09

## Decision

After explicit user authorization, retire the default-off
`GENERATIVEQC_KS_VV10_REBASE_HISTORY` experiment from PR #3037 rather than
retain a production-selectable path that failed its original endpoint gates.
This supersedes the implementation's
[original rationale](../implemented/numerics/2026-11-07-vv10-exact-history-rebase.md)
and follows the unchanged
[negative qualification decision](2026-11-07-vv10-history-cold-promotion.md).

## Evidence and scope

The first complete A100 and RTX5090 cold endpoints regressed by 7.075% and
11.874%, respectively, despite one fewer Fock. Both retained an extra SCF VV10
pair sweep and lost an exact component-cache hit. The paired force differences
also failed their original preservation gates. These are independent refusal
reasons; retiring the experiment does not relabel any endpoint as accepted or
establish an exclusive timing cause. The original raw measurements, failed
verdicts, source/image identities, independent numerical controls and sanitizer
receipts remain unchanged in
`benchmarks/results/vv10-history-cold-20261107/` and Git history.

Remove the environment selector, Python resource-request metadata, optional
capacity reservation, retained density/operator banks, transaction kernels,
and the complete Gram rebuild used only by this transaction. Remove their
promotion/ownership entries and tests of the retired interface; retain all
incumbent Gram, KS admission, cleanup, precision and independent scientific
coverage. No obsolete alias or compatibility shim remains.

## Validation boundary

Restore the affected production consumers and existing tests directly to
master `8b9983820`. A one-off byte comparison verifies that the removed
experiment leaves no production or native-build-input difference from that
baseline. This comparison documents this retirement only; it is not a permanent
source-fingerprint test. The generated Libxc catalog must still be refreshed
with its official writer, not manually relabelled as qualified.

No unchanged rejected experiment is rerun, no evidence is deleted, and no
scientific threshold or resource budget is relaxed. This change records the
failed investigation rather than claiming a new optimization or GPU result.

## Revisit conditions

A future proposal needs a materially different causal mechanism, independent
numerical and lifetime validation, and complete cold plus paired-warm evidence
at the original gates. A lower Fock count or successful transport alone is not
sufficient. Do not revive the retired selector as a default-off fallback.
