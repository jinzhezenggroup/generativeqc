# Decision: omit redundant CuMetal work for verified README edits

Status: implemented
Date: 2026-10-13

## Problem

The separate Apple-GPU workflow runs native compilation and QC qualification for
all live PR/queue events, independently of the main CI README tier. Root README
bytes are not inputs to its native/codegen, compiler-cache or QC-JIT identities.
For natural README-only PR #2455, CuMetal job 115436256241 ran 11m42s, including
9m04s in the native runtime-target build. This is an observed baseline, not a
measurement of the savings from this change or fresh numerical qualification.

## Decision and boundaries

Reuse `tools/ci_prose_scope.py` unchanged. Only a successful, exact `prose\n`
result can omit work; only modifications of the existing regular UTF-8 root
README can produce that result. All other paths, additions/deletions/renames,
executable modes, unknown events, missing refs, and errors retain full work.
Neither a prose suffix nor the last README-only queue node authorizes omission.
The existing classifier resolves the complete unmerged queue against the live
trusted target, rejecting divergent/ambiguous/insufficient ancestry.

The existing Mac job still checks out the immutable event tree with depth one,
credentials disabled, before selection. No runner, serial prerequisite job, or
required check name is added. For PRs, `changed_files` other than exactly `1`
rejects the candidate without starting any subprocess. This metadata never
approves a skip; one-file source PRs still require the complete Git proof.
Merge groups do not use PR metadata. Schedule/manual/unknown events do no extra
fetching or classification and retain their existing work. There is no master
push trigger in this workflow; none is introduced.

A possible PR/queue candidate fetches only its exact event SHA at depth two into
an isolated ref. It does not follow a moving branch or switch the worktree.
Depth two is deliberately not assumed sufficient for stacked queues: missing
ancestry remains full, including an all-README stack whose full range cannot be
proved. If an earlier prefix has actually landed in trusted master, its already
landed files no longer require duplicate queue qualification.

A portable Python wrapper bounds fetch plus classification to 75 seconds, with
at most five seconds of timeout cleanup inside the existing two-minute step
budget. Each child runs in an owned process group; timeout terminates that group,
including descendant Git processes, without touching unrelated processes. No
GNU timeout installation, credential, permission, or cache-policy change is
needed. A failed selector step or missing output also retains full work.

Every original expensive, post/failure, summary and artifact step keeps its
original predicate conjoined with the fail-full scope guard. This includes the
historical one-shot diagnostics, whose PR/before-SHA/attempt/failure constraints
remain distinct and unchanged. Full scientific endpoint lists, numerical
acceptance, explicit existing quarantine/provenance reports, and scheduled/manual
qualification stay intact. No source, test manifest, compiler flag or cache key
changes. Root README claims remain subject to editorial review; this adds no
new automated claim-validation policy.

## Upload provenance

The single, non-matrix producer exposes `prose` only when its selector step
actually succeeded with that marker. The upload-only job is omitted only for a
successful producer carrying this positive proof. A failed producer, failed or
cancelled selector, missing/unknown output, and unrelated main-workflow scope
outputs cannot suppress real result upload. Existing global cancellation and
producer admission remain intact. The upload job's full-history checkout, OIDC
permissions, artifact name, Codecov inputs and report type are unchanged.

No fake JUnit/coverage report is created for omitted scientific work. The main CI,
Documentation and Pre-commit workflows are outside this change.

## Evidence and tradeoffs

Executable tests exercise the actual inline selector and real depth-one Git
repositories: README/source PRs, moved event refs, failed fetches, missing refs,
stacked source/workflow changes below a README tail, insufficient all-README
history, landed prefixes, and divergent targets. Metadata-only rejection launches
no subprocess. Classifier nonzero/malformed/padded/missing outputs fail full.
Actual timeout tests terminate a child plus its inherited-pipe descendant while
an unrelated process survives. Admission tests cover scientific tiers, original
cache predicates, one-shot authorizations, post/failure/cancellation cleanup and
single-producer upload trust with mixed main-workflow outcomes.

A one-off parsed comparison against the parent verifies all 35 original producer
step bodies/action inputs unchanged, 37,800 full-tier condition decisions
identical, existing job dependencies/timeouts intact, and unchanged upload
contents. This is migration evidence, not a permanent source-spelling test.
Existing shallow checkout, quarantine, diagnostics and runtime-provenance tests
remain green. No local Apple-GPU or numerical campaign is claimed.

Multi-file PRs incur only local Python startup. A one-file PR or queue candidate
can incur the bounded fetch/classifier work even if ultimately full. Hosted
latency of that new fetch is unmeasured before publication; observe normal
source/queue runs rather than manufacture a PR or launch an optional benchmark.
The depth-one checkout and runner queue costs remain for successful prose cases.
Revisit only with evidence if the limited ancestry fetch rejects too many genuine
README queues; do not relax full-queue or trusted-target proof to increase hits.

References:
- [README-only baseline run](https://github.com/jinzhezenggroup/generativeqc/actions/runs/38436975337/job/115436256241)
- [Main CI README tier](https://github.com/jinzhezenggroup/generativeqc/pull/2465)
- [Depth-one checkout rationale](2026-10-13-cumetal-shallow-checkout.md)
