# Decision: keep SCF acceptance history out of the module-boundary map

Status: implemented
Date: 2026-10-25

## Problem

`docs/developer/scf_module_boundaries.md` is the current-state ownership map for
HF/SCF: who owns which module, which dependencies are legal, and which
verification gates apply. It had accumulated a closing section that narrated a
pinned 2026-09-22 acceptance snapshot, its failed runtime tests, a tracking-issue
reconciliation, and the repair sequence that followed.

That content is genuinely valuable — it records that a clean Release/CUDA build
and a green runtime suite can disagree, and that structural completion is not
runtime acceptance. But placed in the ownership map it reads as a status claim
about today's `master`, invites re-litigating closed issues, and drifts from
reality as soon as the next fix lands. Readers also had to distinguish which of
the surrounding statements were pinned history and which were current rules.

## Decision

The ownership map keeps only rules that are true now: the ownership inventory,
dependency invariants, the current Direct/DF scope, and the gates a refactor
must satisfy. Historical acceptance narration moves here.

Two behaviours remain documented in the current page because they are rules, not
history:

- a source-level refactor needs CPU and allocated-device RHF/UHF
  energy/force, prepared-batch, changed-geometry, resource/failure and
  independent-reference gates in addition to the structural checks; and
- a build result and a runtime result are separate claims, so a clean
  compilation never substitutes for executed numerical evidence.

The concrete old measurements stay where they already live and are linked rather
than copied: `benchmarks/results/acceptance-closeout-20260922/` (the pinned
acceptance audit) and `benchmarks/results/scf-direct-native-rtx5090/` (matched
direct/DF numbers, builds, incremental objects and timings).

## Rejected alternatives

- **Keep the narration and label it "historical".** Tried first: the section was
  already titled "Historical acceptance, without a stale project-status claim".
  Labelling did not stop it from being read as a status statement, and it still
  required maintaining issue-by-issue progress text in a current-state guide.
- **Delete it entirely.** Rejected: the build-green/runtime-red lesson and the
  reason bucket/Graph lifetime is separate from numerical launch order are the
  kind of thing a future agent will otherwise rediscover.
- **Move it to the Maintainer Guide as a validation case study.** Rejected:
  `maintainer/validation.md` defines how evidence is produced and retained; a
  single historical episode does not belong in that procedure either.

## Invariants

- `docs/developer/scf_module_boundaries.md` states present ownership and present
  gates only.
- Historical failures are never presented as the current test status of any
  branch.
- Raw measurements stay in `benchmarks/results/`; this note does not restate
  their numbers.
- Structural checks (`test_scf_structure.py`, ownership inventory) are never
  described as runtime or numerical acceptance.

## Evidence

- `benchmarks/results/acceptance-closeout-20260922/build-audit.json.gz` and
  `build-times.json` record the clean Release/CUDA build.
- `benchmarks/results/acceptance-closeout-20260922/df-384-runs.json`,
  `df-768-runs.json` and their ablations record the runtime work measured at
  that revision.
- `benchmarks/results/scf-direct-native-rtx5090/measurements.json.xz` and
  `production-acceptance-attempt-2026-09-21.md` record the matched
  direct/DF numerical and timing comparison.
- The charge-DF UHF final-state convergence defect that the reconciliation
  identified is recorded as issue #1041.

## Consequences

The ownership map is shorter and no longer needs editing when a historical issue
closes. A reader who wants the episode must follow one link. The cost is that
the page no longer warns in place about a specific past trap; the "build result
is not runtime result" rule above preserves the actionable part.

## Revisit when

- A new acceptance audit contradicts the rules that remain in the page.
- The retained benchmark archives are retired or their manifests change.
- A genuine, currently-failing gate needs to be visible in the ownership map.

## References

- `docs/developer/scf_module_boundaries.md`
- `docs/maintainer/validation.md`
- `docs/maintainer/performance_engineering.md`
- `.agents/notes/implemented/architecture/2026-09-18-direct-hf-host-control.md`
- `benchmarks/results/acceptance-closeout-20260922/README.md`
- `benchmarks/results/scf-direct-native-rtx5090/README.md`
