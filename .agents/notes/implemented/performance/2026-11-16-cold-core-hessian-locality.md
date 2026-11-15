# Decision: Move the measured cold Hessian tail to file-locality workers

Status: implemented
Date: 2026-11-16

## Problem

The independent whole-job repair in #3099 preserves the existing test-step
allowances after a cold native build; it does not guarantee that the selected
test workload fits those allowances. Protected run `43767173517`, queued source
`b9502149b7578c3e4c05667a1c4edf2eeba26770`, exposes that separate problem:
core-a's test step times out after 30 minutes at 95%. The actual aggregate
rejects the run. No completed JUnit result exists, and partial progress is not
complete qualification.

Artifact `14185095709` retains the workers' native diagnostic events. Completed
setup/call reports account for at least 690.05 worker-seconds in
`test_second_component_schedule.py` and 450.29 worker-seconds in
`test_hessian_rks_directional.py`. These are lower bounds: work active when
GitHub terminates the step has no final duration report. Workers remain active
in different tests; the records do not establish a numerical failure or OOM.

In the same protected run, runtime-heavy completes its test step in 503 seconds
and compiler-heavy in 470 seconds. Accepted #3096's core-a test step previously
passed in 1253.45 seconds on the same native source tree, so success on a faster
host/cache is not proof that the cold workload is bounded appropriately.

## Decision

Transfer both complete Hessian modules to the existing runtime-heavy loadfile
lane through `measured_core_runtime_paths`. Keep the RKS module's original
modulo slot. The second-component module is supplemental and never had a
modulo slot: retain that absence by excluding it through `slow_paths` before
supplemental core dispatch. Every unrelated file retains its owner. The RKS module's
module-scoped fixture also stays with one worker instead of being replicated
across the core worksteal workers. The global-hybrid public Hessian suite
(another 922.05 observed worker-seconds) retains its existing parallel fresh
frontier and every independent lifecycle and failure-recovery test; moving it
would change that separately required ownership/priority contract.

Exercise actual shell selection for all eight shards and five CI events. The
before/after inventory must contain every tracked test module exactly once;
only each named module may change owner. Preserve distribution, coverage
arguments, routine deselections and nightly/manual qualifications.

## Rejected alternatives and invariants

- Rerunning until caches or a faster host hide the tail is not a repair.
- Do not extend test-step or whole-job deadlines, skip cases, relax scientific
  assertions, or remove diagnostic/reporting steps to manufacture a pass.
- A new shard adds another cold native build; use measured spare capacity in
  an existing lane first, while letting complete CI reject an overloaded lane.
- No mathematical/codegen/default/source-identity input changes. This repair
  does not qualify GPU science or resolve the independent TZVP force issue.

## Evidence and revisit conditions

Failure log, annotations, diagnostic archive and per-module duration reduction
are retained under the ignored acquisition directory
`/data/jzzeng/issue3086-qualification-validation-20261115/`, with prefix
`python-budget-protected-core-a`. Source paths and the protected run/artifact
identities above permit independent retrieval; the incomplete diagnostic is
not substituted for a completed test report.

Revisit after fresh complete author and protected CI: measured redistribution
is a hypothesis until both lanes complete within their unchanged finite limits.
Track this operational blocker under #3086; #3099 remains an independent
whole-job-budget fix, not this test-work distribution repair.

Local validation passes 1718 shard ownership, time-budget, coverage and probe
cache tests. The executable before/after checks preserve every selected test
file exactly once across all eight shards and five events, including tracked
missing-file failure, untracked/absent exclusion and the supplemental no-slot
case. Bash syntax, Ruff lint/format and whitespace checks pass. All three
changed paths are outside the 2183-input native source inventory.
