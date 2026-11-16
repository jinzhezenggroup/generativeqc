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

The redistribution is a separate implementation commit integrated into #3099
alongside its original whole-job-budget commit, avoiding circular integration
dependencies between two independently failing CI prerequisites. After normal
integration of actual accepted #3096 and both repairs, 1769 focused contracts
pass; normal and optimized Libxc integrity checks, Ruff and Bash checks pass.
The PR diff against accepted master remains seven operational/test/note paths,
with no native input change. Full protected CI still determines acceptance.

## Preserve workload ownership when the AOT qualification files arrive

The next exact gates head `cbcc498b647ddcc9d64a4064b425b5e1a1290e0e`
fails core-c job `132985948630` in run `43770757909`: the actual test step
times out at its unchanged 30-minute deadline after reaching 99%, without a
completed JUnit report. Retain its log, annotations and debug artifact
`14184614699`; a printed near-complete percentage is not acceptance.

An executable historical replay with separate Git indexes establishes an
additional root cause. P1b adds `test_aot_policy_dimensions.py`; P1c adds
`test_cuda_aot_global_promotion.py`; evidence adds the compile/endpoint replay
modules. Inserting the first one or two into ordinary sorted modulo enumeration
changes the owners of **2227 existing files** against accepted #3096 master.
The four-file evidence inventory changes 2228, including the two intentional
Hessian transfers. These are actual shell-selector/inventory observations,
not a source-shape hypothesis or a numerical assertion.

Assign only those four new qualification modules explicit supplemental owners,
using the existing no-modulo-slot mechanism. Keep every case and the accepted
master's existing file ownership, apart from the two separately measured
Hessian transfers. This prevents each new layer of the same PR stack from
silently moving unrelated numerical/codegen workloads and cold probe locality
between the three core workers. No runtime deadline or scientific gate changes.

The insertion regression exercises all five events and tracked, missing,
untracked and absent states, retaining exactly-once selection and every
unrelated ordinary owner. Historical replay inputs/results remain under
`aot-shard-insertion-historical-diagnosis-v2.*` in the ignored validation
directory; live repository indexes are never replaced. Fresh complete author
and protected runs remain required to qualify this final distribution.

The final insertion regression passes all 100 targeted cases, and 1789
integrated CI/coverage/cache contracts pass. Replaying the actual master,
P1b, P1c and evidence inventories under the fixed selector proves zero
unrelated owner migrations at every partial stack. Against the original
accepted-master selector, exactly the two intentional measured Hessian
transfers remain. Bash, Ruff and whitespace checks pass; native inputs and
all existing deadlines remain unchanged.
