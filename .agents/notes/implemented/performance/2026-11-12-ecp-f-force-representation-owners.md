# Decision: separate orbital-f ECP force representation owners

Status: implemented; hosted scheduling validation pending
Date: 2026-11-12

## Problem

PR #3057 run [43221290971](https://github.com/jinzhezenggroup/generativeqc/actions/runs/43221290971),
ECP job [131193221463](https://github.com/jinzhezenggroup/generativeqc/actions/runs/43221290971/job/131193221463),
exhausted its existing 30-minute **test-step** deadline. The outer job budget was
not the limitation. This was not an apt/DNS failure or a recorded failed numeric
assertion; the incomplete run is not a scientific pass.

Retained debug artifact 13949025723 has 33 completed public-file receipts,
including eleven of twelve f-force cases. The remaining spherical B3LYP UKS
case and the following default-capability/resource-plan cases have no completed
receipt. The controller records the other three workers finishing by roughly
822 seconds, leaving one serial file owner. Saved watchdog stacks are empty,
so no precise native stack at timeout is claimed. Retained resource samples
show zero OOM counters and ample available memory, not a measured endpoint
performance diagnosis.

The current-master passing queue run 43157119998, job 130979546219, retained
pytest timing artifact 13918550996. `test_ecp_public_cpu.py` alone took 1721.849
seconds, while the entire shard took approximately 1726 seconds. The 1800-second
limit therefore had only about 74 seconds of margin. Current completed f-force
endpoint receipts are typically about 21% slower than that run; other families
also slowed. Different hosted runs do not establish a source performance
regression or prove its absence.

## Decision

Keep the exact f-force gate in a non-collected helper. Dispatch all six methods
through thin Cartesian and spherical test modules, following the existing s/p/d
representation-wrapper pattern. Retain their autouse evidence fixture. Give the
long spherical file the first `loadfile` slot and the Cartesian file the second;
keep both on the existing ECP lane and keep each file on one worker. The old
public module retains its capability and resource-plan checks.

These f-force cases already construct their independent prepared endpoints and
oracles per invocation; no module-scoped shared prepared-state fixture is broken.
The common persistent compiler cache is retained. Both new files are excluded
from ordinary core modulo enumeration, so their introduction does not move
unrelated existing core owners.

## Invariants and validation

- All twelve combinations remain: Cartesian/spherical times LDA/PBE/B3LYP and
  restricted/unrestricted spin. No case moves to nightly-only selection.
- Scientific helper arguments and body, analytic and both reconverged finite
  difference checks, native-owner failure injection, thresholds, g/f capability,
  precision, and total scientific work are unchanged.
- No workflow parallelism, deadline, job count, compiler setting or CI retry is
  changed. Use the existing four workers; do not cancel running CI.
- Executable regression tests use the actual shell selector for all five events
  and check exclusive exact-once file ownership and twelve-case collection.
- A deterministic two-representation barrier runs the actual wrappers under
  xdist/loadfile. It verifies distinct workers, complete same-file ownership,
  forwarding to the unchanged gate, native-owner refusal, no duplicate case
  execution, and all twelve autouse evidence receipts. The oracle is substituted
  only for this scheduling test, which is not numerical qualification.
- A one-off before/after AST comparison verifies the helper body and arguments
  are identical; no permanent source-hash/spelling test is introduced.

Full hosted runtime timing and completion remain to be observed. No local full
native ECP qualification is inferred from orchestration tests. In particular,
the unfinished spherical B3LYP UKS case is not relabeled as passed.

## Rejected alternatives

Do not increase the deadline merely to fit an already serial file near the cap.
Do not remove methods, representations, force checks or independent references.
Do not replace all ECP scheduling with case-level work stealing: other modules
have measured shared fixtures, whereas this bounded split preserves their
existing ownership and cache behavior.
