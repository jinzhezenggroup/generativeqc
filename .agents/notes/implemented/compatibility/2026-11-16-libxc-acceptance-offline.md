# Decision: keep retained Libxc scientific acceptance out of routine CI

Status: implemented
Date: 2026-11-16

## Problem
The retained-catalog checker compares live dependency identities with historical
campaigns. Even with advisory generated views, exact transition acknowledgements
can block unrelated PRs after source drift. Rerunning a specialized CPU/CUDA
campaign on every such change is not an available routine-CI resource.

## Decision
Remove the standalone retained native Libxc catalog revalidation step from the
shared PR/merge-group CI workflow. Do not replace it with a softer freshness gate.
Ordinary numerical, receipt-integrity, corruption-rejection and qualification
logic tests remain. The historical domain/CPU replay uses the existing
authenticated recorded-contract fixture instead of requiring its mathematical
and method identities to match every later checkout. Live source demotion and
corrupted provenance continue to be tested separately. The offline checker, strict snapshot checks, explicit
`--require-current` acceptance and archival manifests are unchanged.

## Invariants
Historical measurements are never relabelled as current results. New scientific
or performance claims still need their appropriate independent acceptance.
No branch protections, required aggregate checks, production paths or numerical
tests are disabled. This change does not alter the force-product work in #3090.

## Rejected alternatives
Making generated views advisory alone (#3103 originally) leaves live transition
acknowledgements blocking routine CI. Exempting broad build files from source
identity would instead misrepresent which executable was actually measured.

## Consequences
Ordinary CI establishes tested behavior, not freshness of a specialized retained
campaign. Maintainers invoke `python tools/check_libxc_native_catalog.py` or its
`--require-current` mode explicitly when reviewing a new acceptance claim.
