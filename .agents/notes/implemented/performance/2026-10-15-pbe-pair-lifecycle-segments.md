# Decision: retain PBE pair lifecycle segments and their gaps

Status: implemented
Date: 2026-10-15

## Problem

The point-family producer constructed each Calculator before its preparation
clock. The shared README protocol subsequently described constructor-inclusive
cold work, while the retained 48/96 references had the older protocol wording.
An actual H100 feasibility call stopped at the strict protocol assertion before
E+F execution. Its four-thread wrapper also differed from the reference's eight
threads. Neither failure is a numerical endpoint result.

The retained reference source is checksum-bound to
`83109befe5b97a5542b30e169da74c81d04719e6:benchmarks/readme_omol25.py`.
That reference started its preparation clock before `reference_engine` and
added preparation to its first E+F duration. Merely changing protocol wording
would conceal the native producer's missing constructor measurement.

## Decision

Point-family v3 separately measures constructor, owner preparation and initial
E+F intervals. The old preparation/setup `seconds` fields retain their meaning.
The receipt adds each arm's own-work sum, observed first-to-last span and the
excluded interleaving/provenance gaps. Moved setup records its actual coordinate
rebuild inside execute and does not charge the initial constructor again.

Both arms are prepared before setup; unrelated work lies between an arm's
segments. Their sum is not a continuous cold wallclock measurement. Existing
warm/moved-warm replay timers and scientific/work gates remain unchanged.

The consumer recomputes segment costs and gaps from timestamps, rejects missing,
nonfinite, overlapping or wrongly typed measurements, and requires v3 lifecycle
accounting for current point-family promotion. Historical v1/v2 receipts remain
auditable within their original replay scope.

## Rejected alternatives

- Rewriting old reference protocol/timing data or ignoring the differing field
  would lose the reference's actual measurement contract.
- Calling a segment sum continuous cold timing would hide interleaving gaps.
- Adding preparation or the initial constructor again to moved setup would
  count work that the moved execute did not perform.

Strict reference equality remains. A new reference population must use the
current canonical protocol and original basis/grid, convergence and error gates.
Fresh reference production and complete endpoint profitability are separate
qualification steps; this instrumentation change does not establish either.

## Evidence and revisit conditions

Host tests delay constructor, preparation, first E+F, moved E+F and provenance
work independently of clock-read counts. Consumer mutations cover omitted or
duplicated charges, gaps, overlaps, forged spans, bool-valued times and schema
downgrades. Existing vector, semantic-work and equal-spin controls remain.

Revisit the segmentation if a dedicated isolated cold process is introduced.
Such a producer needs an explicit timing contract and its own source-matched
reference/acceptance evidence.

Refs #2072 and PR #2250.
