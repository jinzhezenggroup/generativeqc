# Decision: retain independent references across implementation-only changes

Status: implemented
Date: 2026-10-21

## Problem

The default-grid campaign's independent WB97M-V/96 reference takes about
77 minutes for XC5/6 and VV10 4/5. Recomputing the same independent scientific
target after each native cache or force-scheduling fix adds no new input
coverage. Conversely, merely removing the source-revision check would allow
accidental reuse across changed basis or model inputs. Original oracle v1
receipts did not export those fields separately.

## Decision

Keep the existing same-revision rule unless an explicit reuse manifest binds
the canonical hash of an entire original reference receipt to its source
revision, actual driver hash and reviewed scientific inputs: method, orbital
basis/representation, charge and spin. Reject missing or mismatched bindings.
An original reference cannot be edited while retaining that binding. Native
measurement receipts, references and producer identities remain unchanged.

This manifest is an input review, not a numerical pass flag. Recheck exact
geometry, package versions, tighter independent controls, dense-grid stability,
matching CPU/GPU spots and every cold/warm error against unchanged gates.
Return the used bindings in qualification evidence. The renderer still rejects
mixed native measurement revisions; all three timed profiles are remeasured
on the unified implementation. Only untimed numerical references are reused.

For the frozen source02337 campaign, review the actual retained driver overlay
on n1, not a guessed Git path: the native source revision and driver hash are
separate provenance fields. The original/current method mapping, orbital basis
owners, r2SCAN-3c basis and corrections, and full-density reference-Fock helper
are unchanged in the audited inputs. Preserve their actual file hashes and the
reference source paths with publication. The one-off semantic source comparison
is review evidence, not a permanent function-body fingerprint test.

## Invariants and evidence

Tests reject unreviewed revisions, changed receipts/drivers, different basis,
representation, charge/spin and missing provenance. A reviewed but unstable
reference remains unverified; an inaccurate warm sample still fails. Original
receipts are never modified by qualification. The selected forty independent
reference receipts reproduce the original 170 passed, 18 failed and four
incomplete/unverified baseline populations before reuse. No tolerance, timing
boundary, reference default or failed population is changed by this policy.

Revisit the legacy-input binding once future oracle receipts export their full
scientific model directly. Do not turn an audited legacy exception into implicit
acceptance of arbitrary old calculations.
