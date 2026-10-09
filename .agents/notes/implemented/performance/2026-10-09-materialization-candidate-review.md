# Decision: source-bound materialization candidate review

Status: implemented
Date: 2026-10-09

## Problem

Issue #1626's output-materialization count is useful only with source identity.
An unchanged count can conceal a replacement site; a helper refactor previously
hid the dense MP2 first stage without removing its buffer (#1998 repaired the
classifier). Line numbers cannot identify a review after source movement.

## Decision

Reuse `audit_native_complexity.audit_text`, the work audit's function inventory,
and the structured audit's strict captured-byte/provenance boundary. The new
comparison scans immutable baseline sources and the current tree with the same
import-bound analyzers, without executing baseline code. No second C++ parser,
scientific rewrite, numerical promotion or timing campaign is introduced.

Version the six current dispositions under `manifests/`. Match exact site
identities and whole-file producer/consumer source bindings. Site identities
survive line motion. Ordinary non-preprocessed files normalize comments and line
motion while preserving literal payloads. Preprocessing spellings, raw/line-spliced
literals and line-sensitive builtins conservatively bind exact bytes: textual
changes there require renewed reviews. No preprocessor equivalence is claimed. Record raw file hashes separately. Verify imported analyzer bytes before
and after scans so edited files cannot relabel already loaded code.

Require owners and reasons, preserve added/removed/changed sets and count deltas,
and reject missing, duplicate, stale or unmatched reviews. Missing/malformed
baseline or evidence is INCOMPLETE and fails the enforcing CLI. CI retains the
compared receipt on failure. The first adoption is supported by scanning base
sources with today's analyzers; no historical manifest is invented.

## Scope and retained gaps

At source base `5d2aab030f6fdd562a57e176aefced0017fa5103`, the established production
complexity census is 635 files, 177 high-order sites and six output-materialization
findings: second/third/local shell transforms, dense and factorized first-stage
MP2 pullbacks, and dense initial orbital-RHS weights. All six are retained pending
evidence, with producer/consumer, residency, lifetime and resource-owner records.
These records do not prove required physical materialization or profitability.

The three old RCCSD(T) projection candidates were retired by #1625/#1632. Their
historical numerical/resource/endpoint acceptance is retained in the issue and
`benchmarks/results/cc-triples-response-cuda-20261003/`; it is not rerun or promoted
to current-binary evidence by this review gate. In particular, total endpoint
improvement is not attributed solely to projection fusion. The old count five
was a scanner false negative, not a fourth retired pass. #1626 remains open.

## Rejected alternatives

- A count-only limit: replacements and hidden helper calls defeat it
- A blanket high-rank failure or exemption list: neither records actionable owners
- Binding line numbers: harmless source movement discards review identity
- Treating bounded host staging or a dense consumer span as a necessity proof:
  resource bounds and an existing ABI do not establish fusion safety or speed
- Executing the old baseline scanner: parser changes can masquerade as retirement

## Limitations and revisit conditions

Function recognition and classification are existing bounded lexical analyses.
Producer/consumer symbols are declared review context, not verified dispatch.
Bindings cover whole declared files, not all transitive dependencies; changes to
unrelated code in those files intentionally cause conservative review churn.
No findings is not evidence of no materialization. Revisit identity precision
only with validated ownership/call-graph evidence, retaining missing/unknown
states and the no-count-only invariant.

## Evidence

`tests/python/test_native_materialization_review.py` covers producer/consumer
changes, line movement, same-count replacement, removal evidence, stale/missing
records, literal identity, exact Git baseline bootstrap, incomplete reports,
import-bound analyzers (including reloads before and during scans), directive
line semantics, line-sensitive builtins, malformed JSON shapes and CI receipt
retention. Existing complexity, structured
and shared-work audit tests protect the reused classifier/provenance boundary.
