# Decision: source-owned native API contract coverage

Status: implemented
Date: 2026-10-09

## Decision

Public native contracts remain beside their declarations in the two supported
headers. A bounded static checker binds the reviewed function/operation
signatures, C++ public record fields and applicable contract facets, then
renders stable per-symbol reference targets. The shared docs workflow runs
negative fixtures, both language gates and the existing warning-as-error
Sphinx build. The native gate also runs in pre-commit.

The C inventory covers literal exported functions; unsupported alternate export
forms fail closed. It is not a complete C parser or a field-by-field C descriptor
inventory. C++ public field types/defaults are bound, and private hidden friends
are rejected because access labels do not make their free functions private.
Constructors and destructors have distinct, checked reference identities.

## Why

A nonempty adjacent comment or a native overview cannot establish usable public
contracts. Applicable structured facets make missing coverage visible; source
review is still required to establish factual correctness. Short identity or
version accessors do not require meaningless numerical/buffer sections.
No declaration has a debt exemption.

## Preserved behavior

The native headers remain executable-token identical. Documentation distinguishes
copied preparation inputs from borrowed handles, descriptor-specific ABI admission,
solve failure from later output-buffer rejection, and explicitly invalidated
records from retained prepared-owner profiles. Diagnostic availability does not
prove a completed or fresh solve; default provenance can be exposed after early
nonconvergence. No runtime change is made to manufacture a stronger guarantee.

## Evidence and revisit conditions

Static negative fixtures reject ordinary undocumented declarations, arbitrary
comments, missing facets, field/signature drift, hidden friends and unsupported
export forms. Source review checks failure/lifetime semantics. Generated anchors
are unique and Sphinx remains strict. Extend the bounded parser and reviewed
inventory explicitly when a genuine new public declaration form is introduced.
Do not silence an unsupported form or substitute a signature count for contract
truth. Related task: #2107; the held #2115 foundation is retained intact.
