# Decision: source-bound public Python contract facets

Status: implemented
Date: 2026-10-09

## Problem

Issue #2107 needs substantive API contracts, not only exported names or nonempty
strings. The held #2115 gate followed ordinary re-exports but missed literal lazy
compiler maps, simple class aliases, modules and value/type/dtype tokens.

## Decision

Resolve every facade export statically to its source owner and inventory all
explicit public/magic members. Preserve the original implementation docs and
follow aliases to one owner. Supported lazy maps remain finite literal data;
unknown forms and source paths outside the package root fail closed. Introspection
must not import optional/native/GPU modules.

The checked-in manifest binds exact exports and source declaration names to an
applicability profile and authoritative facets. Execution requires shape/unit,
failure/publication, ownership/concurrency and backend contracts. Accessors and
version tokens retain concise behavior instead of irrelevant sections. Family
contracts are explicit labeled sections, not a universal backend support matrix.
The current debt list is empty and the gate rejects a nonempty completion baseline.

## Rejected alternatives

- Nonempty docstrings accept unfinished text and cannot establish coverage
- Word counts penalize adequate accessors while admitting verbose empty prose
- Import-based introspection activates optional runtime/compiler dependencies
- Automatic semantic classification can mistake a live diagnostic for a pure
  property; the manifest classification is explicit reviewable policy
- Broad unbound exemptions silently hide new API forms and retired source owners

## Invariants and evidence

Negative unittest fixtures cover undocumented lazy objects/members, assignment
aliases, unknown exports, cycles/path escapes, same-name module/function binding,
placeholder prose, missing/new declaration classification, alias drift and loss
of an applicable facet or its referenced body. The runtime edits change only
docstrings; compare executable ASTs against the integration baseline. Sphinx
references remain warning-as-error checked. Source review and numerical/domain
tests remain necessary: structural coverage does not prove prose factual.

## Consequences and revisit conditions

New public forms require a deliberate resolver and negative fixture. New/changed
APIs must select their applicable profile and contracts, without copying generic
links over a missing operation-specific promise. Revisit when another stable
export convention or a better structured source-doc format is introduced. Do not
replace capability discovery with a handwritten cross-method support matrix.

References: #2107, held #2115, tools/check_public_api_docstrings.py,
manifests/python_api_contracts.json, docs/reference/python_contracts.md.
