# Decision: share the generated built-in basis owner outside the CLI

Status: implemented
Date: 2026-10-16

## Problem

The installed CLI already expanded the audited built-in decimal basis pack in
C++, while the library/Python runtime could not reuse that owner. Linking CLI
parsing into the core would also pull in local JSON, command-line errors and
crypto/file-policy dependencies. Reimplementing the catalog for a public bridge
would instead establish a second owner and risk scientific input drift.

## Decision

Move the existing built-in discovery and atom-ordered expansion to the molecule
layer. CLI names and data types are aliases to that owner; its canonical custom
record/ECP parser remains in the CLI. Generate the AOT table for the library even
when the CLI is disabled, with one generated family and one production source.
Preserve every primitive, coefficient, packed shell order and established name
alias. Add checked index/count/capacity metadata before detached allocation.
These buffer counts exclude allocator overhead and complete system/provider
resource admission; coordinates are irrelevant to catalog shape.

The canonical source-identity inventory now explicitly includes both the audited
basis JSON and its generator. They were formerly CLI-only build inputs, but a
library consuming this table must invalidate identity if either changes. A
matching filename or a custom record's user-supplied name is not authority to
substitute built-in data.

## Scope and remaining work

This private shared owner is the prerequisite for a versioned installed catalog
and thin built-in Python resolver. It does not itself migrate Python model/basis
resolution, authenticate a custom basis, change its identity/normalization or
provide complete resource authority. The compiler, independent oracles and
explicit custom interchange remain separate owners. No complete #2227 R5/R6
cutover or performance/GPU qualification follows from this structural extraction.

## Validation

Native behavior covers discovery, aliases, mixed atom ordering, Cartesian and
spherical AO counts, detached capacities, pinned source primitives and explicit
missing/invalid input refusal. The executable can dump expanded public records
for an independent full-pack CTest comparison, without source-body fingerprints
or a Python-dependent production runtime. The initial standalone probe matched
all 54 basis/element records and 818 primitive pairs exactly. Exact-head
library/CLI and endpoint qualification are
required before a public production cutover is advertised.
