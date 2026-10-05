# Decision: explicit CUDA vendor ownership and frozen migration references

Status: implemented
Date: 2026-10-05

## Problem and decision

#1886 requires method owners to request semantic tensor/solver work. A broad
vendor-name grep could neither distinguish existing provider implementations
from method coupling nor inspect the code emitted by Python generators. Freeze
the current production reference inventory with explicit per-file rationale,
contract and classification, checked by `tools/check_vendor_boundaries.py`.

Retain the existing DF SCF matrix/eigensolver primitive adapter. Its dimensions,
strides, alpha/beta and eigensystem interface are reusable, and its DeviceSolver
already owns workspace. Surrounding SCF policies/selectors still need canonical
binding integration. Do not classify `matrix_library.cpp` as approved merely
because its name sounds generic: it still accepts `use_cublas` from above.

## Invariants and rejected alternatives

Directory-wide exemptions would allow scientific coupling inside a provider
directory. File-only exemptions would permit new SGEMM calls in an existing
DGEMM consumer. Instead require explicit files and symbols, with exact counts
for legacy migration debt and narrow diagnostic/infrastructure exceptions.
Retired references require inventory cleanup so old allowances cannot silently
authorize their reintroduction. Provider/ABI owners may change call multiplicity.

Inspect Python literal fragments and f-string constants without importing or
executing generators. Reuse the existing native comment/literal lexer rather
than implementing a conflicting parser. Synthetic mutation tests cover added
files, added symbols, duplicated calls, addresses, macros, namespaces, emitted
calls and false positives in comments/literals.

## Limits and revisit conditions

This is a lexical ownership guard, not an actual-path proof. Dynamic API-name
construction, arbitrary external templates and replacing one call with another
of the same symbol/count still require review. Binding provenance and independent
scientific gates remain necessary. Selector removal and the inventoried #1890
migrations are unfinished; this inventory explicitly records that debt rather
than declaring the architecture migration complete.
