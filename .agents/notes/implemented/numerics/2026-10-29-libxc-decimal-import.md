# Decision: retain Maple decimal literals until target lowering

Status: implemented
Date: 2026-10-29

## Problem

The importer converted an AST floating literal back to a Fraction using its
already-rounded Python float spelling. `1.0000000000000001 - 1` consequently
became zero before differentiation; `1e-400 / 1e-399` lost both operands even
though its ratio is representable. Pinned PW92 input also contains a decimal
coefficient with more precision than Python binary64. These observations do not
establish a molecular error outside existing acceptance gates.

Updating importer semantics exposed a regeneration cycle: importing the XC
package eagerly loaded FunctionalSpec, which required the very catalog that the
offline import tool needed to regenerate.

## Decision

Parse numeric source spans into exact Fraction leaves in the private syntax tree.
Existing Graph arithmetic performs constant folding and differentiation; the
ordinary emitters own target rounding. Preserve exponent, branch and work-domain
policies. Reject Python-only numeric spellings and boolean constants explicitly.
Advance the importer identity to v12 and regenerate its catalog/provenance.

Resolve the facade's catalog-backed types lazily through their canonical module.
The importer can then bootstrap without a current catalog, while public method
resolution retains the existing missing/incompatible-catalog errors. Do not
accept stale catalog identities or manually relabel generated products.

## Evidence

Regression cases compare values and two derivatives with independent rational
answers, through both Graph evaluation and compiled FP64 code, using relative
gates with no absolute floor. Cases include cancellation, extreme exponents,
large integer-valued decimals, bounded sums, lists and external binding parity.
A subprocess removes access to the generated catalog and verifies importer
bootstrap, public rejection, and canonical facade identity after restoration.
The existing independently referenced Libxc E/vxc/fxc suites remain required.

## Consequences

Existing catalog membership and public scientific admission are unchanged.
Generated provenance intentionally changes; old compiled receipts cannot be
relabelled as evidence for the new importer. This change does not claim a CUDA
endpoint speedup or complete unification of automatic and curated execution.
