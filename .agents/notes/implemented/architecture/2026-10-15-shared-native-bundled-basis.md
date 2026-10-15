# Decision: one compiled bundled-basis expansion owner

Status: implemented
Date: 2026-10-15

## Problem

The installed CLI already expanded the generated bundled BSE subset in C++, but
Python execution rebuilt contractions from its independent JSON snapshot before
calling the descriptor constructor. Moving metadata wholesale into native code
would break native-free validation, immutable snapshots and provenance. A name
alone cannot identify caller-supplied records that intentionally reuse a bundled
name, and matching library/Python package versions do not prove catalog equality.

## Decision

Move the CLI's compiled-table expansion into `molecule/basis_catalog`, generated
unconditionally with the native library rather than only when the CLI is built.
Both the CLI and additive public C/C++ bundled-system constructors use that owner.
The constructors delegate normalization and backend/scientific admission to the
existing descriptor constructor; they introduce no integral equations.

Python keeps native-free immutable records and recognizes a bundle only by full
record identity, including decimal data, contraction order, representation and
provenance. It passes the source-pack canonical-JSON checksum to native creation;
a compiled-pack mismatch rejects before publishing a system and never retries
portable expansion. The optional C checksum check allows SDK-only callers to use
the installed catalog without shipping a second copy of the source pack.

## Retained boundaries

- Older libraries lacking the new constructor retain explicit descriptor creation.
- Custom and ECP records retain descriptor adapters, as does the correlation-only
  auxiliary-through-g domain whose admission must not be widened to generic DF.
- Portable shell expansion remains available for data-only metadata/planning and
  independent oracles. Compiler MethodIR/model resolution is not retired here.
- Raw expansion preserves coefficients, zeros, atom/source order and representation;
  only the established system validator performs normalization.
- No runtime file/network lookup, compiler, Python interpreter or CPU reference
  science is introduced into native production execution.

## Evidence and qualification

The executable catalog probe compares all 54 bundled name/element inventories
against the independent JSON pack, including every FP64 primitive. Public ABI
tests cover aliases, checksum/ABI/representation/missing-element rejection,
caller-buffer ownership and exact normalization against a separate H STO-3G table.
The public-header-only SDK endpoint retains a prepared calculation after its
named system and atom buffers die, checks H2 energy/analytic force and replays
energy-only. The catalog regenerates with CLI building disabled, and the installed
header/library SDK endpoint runs from an empty directory with no interpreter or
compiler on PATH. Python compares Cartesian/spherical cold/warm/moved ragged lifecycles
with the explicit old-symbol path and checks native rejection without retry.
Independent PySCF cases are opt-in to an environment with that oracle installed.
Host tests do not qualify a new CUDA method or physical allocation receipt.

## Rejected alternatives and revisit conditions

Name-only recognition silently substitutes same-name custom models. Unconditional
native loading breaks offline snapshots. Retrying rejected native provenance with
Python data hides installation inconsistencies. None is acceptable. Revisit the
retained custom/MethodIR adapters only when a shared native owner preserves their
exact identities, scientific admission and independent endpoint acceptance gates.

References: #2227 R5/R6, #926, #558.
