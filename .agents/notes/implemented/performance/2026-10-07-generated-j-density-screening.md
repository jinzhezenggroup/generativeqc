# Decision: Density-screen generated Coulomb J

Status: implemented
Date: 2026-10-07

## Problem

The generated full-range Coulomb value owner admitted shell quartets using only
the Schwarz product. Its streaming code explicitly special-cased
`GeneratedFockConsumer::Coulomb` to skip the density bounds already used by
Direct HF/K, and the native dddd value stream made the same exception. Large
SCF builds therefore retained J work that could not materially update the Fock
matrix at the configured screening threshold.

## Decision

The generated-J owner now retains the existing `ShellPairDensityBounds`
storage plus per-system pair-class maxima. After each public-to-Cartesian
density transform it reduces the total density with the same Direct-HF kernels
and the same `screening_tolerance`.

Generated class streams use a class-level density coarse tail and the exact
shell-quartet bound
`Schwarz(ab) * Schwarz(cd) * max(|D_ab|, |D_cd|)`. The coarse pure-J tail uses
only the bra/ket Coulomb pair classes, while exchange/combined consumers retain
their broader crossed-pair bound.

The native dddd pure-J stream uses the existing
`direct_shell_quartet_survives_screening(..., coulomb_only=true)` predicate so
generated and native classes share one screening contract. Optional/manual
topologies that do not publish density bounds retain the legacy Schwarz-only
fallback rather than dereferencing null metadata.

## Rejected alternatives

A new J-specific tolerance was rejected because it would split numerical
semantics from Direct HF. AO-level density screening inside every generated
recurrence was also rejected for this change: shell-level work avoidance is
cheaper and reuses an already validated Direct contract.

## Invariants

- The public screening tolerance and exact full-range Coulomb operator identity
  are unchanged.
- RHF and UHF J screen the total density; exchange density does not enter the
  pure-J exact predicate.
- Geometry-only fallback remains available when density-bound metadata is
  absent.
- Generated and native dddd J classes use equivalent shell-level admission.

## Evidence

Static production tests require the generated Coulomb branch to multiply the
Schwarz bound by the Coulomb shell-density maximum and require the prepared
owner to execute both density reductions. The optional-allocation fixture is
updated for the enlarged owner.

Real-device numerical and complete-endpoint performance qualification remains
required before claiming a speedup.

## Consequences

Each J build adds two small density-reduction launches and O(shell-pairs)
metadata traffic, while potentially removing much larger shell-quartet
recurrence work. The pure-J owner retains the existing three-double density
bound ABI to avoid a second screening representation.

## Revisit when

A measured endpoint shows the reduction overhead exceeds the avoided J work, or
a tighter safe Coulomb-specific bound can reduce work without increasing
metadata or launch cost.

## References

- #1892
- `src/scf/cuda/direct_coulomb.cpp`
- `python/generativeqc_compiler/integral/production_emission.py`

Agent: ChatGPT
Model: GPT-5.6 Sol
