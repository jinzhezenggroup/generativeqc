# Decision: Compose parameterized D4 from MethodIR and the pinned registry

Status: implemented
Date: 2026-09-28

## Problem

VibeQC already generated all 118 pinned `d4.bj-eeq-atm` parameter sets, but
MethodIR exposed D4 as a hand-written PBE special case plus r2SCAN-3c. Adding a
new named method row for every valid functional-D4 pair would duplicate both the
electronic catalog and the generated dispersion registry.

## Decision

A D4 composite is now derived only when both facts already exist: the electronic
method resolves through MethodIR and the pinned D4 registry has an exact
functional-specific parameter set. The compiler builds the composite
declaratively. Public short selectors such as `pbe0-d4-rks` are advertised only
when the electronic projection passes the existing native lowerer gates.

Except for the retained `pbe-d4-rks` compatibility endpoint, execution keeps
the electronic carrier and D4 owner separate. The prepared batch runs the
electronic MethodIR once, runs `D4CorrectionBatch` once, and composes the
correction energy exactly once.

## Rejected alternatives

- Copy all 118 D4 rows into `METHOD_CATALOG`. This creates a second parameter
  whitelist and will drift from the generated upstream registry.
- Treat every XC functional as D4-compatible with shared damping parameters.
  D4 damping is functional-specific; absence of a pinned fit must remain an
  error.
- Generalize r2SCAN-3c's D4+gCP path to all D4 methods. Plain D4 corrections do
  not own gCP or a basis binding.

## Invariants

- Parameter values and provenance remain generated from the pinned dftd4 source.
- A D4 suffix never invents damping parameters.
- Unsupported electronic lowerers remain fail-closed even when a D4 fit exists.
- r2SCAN-3c keeps its canonical basis, gCP and special D4 profile.
- The legacy PBE-D4 endpoint must not double-count the correction.

## Evidence

- Compiler tests compare composed PBE0, B3LYP and r2SCAN electronic primitives
  against their base MethodIR and the generated D4 specs.
- Public discovery tests require lowerer-gated D4 selectors and reject SCAN-D4
  until its electronic lowerer is qualified.
- CPU energy tests compare PBE0-D4 and B3LYP-D4 against independent
  `E_electronic + E_D4` execution through the production D4 owner.

## Consequences

Adding an electronic method or a D4 fit no longer requires a hand-written
cross-product entry. A new public D4 selector appears only after both the
scientific parameter record and electronic execution capability exist.

## Revisit when

Revisit the retained named PBE-D4 native carrier when correction ownership is
fully method-name-free, or when additional D4 variants such as MBD are promoted.

## References

- `tools/sync_dispersion_parameters.py`
- `tests/python/test_dispersion_parameter_sync.py`
- `python/vibeqc_compiler/method/spec.py`
- `python/vibeqc/dispersion.py`
