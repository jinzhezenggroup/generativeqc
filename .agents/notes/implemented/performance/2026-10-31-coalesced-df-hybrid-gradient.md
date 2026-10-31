# Decision: contract full-range RKS J/K derivatives once for total gradients

Status: implemented
Date: 2026-10-31

## Problem

The native stationary total-gradient consumer requested separate Coulomb and
exchange DF sources. On A100, PBE0 DF96 traversed the same three-center derivative
domain twice: approximately 1.66 seconds per traversal, plus two metric
derivative contractions. The existing DF response algebra already accepts both
physical J and K cotangents in one request. Separate publication, rather than a
mathematical dependency, forced duplicate integral work.

## Decision

Allow the total-gradient consumer to request a coalesced full-range RKS source.
The existing density response sums the Coulomb coefficient and the physical
restricted exchange coefficient `-0.5 * exchange.coefficient` before contracting
three-center and metric derivatives. It may consume the existing authenticated
final occupied projection; the ordinary bounded response remains available.

Publication uses a distinct `FullRangeJkSum` kind with one energy-gradient
channel. The compiler has already proved that the native stationary consumer
reduces its named inputs with unit coefficients. Its adapter can therefore store
the sum in J's slot and zero K's slot without another allocation or a second
mathematical implementation. An explicit coalescence mask records that these
storage slots are no longer independent component observables.

Independent component APIs retain separate publication. A source-span request
overlapping either coalesced component fails before writing any output. Duplicate
J/K submission is rejected. Geometry reset clears the coalescence mask. UKS and
range-separated consumers retain separate channels because their source
contracts differ; this optimization must not fabricate those components.

## Evidence and scope

The host provider/transport suite passed 124 tests before the separate portable
Rys changes. It covers actual combined coefficients, optional projection reuse,
failure propagation and unrestricted fallback. Native transport tests additionally
exercise combined-source admission, duplicate refusal, component-span refusal
without partial output, reset and ineligible profiles. Independent PySCF tests
cover PBE0/B3LYP/M06-2X complete forces at original and moved geometries, in
addition to the retained projection-hit and constrained-memory tests.

Exploratory A100 PBE0 DF96 complete endpoints at the unchanged 256-point tile
changed from 7.156195 to 5.379580 seconds warm. Cold endpoints changed from
17.989189 to 16.179205 seconds. Both are one cold plus three warm calls, not an
isolated-kernel speedup. Raw identities, all samples, diagnostics and independent
qualification are under `.artifacts/a100-catchup-20261031/`. Later Rys and grid
batching experiments are separate arms and must not be attributed solely to
this change. Clean portable-Rys plus coalescing subsequently completed all 32
README DF cases on A100, and automatic SM120 plus coalescing completed the same
cases on RTX 5090 and PRO 6000. Each device passed 18 independent tests and
retained the same five default-grid/fitting gate failures in the README matrix.
The [combined evidence](../../../../benchmarks/results/ecnu-portable-df-20261031/README.md)
retains every timing population, numerical gate, sanitizer scope and actual
measured binary identity. This does not establish an all-functional ranking.

## Rejected alternatives and invariants

Do not return the aggregate as a Coulomb component with an apparently valid zero
exchange component: that would corrupt diagnostics. Do not add a second response
algebra or keep two derivative traversals merely to preserve private slot layout.
Preserve provider lifetime, stream ordering, finite-value checks, failure drains,
the explicit fallback and complete independent-reference acceptance gates.

## Revisit when

A consumer needs component diagnostics alongside the total gradient, or a
compiler reduction uses coefficients other than a unit sum. Such consumers need
their actual component contract, rather than reinterpretation of this source.
