# Decision: bind materialization rules to consumers and distribute only private sums

Status: implemented
Date: 2026-10-20

## Problem

#2634 supplies a legal single-producer substitution pass, but an opaque-source
recommendation is only text and visible sums still form a high-rank intermediate.
#1580 needs a shared rule-request boundary and further exact contraction rules,
without reopening completed MP2/CC production-streaming work.

## Decision

Use one common immutable JSON snapshot for analyzer-owned rule requests. Bind
source/producer/consumer identity and specs, operand positions, precision policy,
output retention, byte/work/live limits and search bounds. Source analyzers and
TensorIR can carry the same request; request metadata is never a proof or an
execution permission. Keep unknown alias, derivative, consumer and provider
obligations explicit.

An additional finite compiler rule distributes one einsum over a bounded sum of
visible private einsums. Inline each term using existing exact label substitution
and binary tree search; recombine its rational coefficient through the existing
add primitive. A term with another reader or a public output is not inlined.
Require explicit floating-point reassociation permission, FP64, and the existing
whole-program nonincreasing work/live-peak and strict materialization savings.

## Rejected alternatives

- Trust arbitrary factor/provider claims for opaque four-index inputs: no
  separability, derivative or consumer-layout proof is available.
- Always distribute to save memory: the actual RCCSD 2 occupied/4 virtual ladder
  offers less storage but more estimated work (4992 versus 4928), and is rejected.
  This losing shape remains an executable acceptance case.
- Reimplement MP2/CC force drivers: #1574/#1575 already own their completed
  representation/streaming work. Reuse equations as compiler acceptance workloads.
- Duplicate contraction search, scientific derivative algebra or lifetime analysis.

Budget validation imports the canonical resource checker only when a request
actually supplies a byte limit. A top-level dependency breaks the intentionally
isolated stdlib-only policy reload fixture; preserve that existing integrity
test rather than broaden its copied source closure.

## Invariants

Ordinary preparation and production dispatch remain unchanged. Retain original
programs for public dense outputs, shared/repeated terms, opaque/nonlinear
boundaries, strict arithmetic schedules, unsupported rules or resource failure.
Input aliasing grants no savings. Descriptor/JSON serialization grants no
scientific capability. Do not claim reduced formal CC scaling or GPU speedup.

## Evidence

Tests use the MP2 Fock-like weight formula documented by
`FactorizedTwoElectronWeights` and the actual shared RCCSD D05 ladder source.
Independent literal/NumPy arithmetic, analytic directional derivatives,
generated/interpreted JVP/VJP and central differences qualify changed equations.
MP2 value/derivative tolerances are 2e-13/3e-13; CC component gates use 3e-12,
with finite differences at 1e-5 and 3e-6 under 3e-8. Rejected small CC and selected
larger virtual-space cases are both retained; these are compiler estimates and
component arithmetic, not complete force endpoint measurements.

## Consequences and revisit conditions

This is another bounded #1580 slice. Opaque source rules require an explicit,
independently qualified factor/block-source adapter before execution. Future
provider integrations should consume this request rather than introduce another
analyzer, and must qualify actual endpoint memory, work and numerical behavior.
Wider nonlinear/precision/sum support requires additional proofs and acceptance.

Agent: Codex
Model: GPT-6
