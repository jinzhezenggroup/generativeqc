# Decision: qualify complete UCCSD(T) through one shared tiled owner

Status: implemented
Date: 2026-10-12

## Problem

The shared UCCSD solver and four unrestricted triples TensorIR channels supplied
the scientific components, but fixed-amplitude tiles were not a complete
UCCSD(T) energy endpoint. State identity, fresh physical acceptance, whole-grid
coverage, complete numeric/work admission and failure publication still needed
one owner. Reusing only the restricted triples facade would import the wrong
spin/permutation/normalization contract.

## Decision

`PreparedUCCSDT` borrows one live `PreparedUCCSD` and its selected native executor,
and owns immutable copies of accepted amplitudes. Parent reference, equations,
feeds, options and executor bindings remain configuration-stable. The provider
and AO source can close after UCCSD preparation; no new integral requests or
reference-oracle calls belong to triples execution.

Admission plans full/edge shapes and exact multiplicities before execution.
Each template is identified by the graph **and its canonical operand bindings**;
physical orbital offsets do not enter that identity. Deterministic largest-axis
shrinking supplies a bounded-memory/per-call fallback, while complete tile and
semantic-work caps remain independent. Work includes fresh parent replay and
all triples tiles. CPU code generation is a pure metadata work model on CUDA,
not a CPU equation fallback. Native CUDA arenas close between executions, so
artifact reuse does not imply simultaneous device-memory retention.

Each energy call replays the parent physical energy and five full residual
arrays on the selected backend. It then gathers bounded inputs and accumulates
all ordered channel coordinates with compensated host FP64 sums. No partial
channel or corrected scientific energy survives a late triples failure.

## Non-obvious numerical gates

Repeated same-spin coordinates have zero exact moments but still execute the
graph's division. Guard them as well as distinct coordinates: a tiny denominator
can amplify antisymmetry cancellation roundoff. Also guard every occupied-then-
virtual prefix against overflow.

The denominator extrema use that same FP64 addition/subtraction order. Computing
three gaps first and multiplying by channel population is not equivalent at large
energy offsets. With occupied energies `1e16` and virtual energies `1e16+2`, the
exact gap sum is `-6` while the executed ordered FP64 denominator is `-4`. The
configured threshold must protect the executed division. Extrema scan entire
segments because snapshot admission allows small energy-ordering deviations;
endpoint indexing alone assumes a stronger ordering than the reference contract.

Strict unbroken spin-zero oracle fixtures use a tightly converged RHF reference
and explicitly convert it to UHF before constructing a fresh PySCF UCCSD object.
Asymmetric H6 UHF DIIS/Newton stalled at gradient norms around `3e-9` despite an
unchanged energy; the identical-geometry RHF converged below `1e-10` (measured
`2.84e-11`). This preserves the collapsed-spin fixture and the stronger gate
instead of weakening snapshot/SCF validation. Open-shell and broken-symmetry
fixtures remain genuine UHF. Rebuild the CC object after SCF polishing: retaining
an earlier object can retain stale MO coefficients despite refining its SCF.

## Rejected alternatives

- RCCSD(T) projection/prefactors: invalid for unrestricted sectors.
- Trusting `converged` or identity alone: a forged finite stationary state still
  requires a fresh energy and physical residual replay.
- CPU triples execution behind CUDA: breaks explicit backend provenance.
- Grid materialization or unbounded fallback: bounds memory but not necessarily
  semantic work or host scheduling storage.
- Distinct-index-only denominator admission or gap-first arithmetic: fails to
  protect the actual executed ordered FP64 divisions.
- Publishing partial energies after failures: tile diagnostics are not a
  scientifically complete correction.

## Evidence and consequences

`tests/python/test_uccsd_t_owner.py` compares converged per-channel and complete
energies with independent PySCF for Li/OH/NH2/broken H4/asymmetric H6, checks the
complete RCCSD(T) restricted limit, and preserves nonzero molecular energy under
numeric-budget fallback. Empty-domain, overflow/sum-order, complete-cap,
foreign/forged-state, immutability/lifetime and late-failure tests are separate
gates. `test_uccsd_native.py` composes native UHF and streamed native integrals
with the complete energy owner against the exact-shell independent oracle.
The CUDA opt-in tier covers NH2, broken H4 and asymmetric H6 on the same graph.

Host staging, conventional four-index feeds and ordered-grid algorithmic work
remain explicit limitations. No throughput, resident-device, DF, Lambda or
analytic-force promotion follows from this owner. Timings cover triples owner
preparation/replay/execution, not an earlier HF or CCSD solve.

## Revisit when

Introduce a different traversal or resident handoff only with independently
qualified all-channel energies, exact complete-work/resource accounting, bounded
fallback and identity/lifetime contracts. UCCSD(T) forces must scatter shared
tile cotangents into the UCCSD Lambda/Hamiltonian and shared UHF response owners;
fixed-amplitude differentiation alone does not supply relaxed nuclear forces.

## References

- Issues #1821, #1822, #1824 and #1825.
- `docs/developer/uccsd_t_owner.md` and `docs/developer/uccsd_triples.md`.
- `2026-10-12-uccsd-shared-iteration-provider.md`.
- `2026-10-12-uccsd-triples-offset-independent-tiles.md`.
