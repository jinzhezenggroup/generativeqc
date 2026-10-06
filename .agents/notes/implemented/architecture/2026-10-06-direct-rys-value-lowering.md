# Decision: give Direct Rys values a derivative-free lowering

Status: implemented
Date: 2026-10-06

## Problem

The existing component-lane Rys Fock worker reused the preceding force emitter's
root tables, primitive geometry type and HRR decoder. CUDA capabilities rejected
Rys IR without FORCE, and the production selector interpreted every two-root
Rys request as a scalar force schedule. A value-only Direct Fock request could
therefore neither own its lowering nor use its own quadrature order.

This matters for odd angular order: ppps values require two roots, while ppps
first derivatives require three; dddp values require four rather than the five
roots of its first derivative.

## Decision

Accept derivative-free DIRECT_FOCK Rys IR within the implemented component
decoder domain. Derive root count from IntegralIR and addressed TRR bounds from
the two shell-pair angular sums, without a raised derivative state. Emit strict
root tables from the existing compiler table owner, then fuse values through
the existing canonical RHF/UHF scatter and ordinary streaming artifact ABI.

Enumerate and validate only the implemented one-component-per-lane mapping for
these value candidates. Reject an unsupported root, decoder or launch geometry
instead of silently emitting Cartesian recurrence under a Rys artifact identity.
Share the HRR source owner with the existing force emitter. Legacy force/Fock
artifacts retain their previous root policy and emitted source bytes.

This is source capability, not a production promotion. The checked-in production
manifest and native plan selection are unchanged. J and K continue to own their
own prepared consumers; sharing a value equation or force output layout does
not require them to select the same lowering or quartet traversal.

## Invariants

- Full-range Coulomb mathematics, primitive-pair inputs, AO normalization,
  canonical pair orientation, screening and RHF/UHF scatter remain unchanged.
- A derivative-free Rys value artifact defines no force task or force launch
  wrapper. Existing Cartesian value-only compatibility artifacts are retained.
- The current runtime-indexed decoder covers s/p/d centers with at most p on
  the fourth center. A CTA must fit every Cartesian component and the target.
- Generic Direct remains the exact fallback. No native recurrence copy, method
  allowlist or molecule/class promotion table is introduced.

## Rejected alternatives

Adding a dormant FORCE consumer would keep the unwanted derivative root count
and prevent independent value planning. Globally promoting Rys before separate
J/K shell-class and endpoint measurements would conflate source capability with
profitability. Extending dddd first would assume it dominates K without the
profile requested by #2015.

## Evidence and qualification boundary

The companion tests exercise value-only and streaming artifact emission, exact
root/TRR bounds, unsupported-schedule rejection and deterministic source. The
numerical gate compiles the emitted strict root/TRR/HRR arithmetic with ccache
and contracts signed unequal primitive lists against independent Libcint for
ppps/dpss/ddss/pppp/ddpp/dddp, including coincident and reversed-pair records.
These classes sample the decoder domain; they are not a production allowlist.

The standalone CUDA gate also launches the emitted queue workers through their
ordinary C ABI for RHF/UHF and Combined/J-only/K-only matrices. Its independent
Libcint dense contraction includes signed unequal primitive lists, coincident
centers, reversed cache records, empty worker queues and unused component lanes.
It does not route production through a CPU/reference consumer.

Initial host qualification on node1 passed 158 checks (two separately enabled
NVCC checks skipped); compiler structure checked 465 modules with zero dependency
errors. Five existing fused force/Fock class artifacts retained byte-identical
CUDA source across the shared HRR extraction.

Production promotion still requires RHF/UHF CUDA fixed-density matrices,
resource/spill/occupancy gates, separate J/K class work and timing, and complete
48/96-atom energy/analytic-force endpoints, including changed geometry.

## References

- #2015: production compiler-selected Rys lowering for exact J/K.
- #1892: primitive-pair materialization and shared Direct work.
- #2007 and #2011/#2012/#2014/#2016: common force execution foundation.
