# Decision: compose a native moving-grid GGA Fock source from canonical AD

Status: implemented (optional development component, not endpoint admission)
Date: 2026-11-12

## Problem

The native metric/RHS/D/W consumer requires authenticated physical nuclear
fixed-density Fock directions. An arbitrary field-direction control does not
provide these. Retained partition-weight motion also omits moving AO density
fields and the moving AO factors in the AO Fock matrix.

## Decision

Add `HostGridGeometry` and `HostGgaNuclearXc` to the optional CPU-only native KS
response target, not the independently linkable scalar target or public core.
The geometry reader borrows the existing normalized through-f AO provider and
exact retained grid. Ordinary order-two spatial jets feed the first Taylor-AD
motion specialized at its exact expansion origin. Higher mixed-motion roots
remain unchanged. All spatial jets must be finite before that specialization.
Kinematic relative motion is grid-owner minus AO-center nuclear displacement.

Compose weighted primal and moving Cartesian GGA cotangents from unchanged
production `build_roots` physical gradients/Hessians and the existing coefficient
JVP. Generic TensorIR AO feature and potential JVPs provide the contractions.
This reuses no WB97M-V point-field flooring or nonlocal policy, even though the
generic potential JVP currently belongs to its nuclear map builder. Weight
motion enters the scalar cotangent once; the potential JVP must not weight it
again. RKS uses total D and UKS uses occupation-one spin Ds.

Move the verbatim scalar DAG importer from WB97M-V to neutral
`integral.graph_composition`. Bind simultaneously, preserve operations/payloads
and lazy branches, and validate root/replacement graph ownership and complete
bindings before modifying the target. The one-off before/after WB97M-V source
comparison is byte-identical; it is not a permanent source/body fingerprint test.

## Rejected alternatives

- Reimplementing AO/GGA response algebra in C++ creates a second scientific
  authority rather than a native consumer of the compiler equations.
- Reusing WB97M-V field policies changes B3LYP's endpoint/vacuum contract.
- Finite differences or PySCF inside production would introduce displaced
  geometries or an undeclared reference dependency.
- A full point/nuclear Jacobian needlessly grows storage with the entire grid.
- Calling this component a native Hessian or TS endpoint would omit the actual
  integral/ECP directions, second derivatives and stationary assembly.

## Invariants

Mandatory exact linked leaf/method/spin identities and owner validators;
matching basis/grid centers; stable borrowed shapes/pointers; immutable D;
bounded FP64 tile/map/matrix arenas; explicit complete-source AO-point/pair
visits and generated map-term caps. Provider/scalar stack, borrowed storage,
metadata, allocator overhead and RSS are not relabeled as owned numeric bytes.
The census does not count every provider/validation instruction or establish an
endpoint timing bound. Nonfinite/undefined derivatives fail without floors.
Outputs publish transactionally; stale owners, aliases, failed caps and reentry
must preserve outputs and permit recovery when the original owner is restored.

## Evidence

`tests/python/test_scalar_graph_composition.py` executes simultaneous binding,
shared subexpressions, lazy branches and ownership/missing-binding refusals.
`tests/python/test_host_xc_nuclear.py` strictly compiles the current grid and
new native consumers with verified compiler caching. Independent normalized
libcint spherical s/p/d/f displaced-basis/point jets check AO motion at three
steps. Independent AO/Libxc full frozen-Fock differences cover both spin
layouts, grid versions 1--4 including native default, three steps, translation,
tile parity, exact vacuum behavior, capacity and failure recovery.
The linked CMake executable executes both spin layouts with Python absent from
PATH and PYTHONPATH, checking a nonzero source, translation and exact work census.

The source-matched local qualification directory is
`build/qualification/host-xc-nuclear-20261112/`. Its final receipt distinguishes
directly compiled current grid/consumers from the explicitly older CPU core
borrowed for the unchanged AO provider. It does not supersede previous accepted
source-cohort receipts or claim current-source installed acceptance.

## Consequences and revisit conditions

This removes a concrete missing semilocal nuclear source, not the remaining
native Hessian/CLI or scientific endpoint gates. Validation requires borrowed
immutability/epoch authentication even when pointers and shapes remain stable.
The materializing serial contraction schedule favors simple bounded correctness
over performance. Revisit its schedule only with complete-source and endpoint
work/timing evidence, retaining exact physics, branch, failure and lifetime gates.
