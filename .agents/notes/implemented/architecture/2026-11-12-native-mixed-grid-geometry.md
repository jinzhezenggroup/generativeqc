# Decision: bounded native CPU mixed AO/grid geometry

Status: implemented
Date: 2026-11-12

## Problem

Composed native nuclear S'/F'/D'/W' is first response, not a Hessian. Frozen XC
curvature additionally needs independent left/right/mixed AO and partition
directions on precisely the stationary owner's retained quadrature. Replaying
the first source per left coordinate multiplies AO/grid work and does not
provide a bounded mixed source owner.

## Decision

`HostGridMixedGeometry` borrows the existing authenticated geometry reader and
an explicit immutable pre-partition measure export from the same grid. It
evaluates each tile's normalized order-three spatial AO jets once, then applies
the incumbent compiler Taylor program. Host generation specializes its exact
Taylor origin; CUDA generation remains the existing default. Scalar mixed
norm/ratio/Becke/product algebra remains in the shared Graph authority.

Partition scheduling uses a common exact binary scale for every field of each
product jet. Normalizing by the largest jet field, not just the primal, retains
zero-primal/nonzero-derivative contributions. A common scale cancels between
the partition numerator and denominator. The reconstructed primal partition is
gated against retained weights before any output publishes, in dimensionless
units so far-radial atomic measures do not amplify a harmless switching ulp.

Mandatory ceilings cover the two readers' FP64 arrays and source pair/point/AO
visits. Integer metadata, borrowed storage, stack, allocator overhead and RSS
are distinct. Each direction pair traverses center pairs once and each requested
point's pairs once. No coordinate-major AO derivative tensor or source replay
per left coordinate is retained. Both directions may alias external outputs;
all outputs stage until final owner reauthentication.

## Rejected alternatives

- Finite-difference grid/basis reconstruction in production: independent oracle
  only, not a native second source.
- A second native derivative algebra: generated host and CUDA forms reuse the
  same scientific Graph roots.
- Dividing by a retained partition weight to recover the atomic measure: exact
  zeros and underflow make that representation scientifically unsafe.
- Quietly reconstructing an entire grid-sized measure array per tile: borrow
  its explicit owner-exported storage instead, with identity and lifetime checks.
- Admitting coincident nuclei or points on nuclei by an arbitrary second-order
  convention: these non-smooth norm branches refuse before publication.

## Invariants and evidence

Versions 1--4 and all five Becke iteration counts preserve the fixed topology,
element/radial/angular atomic measure, radius adjustments and owner motion.
Cartesian/real-spherical s/p/d/f controls use independent normalized libcint
primal AO differences. Three steps and 70-digit independent direct products
qualify left/right/mixed weights without reusing production log/AD programs.
Tests also cover translation, bilinear scaling/symmetry, partial tiles, all
caps, borrowed/output aliases, staged/final owner failure, reentry and recovery,
measure mismatch, finite work at 25/40 centers, and a Python-free executable.

These are component controls. The 25/40-center geometry tests are NOT complete
25--40-atom electronic endpoint qualification, and this reader is NOT frozen XC
curvature, a stationary Hessian, installed endpoint admission or a TS optimizer.

## Revisit when

The complete native stationary Hessian owner can authenticate and retain the
atomic measures directly, or a shared bounded atomic-measure tile export removes
the need for explicitly borrowed whole-grid storage without source replay.

The source-driven whole-column successor is recorded in
[the moving-grid column decision](2026-11-12-native-moving-grid-curvature-columns.md).
It extends the source schedule without rewriting the scalar-pair qualification.
