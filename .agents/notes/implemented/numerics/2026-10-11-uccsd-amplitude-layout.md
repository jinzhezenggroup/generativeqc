# Decision: exact five-block UCCSD amplitude coordinates

Status: implemented
Date: 2026-10-11

## Problem

The restricted coordinate involution exchanges occupied and virtual pairs
together. UCCSD same-spin doubles instead require two independent negative
exchanges, while mixed-spin doubles have independent alpha/beta domains.
Reusing restricted pair packing would admit the wrong coordinate quotient.

## Decision

Use a separate compiler-owned `cc/uccsd_layout.py` with derived descriptors and
raw representative coordinates. Reuse TensorIR IndexSpace, Symmetry and TensorSpec,
plus common DenseLayout and checked byte contracts. Store same-spin `i<j,a<b`
representatives and every mixed-spin `ijab` entry. Keep CPU exact conversion
explicit and separate from production method registration and scientific equations.
Zero/one-spin domains remain representable without implying reference admission.

## Rejected alternatives

- Restricted pair packing: wrong symmetry and mixed-spin identity.
- Silent antisymmetry projection: changes arbitrary input and hides invalid
  scientific state; a future rounding policy needs separate, explicit admission.
- Metric-normalized representatives: obscures excitation coefficients and makes
  future residual consumers depend on an implicit sqrt/multiplicity convention.
- Generated native kernels for this metadata slice: adds a runtime and allocation
  architecture before there is an admitted scientific consumer.

## Invariants

The current contract is documented in
[`docs/developer/uccsd_layout.md`](../../../../docs/developer/uccsd_layout.md).
Maintain exact index/spin identity, independent exchange signs, numerical-zero
diagonals, finite real-FP64 admission, unscaled coefficients, and distinct
five-block/full-spin-orbital Frobenius versus determinant excitation metrics.
Pack must not silently average or project. Empty inputs still undergo dtype checks.
Integer inverse coordinates must not rely on FP64 square roots.

## Evidence

`tests/python/test_uccsd_layout.py` builds independent full spin-orbital tensors
and acts with creation/annihilation operators on complete small fixed-particle
determinant spaces. It compares packed coefficients against full `1/4` sums and
requires wrong sign/normalization controls to fail. Restricted embedding is
checked against the independent spin-free `(1/2) R2 E E` operator. Admission,
counts, order, metric multiplicity and large integer coordinates are also tested.
No GPU, native compilation, molecular energy, or iterative solver evidence is
claimed by these CPU layout gates.

## Consequences and revisit conditions

Consumers can use logical TensorSpec and full DenseLayout directly, or deliberately
use packed_layout, coordinate/representative and block_slices. CPU conversions
materialize arrays and are not production memory/work bounds. A later residual,
DIIS or native consumer must choose its metric explicitly and independently
qualify new equations, reference domains and resources. Revisit tolerance,
native lowering or scaled coordinates only with that concrete consumer and
independent numerical evidence.

## References

- [Parent issue #1821](https://github.com/jinzhezenggroup/generativeqc/issues/1821)
- `PYTHONPATH=python python -m pytest tests/python/test_uccsd_layout.py -q`
- `python tools/check_compiler_structure.py`
