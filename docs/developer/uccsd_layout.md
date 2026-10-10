---
orphan: true
---

# UCCSD amplitude layout and coordinates

`generativeqc_compiler.cc.uccsd_layout` supplies an internal, real-FP64
amplitude storage contract. It provides five spin-labelled TensorIR descriptors,
exact coordinates, and CPU pack/unpack conversion. It does not register a public
method or implement energy, residuals, denominators, DIIS, iteration, molecular
convergence, or UHF reference admission. Layout validation alone is not UCCSD
solver qualification.

## Logical and packed axes

`UCCSDLayout(oa, ob, va, vb)` takes independent alpha/beta occupied/virtual
populations. Dimensions must be Python `int` (not bool or NumPy integer),
nonnegative, within the signed-int64 domain. Zero populations, singleton spaces,
and either spin population absent are valid. Every logical/packed layout and the
total full/packed FP64 byte counts must fit the existing portable int64 size
contract; DenseLayout also checks strides. Overflow is rejected before allocation.
These are representability checks, not an available-memory guarantee.

| Block | Full C-order shape | Index spins | Packed shape | Stored count |
| --- | --- | --- | --- | --- |
| t1a | `(oa,va)` | alpha,alpha | `(oa,va)` | `oa*va` |
| t1b | `(ob,vb)` | beta,beta | `(ob,vb)` | `ob*vb` |
| t2aa | `(oa,oa,va,va)` | alpha,alpha,alpha,alpha | `(C(oa,2),C(va,2))` | `C(oa,2)*C(va,2)` |
| t2ab | `(oa,ob,va,vb)` | alpha,beta,alpha,beta | `(oa,ob,va,vb)` | `oa*ob*va*vb` |
| t2bb | `(ob,ob,vb,vb)` | beta,beta,beta,beta | `(C(ob,2),C(vb,2))` | `C(ob,2)*C(vb,2)` |

Singles have logical axes `ia`; doubles have `ijab`. The virtual axis `b` is
fastest. TensorSpec uses `representation="spin_orbital"`, `dtype="float64"`,
`role="input"`, and explicit IndexSpace identities `oa,ob,va,vb` with spin labels.
Same-spin descriptors declare `(1,0,2,3),-1` and `(0,1,3,2),-1`. Mixed-spin
descriptors declare no exchange symmetry, even when dimensions are equal.
Descriptors own no native storage, equation inventory, or allocation lifetime.

## Canonical representatives and exact conversion

For same-spin doubles, representatives satisfy **`i<j` and `a<b`**. Occupied
pairs are lexicographically ordered; within each occupied pair virtual pairs are
lexicographically ordered. The pair rank is
`P_n(i,j) = i*(2*n-i-1)//2 + j-i-1`. The local slot is
`P_o(i,j)*C(v,2)+P_v(a,b)`. The inverse uses integer arithmetic.

`block.coordinate((i,j,a,b))` returns `(local_slot, sign)` after sorting each
same-spin pair independently. Swapping either occupied or virtual indices changes
the sign; swapping both preserves it. Repeated `i=j` or `a=b` returns `(None,0)`
and denotes strict numerical zero. All coordinates are checked for arity, Python
integer type and range before this zero rule. `block.representative(slot)` returns
the canonical logical indices of a checked slot. For singles and mixed-spin
doubles the mapping is plain C-order flattening, sign `+1`, with no sorting.

`layout.blocks` returns descriptors in fixed `t1a,t1b,t2aa,t2ab,t2bb` order.
`layout.block_slices` supplies cumulative offsets in one packed vector;
`layout.packed_size` supplies its length. Empty blocks have empty slices.
`block.pack(full)`/`block.unpack(vector)` convert one block using **one-dimensional**
packed vectors, even though `packed_layout` describes their multidimensional
physical interpretation. `layout.pack(mapping)` requires exactly the five named
blocks. `layout.unpack(vector)` returns a dictionary of fresh full arrays.

All conversion inputs must be base NumPy ndarrays of native real-FP64 dtype, exact
shape, and finite values. Arbitrary array strides are accepted and interpreted in
logical C order. Array subclasses (including masked arrays) are rejected because
they can override finiteness/equality. No input is modified or aliased by returned storage. Same-spin
pack checks both antisymmetries by exact numerical equality and explicitly checks
zero repeated-index entries. It rejects even a one-ULP violation; it never
averages or projects inputs. Unpack scatters each raw representative with the
four signs `+,-,-,+` and fills repeated-index entries with zero. No tolerance or
sqrt/multiplicity rescaling is applied. Signed zero denotes numerical zero;
its sign bit on repeated-index entries is not a storage invariant. This explicit
CPU conversion materializes the full array; it is not a bounded native solver path.

## Metrics and excitation normalization

Let `E_ai = a†_a a_i` and `D_ijab = a†_a a†_b a_j a_i` with each axis carrying
the block's declared spin. The block expansion is

```text
T1 = sum_ia t1a_ia E_ai(alpha) + sum_ia t1b_ia E_ai(beta)
T2 = (1/4) sum_ijab t2aa_ijab D_ijab(alpha,alpha)
     + sum_ijab t2ab_ijab D_ijab(alpha,beta)
     + (1/4) sum_ijab t2bb_ijab D_ijab(beta,beta)
```

`operator_prefactor` records these exact rational full-block coefficients.
In packed same-spin sums over `i<j,a<b`, every representative multiplies its
ordered `D_ijab` with coefficient **one**. Mixed-spin coefficients are also one.
In a complete spin-orbital tensor, the alpha-beta sector has four signed
occupied/virtual permutations in distinct sectors, and the global T2 sum has
prefactor `1/4`. Those other mixed sectors are not additional stored blocks.

For the Frobenius inner product of the **five full block arrays**,
`full_metric_weight` is 4 for t2aa/t2bb and 1 for singles/t2ab. Thus the full-block
metric of packed vectors is the sum of weighted per-block dots. It is different
from an unweighted packed dot. For the complete spin-orbital doubles tensor,
t2ab also has multiplicity 4. With a normalized reference determinant the squared
norm of `T1|Phi> + T2|Phi>` is the unweighted packed squared norm; this operator
norm must not be confused with either full-tensor Frobenius convention.

## Restricted embedding and independent validation

For a common alpha/beta orbital basis and pair-symmetric restricted spatial
doubles `R_ijab = R_jiba`, the amplitude embedding is
`t1a=t1b=R1`, `t2ab=R2`, `t2aa=t2bb=R2-R2.swapaxes(2,3)`.
The corresponding spin-free operator is
`T2=(1/2) sum_ijab R2_ijab E_ai E_bj`, where each E sums alpha and beta spins.
The embedding is a layout/operator identity, not a molecular solver claim.

Run from a checkout with compiler dependencies, NumPy and pytest:

```sh
PYTHONPATH=python python -m pytest tests/python/test_uccsd_layout.py -q
python tools/check_compiler_structure.py
```

The tests independently build spin-orbital tensors, then compare full `1/4`
and packed excitation operators on the entire fixed-particle determinant space
using fermionic creation/annihilation bit actions. Unequal spin populations,
empty/singleton/one-spin cases, restricted spin-free embedding, exact pair order,
large integer inverse coordinates and admission failures are covered. Deliberately
wrong signs and `1/4` packed normalization are required to fail the same oracle.
No PySCF or native library is imported, and no device test is substituted or skipped.
