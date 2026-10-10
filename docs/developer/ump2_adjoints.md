# Fixed-orbital UMP2 correlation cotangents

`tools.generativeqc_mp2.unrestricted_adjoint` supplies internal CPU/NumPy FP64
MO weights for the first derivative of canonical UMP2 correlation energy.
The scientific equations remain owned by
`generativeqc_compiler.mp2.equations.unrestricted_energy_program`; derivatives
use the shared TensorIR VJP. These are **unrelaxed fixed-orbital weights**.
UHF orbital response, AO derivative contractions, relaxed density/overlap
weights and nuclear forces are outside this interface. The public `ump2`
method continues to support energy only.

## Tile interface

`tile_energy_adjoint(feeds, channel=..., budget_bytes=...,
denominator_threshold=...)` accepts one positive rectangular ordered ijab tile.
Channels are `alpha_alpha`, `beta_beta` and `alpha_beta`, retaining spin identity
in the equation and derivative hashes even for identical extents.

- `g[i,j,a,b] = (i_left a_left | j_right b_right)` is a direct Coulomb feed.
- Same-spin `x[i,j,a,b] = (i b | j a)` is a separately read/reordered exchange
  feed with the same tile shape. Swapping axes inside a disjoint rectangular
  direct tile cannot supply it. Alpha-beta accepts no exchange feed.
- `ei`, `ej`, `ea`, `eb` have the respective tile-axis lengths.
- Same-spin energy is `sum((g-x)^2 / D) / 4`; opposite-spin energy is
  `sum(g^2 / D)`, where `D = ei + ej - ea - eb`.

The result keeps independent `direct`, optional `exchange`, and four epsilon
feed cotangents. It has not yet accumulated shared physical sources. Arrays
own immutable storage; no input or other output array aliases them.
Empty tile dimensions fail explicitly; canonical assembly skips empty channels.

## Canonical assembly

`canonical_energy_adjoint(integrals_iajb, orbital_energies, occupied,
reference_identity=..., hamiltonian_id=..., tile_shape=(1,1,2,2),
budget_bytes=..., denominator_threshold=...)` accepts three integral arrays
keyed by channel, two epsilon vectors and two integer occupied counts keyed by
`alpha`/`beta`. Each integral array has shape
`[no_left, no_right, nv_left, nv_right]`; epsilon vectors place occupied orbitals
before virtuals. Counts can differ between spins. No ERI symmetry compression,
occupation multiplier, screening or regularization is applied.

Reference and Hamiltonian identities are required **caller attestations** about
these MO buffers. They do not prove SCF stationarity or inspect a supplied UHF
snapshot. Supply identities from the validated reference and integral source
that produced the buffers; finite perturbation tests intentionally hold orbitals
fixed. Inputs are borrowed synchronously and must remain unchanged during a call.

For same spin, assembly adds the direct bar to `[i,j,a,b]` and transposes the
exchange bar back into `[i,j,b,a]` of the same physical integral array. These
slices can overlap; both contributions must be added. For opposite spin there
is one owned alpha-beta block, with all ordered occupied and virtual tuples.
The four energy bars scatter-add into the two global spin vectors, including
repeated occurrences in same-spin denominators and shared-spin contributions
across channels. The result mappings and arrays are immutable, contain all
three channels (including correctly shaped zeros), and carry reference,
Hamiltonian, primal/derivative identities, tile count, minimum absolute
denominator, backend and admitted numeric capacity.

## Admission and failures

Inputs must already be real FP64 NumPy arrays; lists, complex and FP32 buffers
are rejected without coercion. Shapes, keys, integer extents, identities, finite
values and a finite positive denominator threshold are checked. Every active
denominator must be negative with absolute value strictly greater than the
threshold. Extremal checks inspect all spin-channel energy ranges before AD.
Empty channels have no denominator and contribute zero weights.

Numeric capacity includes borrowed inputs, dense weight arrays, immutable
publication copies and scratch, the largest full/tail tile feeds, conservative primal/reverse
workspace and a bounded finite-scan buffer. The tile workspace is four times
the shared equation's `cpu_capacity`, which covers retained primal and reverse
nodes, primitive temporaries and tile-bar copies for this fixed DAG inventory.
Capacity is a conservative numeric-byte admission contract; Python object
headers and allocator overhead are excluded. It is not a process RSS limit or
a performance claim. Canonical outputs remain dense, so tiling cannot admit an
otherwise over-budget output. Insufficient budgets fail before output allocation
or any VJP. No partial result is published on validation, AD, cumulative overflow
or publication failure. There is no scientific fallback.

## Reproducible acceptance

```sh
PYTHONPATH=python:. python -m pytest tests/python/test_ump2_adjoint.py -q
```

The NumPy-only suite uses seeded independent four-spin spin-orbital enumeration
and fixed-orbital directional perturbations, per-feed rectangular checks,
alpha/beta permutation, restricted-limit cumulative weights checked against
independent RHF energy perturbations, tile/global equivalence, sign/factor
negative controls, empty channels, pre-execution budget rejection and late
failure/no-alias checks. The pinned PySCF 2.14.0 reference environment additionally
checks actual open-shell Li and broken-symmetry stretched H2 UHF states, compares
the independent four-spin energy with PySCF UMP2 and qualifies the MO cotangents
against four-spin gradients and fixed-orbital perturbations. Those two tests
explicitly skip when PySCF is unavailable. Routine Linux reference CI installs
that pinned dependency and includes new test files in its core shards.

These checks qualify this MO derivative slice, not reconverged nuclear forces,
CUDA execution or a complete production force endpoint. Parent issue #1823
retains the response, relaxed-weight, derivative-consumer and nuclear-force
acceptance work.
