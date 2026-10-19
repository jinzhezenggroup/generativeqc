# Fixed-orbital UMP2 correlation cotangents

`tools.generativeqc_mp2.unrestricted_adjoint` supplies internal CPU/NumPy FP64
MO weights for the first derivative of canonical UMP2 correlation energy.
The scientific equations remain owned by
`generativeqc_compiler.mp2.equations.unrestricted_energy_program`; derivatives
use the shared TensorIR VJP. These are **unrelaxed fixed-orbital weights**.
UHF orbital response, AO derivative contractions, relaxed density/overlap
weights and nuclear forces are outside this fixed-orbital interface. The
[complete native endpoint](#complete-native-nuclear-forces) composes these
derivative semantics with UHF response and nuclear derivative consumers.

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
Its Python coverage artifacts also retain JUnit records so molecular gate
success can be checked explicitly, including failure/skip status, at the tested head.

These checks qualify the fixed-orbital MO derivative slice. The complete CPU
endpoint has separate molecular gates below; CUDA is not admitted.


## Complete native nuclear forces

`mp2::unrestricted_force_cpu` consumes the exact native UHF reference and its
live prepared conventional J/K owner. The three canonical spin equations and
native VJPs are generated from `unrestricted_energy_program` through the shared
TensorIR transpose and bounded CPU emitter. Runtime accumulation maps exchange
bars to the physical same-spin ijab block and scatter-adds all epsilon bars.
There are no restricted occupation multipliers.

The runtime retains three ijab cotangents and rank-two spin Fock multipliers.
Finite occupied/occupied and virtual/virtual density blocks are assembled from
leg contractions of the retained physical ijab cotangents, without dividing by
same-space orbital-energy differences. A generated bilinear helper polarizes
the existing epsilon VJP through the shared TensorIR JVP; its physical-bar
normalization is evaluated from that same VJP. Every occupied and virtual leg
is accumulated separately, including repeated same-spin legs. The generated
epsilon diagonals retain their original summation; offdiagonal contributions
are stored symmetrically in the existing rank-two matrices. This keeps the
force invariant under rotations within exactly degenerate occupied or virtual
spaces and does not require an extra amplitude tensor. After this finite
assembly, same-space pairs with an energy gap at most 1e-9 still require an
antisymmetric gradient at most 1e-9. This compatibility check is separate from
the later 1e-7 full relaxed residual gate. One packed
alpha-then-beta Z-vector uses `response::spin_orbital_problem`, the actual native
`J[delta-P-alpha + delta-P-beta] - K[delta-P-spin]` provider and common native
GMRES. The physical unpreconditioned residual and the final full Lagrangian
antisymmetric residual must pass independently. Empty rotation spaces require
no solve. The relaxed overlap weights follow the symmetric coefficient gradient.

One-electron and Pulay weights are pulled back in each spin frame and summed.
The factorized ERI cotangent is pulled back in four bounded shell stages,
retaining alpha/beta coefficients in their own slots, then immediately consumed
by the existing weighted ERI derivative. Nuclear repulsion is counted once.
Neither a molecular AO rank-four cotangent nor a coordinate-major ERI derivative
array is materialized. The endpoint still retains dense ijab weights and streams
O(N^3) MO blocks; this is not an arbitrary-size or performance-qualified route.

The conservative numeric plan is checked before cotangent or response execution.
It includes the provider/reference, generated VJP and bilinear-helper workspaces,
three ijab outputs, rank-two state, four MO blocks, complete Krylov storage,
simultaneous shell staging, result storage and bounded derivative scratch. For
`C = 3 * atom_count`, checked coordinate storage reserves `C*C + 16*C` doubles:
the one-electron consumer retains C center Jets with C derivative elements,
plus simultaneous scalar values, output planes and generated-kernel wrapper
scratch. The bounded derivative route admits s/p/d/f shells; higher-angular-
momentum fallback recurrences are outside this force capacity contract. Planned and measured
memory are distinct. No force or success diagnostic publishes after a resource,
source, response, denominator or nonfinite failure.

```sh
PYTHONPATH=python:. python -m pytest tests/python/test_ump2_forces.py tests/python/test_ump2_public.py -q
ctest --test-dir build/cpu-1823 -R generativeqc_ump2_contract_tests --output-on-failure
```

Independent PySCF 2.14.0 analytic UMP2 gradients use the exact native Gaussian
shells. Their small response oracle constructs PySCF's own physical J/K matrix,
uses a bounded NumPy SVD solve (relative cutoff 1e-12, at most 128 rotations), and
requires a fresh residual below 1e-11. Compatible exact zero modes use the
least-norm solution. This avoids treating iterative Krylov termination as a
response accuracy certificate; no production coefficient/response or generated
UMP2 graph is consumed by the oracle. H2, asymmetric open-shell H3, OH, NH2 and localized stretched H2 provide total
force gates; a same-space-degenerate H6 fixture additionally checks analytic and
reconverged forces, and native alpha-only, beta-only and both-spin occupied-gauge
rotations. Li and Be retain successful forces with degenerate virtual p spaces.
Reconverged central differences use multiple steps. Acceptance is
2e-7 Hartree/Bohr against analytic/directional oracles, 1e-9 translational closure,
2e-9 restricted-limit parity, and 1e-10 alpha/beta permutation parity. Prepared
cold/warm/changed geometry, inclusive budget admission, source failure recovery,
stale owners and failed-result publication have separate controls.
