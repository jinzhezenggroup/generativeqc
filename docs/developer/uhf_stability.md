# Bounded real internal UHF stability diagnostic

`tools.generativeqc_response.stability.diagnose_uhf_stability` is an internal
CPU/NumPy diagnostic for a tiny **complete real occupied–virtual domain** of a
converged, real FP64 canonical UHF reference. It consumes the existing shared
`UHFResponseOperator`; it does not implement another SCF or Hessian equation.
It is not an installed/public method or a production fallback. Dense AO ERIs
and pinned PySCF are test oracles only.

## Coordinates and physical curvature

The existing spin layout packs alpha followed by beta, with occupied-major,
virtual-minor ordering within each spin. For each spin define `K_ia=x_ia` and
`K_ai=-x_ia`, with all other entries zero, and rotate `C(t)=C exp(-t K)`.
Then `dP/dt=C sym_OV(x) C.T`. The real orbital-energy gradient is `2 F_ai`.
At the stationary reference its derivative is `2 A x`, where the shared response
Jacobian is

```text
(A x)_ia = (epsilon_a - epsilon_i) x_ia + (C.T delta_F C)_ai.
```

Thus the **physical energy Hessian is H=2*A**, not A, and
`E(t)=E(0)+t² x.T H x/2+O(t³)` at a stationary reference. A direction has unit
Euclidean norm in the packed OV coordinates; its alpha and beta norms sum in
quadrature to one. No extra sqrt(2) or RHF spatial-occupation factor is used.
The sign convention matters when a later recovery consumer builds a generator.
Occupied/occupied and virtual/virtual rotations are excluded gauge directions;
degeneracies within those spaces need no inverse orbital gaps. Alpha and beta
spaces may have different sizes or empty blocks.

## Admission and qualification

`StabilityOptions` defaults to dimension at most 128, at most 257 evaluations,
and 64 MiB of accounted numeric host storage. Admission happens before provider
validation, operator calls, and diagnostic array allocation. For dimension n,
the complete request needs exactly **2*n response applications plus one reference
Fock evaluation**. The latter uses the shared spin Fock seam to verify that the
snapshot's Focks actually belong to its hcore and current provider. Both the
reported SCF residual and freshly checked canonical residual must satisfy the
reference tolerance (default 1e-8, maximum permitted 1e-7).

Accounted bytes include the snapshot, declared backend host workspace, shared AO
action storage, diagnostic matrices and conservative eigensolver scratch:
`reference.numeric_bytes + backend.host_workspace_bytes + 8*(12*n²+20*n+16*nbf²)`.
This is a pre-admission contract for numeric storage, not a process RSS cap:
Python/source-reading overhead and vendor LAPACK allocator behavior are outside
it. A provider without an explicit nonnegative integer workspace declaration is
unqualified. The caller remains responsible for any separately retained native
or device resources. The generic backend uses three J/K calls per evaluation;
a spin provider uses one spin-J/K call. No endpoint speedup is claimed.

The diagnostic builds every column through `apply`, checks the Frobenius
self-adjointness defect, diagonalizes the symmetric part, then independently
reapplies the operator to **every eigenvector**. It never relies on a transpose
alias as evidence of self-adjointness. The aggregate fresh-action residual is a
Frobenius bound; the classification uncertainty includes that bound, half the
symmetry defect, and a reference curvature bound. For each spin, the latter is
`4*norm(C.T F_actual C - diag(epsilon), 'fro')`, maximized over spins: this bounds
the error from replacing the actual occupied/virtual Fock blocks with diagonal
orbital energies, including accepted canonical residuals in a poorly conditioned
AO basis. Symmetry and residual defaults are absolute 1e-9 in physical
Hessian units. The curvature threshold is absolute 1e-6 hartree in these
coordinates. Classification requires curvature beyond threshold plus measured
uncertainty; these diagnostics are numerical checks, not interval arithmetic.

| Status | Meaning in the declared real internal OV domain |
| --- | --- |
| `STABLE` | All eigenvalues are strictly positive beyond threshold and uncertainty. |
| `UNSTABLE` | A negative lowest eigenvalue is beyond threshold and uncertainty; a unit packed direction is supplied. |
| `NEAR_SINGULAR` | The lowest curvature is unresolved at the chosen threshold; no stable certificate or recovery direction is supplied. |
| `EMPTY` | There are no active OV rotations; no assertion about other domains is made. |

Budget rejection, incompatible state, nonfinite/complex/wrong-shaped action,
non-self-adjoint matrix, failed fresh residual, reference mismatch, or any stale
or failed provider raises `StabilityUnqualified`. Failure returns **no partial
certificate or direction**. Exceptions preserve the underlying cause.

## Certificate consumption

```python
from tools.generativeqc_response.stability import diagnose_uhf_stability

certificate = diagnose_uhf_stability(operator)
certificate.assert_current(operator)  # immediately before subsequent consumption
if certificate.status == "UNSTABLE":
    alpha, beta = operator.problem.layout.split(certificate.direction)
```

The immutable certificate binds response problem, full reference, Hamiltonian,
operator, provider, layout and diagnostic/operator/provider Python source hashes.
Native source/library provenance continues to be owned by the provider identity.
Its own identity covers thresholds, spectrum, direction and numerical defects;
elapsed time is reported separately. Eigenvalues and direction have immutable
byte-backed storage and cannot alias provider scratch. Lifecycle/identity checks
run before and after every action and before publication, including the empty
domain. `assert_current` rejects use with a changed or closed provider/reference.
Providers must update identity when their Hamiltonian changes; in-place equation
overrides or unreported provider mutation are outside the supported contract.

This API neither rotates nor reconverges orbitals. It makes no claims about
complex, spin-flip, external, UKS/XC, geometry tracking, root equivalence, spin
population, nuclear forces or complete root-workflow qualification. Parent
#1826 still requires recovery, intended-root identity and energy/force publication
gates.

## Scientific acceptance

Run the focused gate on Linux with `pyscf==2.14.0`, NumPy, SciPy and pytest:

```bash
PYTHONPATH="$PWD/python:$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
GENERATIVEQC_REQUIRE_UHF_STABILITY_ORACLE=1 python -m pytest \
  tests/python/test_uhf_stability.py -q -o junit_family=legacy \
  --junitxml=uhf-stability.xml
```

Synthetic gates include an independently evaluated common-hcore UHF energy,
mixed-spin finite-difference Hessian, sign/factor controls, spin permutation,
unequal/empty spin domains, internal degeneracy, near-zero thresholds, immutable
scratch ownership, strict pre-admission and stale/late-failure behavior.
Molecular gates use genuinely converged stretched H2 symmetric UHF saddle,
lower broken-symmetry H2 and stable Li doublet references. An independently
rotated UHF energy validates the full Hessian and negative direction; pinned
PySCF's full real internal `2*gen_g_hop_uhf` Hessian is checked after mapping its
virtual-major coordinates to the tool's occupied-major coordinates. Its
`uhf_internal(with_symmetry=False)` status is an additional classification check;
the default single symmetric Davidson seed can miss an antisymmetric spin
instability at an exactly symmetric saddle. The
diagnostic's own dense matrix is never the independent oracle. A PySCF skip is
not a scientific PASS; the environment flag makes missing PySCF a failure.
JUnit records version, reference residual/energy, curvature and certificate
identity. Record source hashes and the actual focused execution artifacts with
the PR's exact head; generic CI green cannot replace these molecular gates.
