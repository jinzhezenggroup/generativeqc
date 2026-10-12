# UCCSD physical response and shell-streamed forces

`tools.generativeqc_cc.PreparedUCCSDPhysicalResponse` borrows a live standard
[UCCSD Lambda owner](uccsd_lambda.md), its result, and an identity-matched
conventional provider/source. `PreparedUCCSDGradient` adds complete all-electron
conventional nuclear derivatives in the supplied reference's AO frame. These
are internal APIs, not Calculator/native method registration, frozen-core/ECP
support, or corrected-UCCSD(T) forces.

## One physical Hamiltonian DAG

`generativeqc_compiler.cc.uccsd_hamiltonian` rotates `h_a,h_b,g_aa,g_ab,g_bb`
with independent spin-owned `rotation_a,rotation_b`, then normal-orders the
rotated Hamiltonian with unit spin occupations. ERIs use chemists' order;
`g_ab` owns alpha on the first pair and beta on the second. `g_ba` is its
pair-transposed view, never another independent input. Pair exchange is not
a symmetry of fixed-owner `g_ab` when the two orbital frames differ.

The compiler substitutes these raw bindings into the existing UCCSD
energy/residual DAG before a single generic AD pullback. There is no second CC
equation algebra or one full Lagrangian replay per canonical input. Seeds are
unit correlation energy, physical residual dual `U lambda/w`, optional unit
electronic HF energy, and the shared UHF constraint adjoint on each spin Fov.

The shared UHF convention is `C'=C exp(-K)`, `K_ia=x`, `K_ai=-x`.
Its generated Fov JVP therefore uses `dU=-K`, orbital RHS is
`+antisym(G_U)_ia`, and constraint seeds are `+z`. Per-spin overlap weights
are `-(G_U+G_U.T)/4`. Do not import a restricted sign or occupancy factor.

## Preparation and fresh scientific gates

Both owners admit logical numeric state, native tensor staging and generated
scalar-loop work before scientific execution. The physical owner additionally
caps producer tile traversals, primitive source tuples and UHF operator actions;
the force owner caps all ordered shell quartets and derivative primitive tuples.
The provider's own budget admits all three full raw-MO Coulomb blocks. A CUDA
provider requires an explicitly supplied matching exact `CudaSpinJKBackend`;
there is no CPU orbital fallback. CPU uses the same source through
`NativeJKBackend`. Krylov control, frame transforms in shared UHF, and classical
nuclear Coulomb arithmetic remain explicit host work.

Every solve independently checks actual standard-CCSD amplitude stationarity,
all canonical raw bindings, electronic HF plus nuclear energy, correlation
energy and CC residuals. Two generated Fov JVPs must agree with shared UHF
actions. Checked GMRES establishes the true Z residual, then a fresh physical
pullback checks **every** antisymmetric orbital block, not only occupied/virtual
entries. Stationary broken-symmetry branches need not have positive curvature.

`UCCSDRelaxedCotangents` owns immutable full-Frobenius raw-MO and overlap
weights. These are not packed RDM conventions and are not forces by themselves.
Borrowed providers, sources, solvers and backends must remain open and stable.
The nuclear binding is verified against the native source. Non-native supplied
providers require an explicit finite nuclear repulsion and cannot supply the
native complete force endpoint.

## Shell-local nuclear binding

```python
from tools.generativeqc_cc import (
    PreparedUCCSDLambda,
    PreparedUCCSDPhysicalResponse,
    PreparedUCCSDGradient,
)

with PreparedUCCSDLambda(parent, state) as lambdas:
    multipliers = lambdas.solve()
    with PreparedUCCSDPhysicalResponse(lambdas, multipliers, provider) as physical:
        with PreparedUCCSDGradient(physical) as gradient_owner:
            result = gradient_owner.solve()
            forces = result.forces
```

For CUDA, pass `response_backend=matching_exact_spin_backend` to the physical
owner and increase explicit complete-owner budgets as necessary. The parent,
provider, source, executor and spin backend remain borrowed, not closed by these
response/force owners.

One-electron and overlap weights accumulate from each spin using its own
coefficient frame. The ordered shell ERI weight is the sum of three shared
four-coefficient transforms for `aa,ab,bb`, with each axis's actual spin owner.
There is no factor of two on `ab`, pair folding, molecular AO rank-four
cotangent, or coordinate-major derivative tensor. All four independent slot
derivatives are scattered to physical atoms after native success, including
repeated atoms. Nuclear repulsion is added exactly once. Results contain a
positive energy gradient and exactly its negative forces.

CPU ERIs require existing packaged s/p/d derivatives and fail if unavailable.
CPU one-electron derivatives use the explicitly named
`native-compatibility` differentiated recurrence, not a hidden generated-kernel
claim. CUDA uses the existing generated one-electron and shell-ERI consumers
without CPU derivative fallback. The current complete endpoint supports s/p/d
in Cartesian and real-spherical representations.

## Resource and qualification boundaries

Byte limits are logical numeric reservations, not process RSS. The force budget
includes a CPU compatibility scratch allowance rather than an allocator-enforced
cap on that recurrence; CUDA derivative staging is enforced natively. Compiler,
interpreter, CUDA context and opaque library allocations are outside the model.
Primitive tuples are a separate semantic gate, not scalar recurrence FLOPs.
Host Krylov implementation work is not scalar-counted. Ordered shell transforms
can amplify work; the full schedule is admitted, and no performance claim is
implied by avoiding a dense AO tensor. Earlier SCF, CC and Lambda solves must be
included separately in complete-method timing claims.

Qualification tests are `test_uccsd_hamiltonian.py`,
`test_uccsd_physical_response.py`, and `test_uccsd_gradient.py`. They cover an
independent determinant-space/CAR Lagrangian, explicit NumPy spin transforms,
native producer chains, independent molecular analytic gradients, stricter
reconverged total-energy finite differences, restricted-force agreement and
fail-closed resource/publication paths. The installed independent open-shell
analytic oracle can differ at microhartree scale, so its `2e-6` force gate is
paired with a separate `2e-7` complete-energy finite-difference gate; the latter
is not relaxed. Real CUDA tests require finite Slurm allocation and explicit
`GENERATIVEQC_TENSOR_CUDA_TEST=1`.
