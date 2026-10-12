# Native UCCSD Lambda and fixed-orbital input adjoints

`tools.generativeqc_cc.PreparedUCCSDLambda` borrows a live
[conventional UCCSD owner](uccsd_solver.md) and its converged state. It solves
the five-block unrestricted adjoint using the parent's native CPU or
ordinary-stream CUDA executor. Both backends differentiate the same physical
[energy/residual TensorIR](uccsd_equations.md) through the generic tensor AD
machinery. There is no production PySCF/interpreter fallback, restricted Lambda
substitution, or dense amplitude Jacobian.

This is an internal supplied-reference response API. Its parameter outputs are
**fixed-orbital correlation-Lagrangian input cotangents**, not relaxed RDMs,
normal-ordering pullbacks, orbital response, complete nuclear forces, a
corrected-(T) Lambda solve, or native Calculator registration.

## Physical primal and dual coordinates

The [five-block layout](uccsd_layout.md) uses unweighted independent excitation
coordinates. Let `U` expand them into full signed arrays, and let `w = U*U`:
`w=4` for `aa`/`bb` doubles and `w=1` for singles and `ab` doubles.

- Amplitude directions expand with `layout.unpack`, namely `U`.
- Full amplitude cotangents pull back with `layout.pack_dual`, namely `U*`.
- Packed residual cotangents lift with `layout.unpack_dual`, namely `U/w`.
- Published physical Lambda arrays use ordinary `layout.unpack(lambda)`,
  **without** the quarter factor. Their keys are
  `l1a,l1b,l2aa,l2ab,l2bb`.

Thus `J* lambda = -grad(E_corr)` and stationarity of
`E_corr + lambda dot r` use an unweighted packed dot. In a full-array residual
VJP the same-spin seed is physical Lambda divided by four. Ordinary amplitude
packing is not a cotangent pullback. RCCSD's pair-symmetric/sqrt-weighted
coordinate policy must not be reused here.

`generativeqc_compiler.cc.uccsd_response.build_response_programs` provides
energy VJP, residual JVP/VJP and combined Lagrangian amplitude VJP programs.
`build_parameter_vjp` builds one canonical independent input VJP on demand.
These differentiate equations at fixed amplitudes, not iterations, DIIS,
proposal denominators, damping or level shifts.

## Lifecycle and qualification

```python
from tools.generativeqc_cc import PreparedUCCSD, PreparedUCCSDLambda

with PreparedUCCSD(snapshot, provider) as parent:
    state = parent.solve()
    if not state.converged:
        raise RuntimeError(state.reason)
    with PreparedUCCSDLambda(parent, state) as response:
        multipliers = response.solve()
        bar = response.parameter_cotangent(multipliers, "fov_aa")
        directional_correlation_response = bar.contract(delta_fov_aa)
```

The borrowed parent, reference, immutable feeds, options and executor must stay
open and configuration-stable. An injected transpose solver must preserve its
admitted identity, tolerances and workspace contract. Provider and AO-source
lifetimes are not borrowed: they may close after parent preparation. Closing
Lambda releases its own state/action metadata, never the parent or executor.

Admission verifies reference/Hamiltonian/integral/equation identities, finite
consistent energies, numeric/work limits and Krylov storage before amplitude
copies or native execution. The five full arrays must exactly agree with the
packed converged state. Every solve freshly replays the physical parent energy
and all five residuals. Default CC and Lambda tolerances are `1e-9`; neither
may be weakened above this value. Energy drift must be at most
`min(1e-10, parent.options.energy_tolerance)`.

The shared checked transpose solve verifies the true linear residual rather
than trusting the solver's success flag. Default GMRES uses `rtol=0`,
`atol=1e-11`, `max_iterations=400`. A fresh combined Lagrangian VJP then checks
the packed stationarity norm and each full block's weighted maximum against the
Lambda tolerance. These component/combined checks share the same equation DAG;
they are not independent scientific oracles. Any failure raises before result
publication. Empty excitation spaces publish exact zero Lambda without GMRES.

Results own non-aliasing immutable FP64 arrays and recursively frozen
provenance. Parameter requests verify the supplied Lambda identity and actual
stationarity afresh; a forged result's success flag or hashes are insufficient.

## Parameter scope

`response.programs.parameter_names` lists canonical independent Fock and
antisymmetrized-integral inputs. `parameter_cotangent` returns exactly one
input's bar, with the physical residual seeds and unit correlation-energy seed.
One physical Hamiltonian perturbation can affect several canonical inputs;
the upstream consumer must accumulate every binding and its transpose or signed
permutation. Independent graph inputs must not be mistaken for a complete
physical Hamiltonian pullback.

In particular, physical ERI changes also affect normal-ordered Fock matrices.
Orbital energies, orbitals, overlap, HF energy and nuclear coordinates require
their own explicit chains. Those bindings and shared UHF orbital response are
outside this owner's contract. Standard CCSD stationarity is enforced here;
corrected-(T) multipliers cannot simply be supplied to this method.

## Bounded work and storage

Defaults are `max_bytes=512<<20`, `max_operator_actions=2048` and
`max_total_work=1_000_000_000_000`. Admission charges retained parent reference,
feeds and denominators, coordinate/state scratch, Krylov workspace and the
largest native action's logical storage. Parameter admission additionally
charges its output/copy scratch and native action. These are logical numeric
bounds, not process RSS or compiler/cache metadata bounds.

The complete Lambda work reservation includes fresh parent replay, the energy
RHS, the maximum permitted transpose actions, and combined stationarity.
Actual transpose calls are capped even for injected solvers. Each parameter
request admits its replay, stationarity and VJP before native execution.
The common generated CPU scalar-loop model counts semantic contraction work
on both backends; using that metadata on CUDA does not execute a CPU oracle.
It is not hardware FLOPs and excludes host Krylov/coordinate work.

`lambda_endpoint_seconds` includes owner preparation, fresh replay, native
compilation/execution, Krylov control, stationarity and bulk result validation/
copying. `parameter_endpoint_seconds` covers the complete request through bulk
publication preparation. Earlier SCF/provider/UCCSD work is excluded and must be
accounted separately in complete-method performance claims. These diagnostics
alone are not a Calculator performance claim.

## Independent qualification

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=python:. \
  GENERATIVEQC_LIBRARY="$PWD/build/libgenerativeqc.so" \
  python -m pytest tests/python/test_uccsd_response.py \
  tests/python/test_uccsd_lambda.py tests/python/test_uccsd_native.py -q
```

The derivative tier checks signed-map duality, native JVP/VJP transpose dots,
and all canonical input VJPs against an independent determinant-space/CAR
Hamiltonian Lagrangian finite difference. A deliberately wrong same-spin seed
factor must fail the transpose gate. Molecular Lambda gates cover Li, OH, NH2,
broken H4 and asymmetric H6 against independent PySCF, with every multiplier
block's absolute error at most `2e-7`. The restricted limit agrees with the
existing RCCSD full-Frobenius Lambda after the explicit dual conversion.

Fixed-orbital Fock bars are checked against reconverged independent PySCF CC
correlation-energy finite differences. Native-producer tests additionally
cover native UHF, streamed native shell integrals and native Lambda/parameter
execution for H2, Li and OH. They share only exact shell definitions with the
oracle; symmetry-preserving AO core-Hamiltonian directions avoid identifying
MO phases or degenerate electronic-state orientations between independently
converged references. Perturbed CC solves start from independent oracle CC,
with unchanged tight convergence and finite-difference acceptance gates.

Real CUDA qualification uses `-k complete_cuda`,
`GENERATIVEQC_TENSOR_CUDA_TEST=1`, an explicit CUDA compiler/architecture, and
a permitted GPU through a finite Slurm allocation. NH2, broken H4 and asymmetric
H6 exercise CUDA transforms, converged UCCSD, solved Lambda and Fock/same-spin/
mixed-spin integral VJPs. A host skip is not CUDA evidence.

The durable dual-normalization and ownership rationale is in
[the Agent Note](../../.agents/notes/implemented/numerics/2026-10-12-uccsd-physical-lambda-duals.md).
