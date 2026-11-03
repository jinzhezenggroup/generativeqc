# Native unrestricted coupled cluster

Use `uccsd` or `uccsd(t)` for all-electron or scalar-ECP UHF-based energies and analytic forces.
The C ABI, C++ SDK, native CLI and Python `Calculator` select the same C++
method owner. Python is a source-build generation dependency, not a native
runtime computation dependency. Method names and declared properties come
from the [public method manifest](../public_methods.md).

## Calculate an open-shell energy

```python
from generativeqc import Calculator

calculator = Calculator(
    method="uccsd(t)", basis="sto-3g", device="cpu", initial_guess=None,
    ccsd_max_iterations=200,
    ccsd_energy_tolerance=1e-13, ccsd_residual_tolerance=1e-11,
)
result = calculator.singlepoint([("Li", (0.0, 0.0, 0.0))], multiplicity=2)
print(result.energy)
print(result.correlation.ccsd_correlation_energy)
print(result.correlation.ccsd_t_triples_energy)
```

Coordinates are in bohr and energies in Hartree, as for other `Calculator`
methods. `ccsd_*` controls apply to the unrestricted CCSD parent as well as
restricted CCSD. `ccsd_level_shift` modifies only iterative preconditioning;
physical residual replay and the standard triples correction use unshifted
physical Fock/energies. Nuclear repulsion is included exactly once in total
energy by the UHF reference.

The Python-free CLI uses the same options and backend admission:

```bash
generativeqc run li.xyz --method 'uccsd(t)' --basis sto-3g \
  --backend cpu --multiplicity 2 --ccsd-max-iterations 200 --json
```

For C/C++ consumers, select `GENERATIVEQC_METHOD_UCCSD` or
`GENERATIVEQC_METHOD_UCCSD_T` in the existing method descriptor; no ABI field
layout changes are needed. See the [native C++ guide](native_cpp.md).

## Current admission and failure behavior

This native owner requires real FP64, unscreened
conventional or explicitly fitted integrals and s/p/d shells, including supplied
Cartesian or real-spherical bases. Each spin retains its own occupied and
virtual dimensions and all five amplitude blocks remain independent. A singlet
still runs through UHF/UCCSD rather than selecting a restricted CC owner.
There is no fixed AO-count cutoff; admission uses checked dimensions and the
complete storage/work limits described below, before starting the reference.

The native contract supports CPU and CUDA with real FP64. Forces are opt-in
for `Calculator`: pass `properties=("energy", "forces")`. They differentiate
the converged physical CCSD/(T) Lagrangian, including Lambda, full same-space
triples Fock response, the UHF adjoint, Pulay terms and nuclear repulsion.
Fresh amplitude and full orbital stationarity checks precede publication.
No production finite differences or reference-oracle imports are involved.

CUDA is an explicit hybrid endpoint, not a fully resident CC implementation:
integral transforms, CC/response graph arithmetic and nuclear integral
derivatives execute on the selected device. Packed DIIS/Krylov control,
MO-to-AO cotangent transport and small host matrix projections remain native
host consumers. UMP2 also retains native host scalar cotangent assembly.
Conventional CUDA UHF uses native host SCF control and FP64 diagonalization
with device J/K; full DF uses the prepared fitted owner's eigensolver.
Selecting CUDA never retries an operation on the CPU backend.

Supply `auxiliary_basis` and `density_fitting="correlation_cuda"` for fitted
correlation with a conventional UHF reference, or `"full_cuda"` for fitted
UHF J/K and fitted correlation using the same auxiliary/metric policy.
These two unrestricted policies also work with `device="cpu"` for native
reference qualification. `"cpu"`/`"cuda"`/`"auto"` fitting modes select
correlation-only fitting. Host FP64 spectral conditioning is explicit in
the fitted contract. All raw three-center and metric derivative contributions
are included, including auxiliary-center motion and retained/discarded metric
cross terms. CUDA fitted response currently materializes bounded coordinate
derivative tensors and does not claim the conventional shell-streaming flags.

Scalar ECPs retain the checked local/nonlocal provider and its physical-center
derivatives. Occupations and nuclear repulsion use ionic charges; all remaining
valence electrons are correlated. Their staging is reserved in the complete
correlation memory admission. See the [ECP conventions](ecp.md) and
[CBS Hamiltonian identity contract](ccsdt_cbs.md#scalar-ecp-references).

Prepared CC batches, frozen core, KS references, mixed precision and
preliminary initial-guess controls remain unsupported. Internal prototypes
are not production fallbacks.

Preparation admits the simultaneous reference, MO feeds, generated arenas,
packed state, DIIS and triples workspace against
`correlation_memory_budget_bytes` (default 256 MiB). It also bounds worst-case
CCSD/triples scalar contraction work at 100 billion summands using the
configured iteration limit; response actions have the same cumulative work
ceiling. Forces and fitting reserve their additional raw-MO/AO tensors,
metric/derivative buffers and response arenas. The current fitted interaction
is a dense resource-bounded AO Gram map, not a low-memory rank-three CC solver.
These are numeric storage/work limits, not an RSS
or runtime guarantee. Increase neither limit by silently running a different
method/backend; split work or use a qualified endpoint instead.

Every successful endpoint passes a fresh physical replay of all five CCSD
residual blocks. `uccsd(t)` then evaluates all admitted `aaa`, `aab`, `abb`,
`bbb` channels using singleton tiles, without retaining a six-dimensional
triples tensor. Nonconvergence, nonfinite values, small denominators or late
triples failure publish neither a partial energy nor stale correlation
diagnostics. `result.correlation` and `result.cc_performance` report only a
successful complete endpoint.
