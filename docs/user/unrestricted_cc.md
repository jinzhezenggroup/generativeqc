# Native unrestricted coupled cluster

Use `uccsd` or `uccsd(t)` for conventional all-electron UHF-based energies.
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

This native owner is bounded to at most twelve AOs, real FP64, unscreened
conventional integrals and all-electron s/p/d shells, including supplied
Cartesian or real-spherical bases. Each spin retains its own occupied and
virtual dimensions and all five amplitude blocks remain independent. A singlet
still runs through UHF/UCCSD rather than selecting a restricted CC owner.

The current public native contract is **CPU energy only**. CUDA, analytic
forces, prepared batches, DF/RI, ECP, frozen core, KS references, mixed
precision and preliminary initial-guess controls are rejected explicitly.
The separate internal response/gradient prototypes are not production
fallbacks and their earlier GPU/force evidence does not qualify this owner.

Preparation admits the simultaneous reference, MO feeds, generated arenas,
packed state, DIIS and triples workspace against
`correlation_memory_budget_bytes` (default 256 MiB). It also bounds worst-case
CCSD/triples scalar contraction work at 100 billion summands using the
configured iteration limit. These are numeric storage/work limits, not an RSS
or runtime guarantee. Increase neither limit by silently running a different
method/backend; split work or use a qualified endpoint instead.

Every successful endpoint passes a fresh physical replay of all five CCSD
residual blocks. `uccsd(t)` then evaluates all admitted `aaa`, `aab`, `abb`,
`bbb` channels using singleton tiles, without retaining a six-dimensional
triples tensor. Nonconvergence, nonfinite values, small denominators or late
triples failure publish neither a partial energy nor stale correlation
diagnostics. `result.correlation` and `result.cc_performance` report only a
successful complete endpoint.
