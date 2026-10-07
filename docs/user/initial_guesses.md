# Cold-start initial guesses

`Calculator` can explicitly prepare a cheaper density before the requested SCF
calculation. The default remains `initial_guess=None`, which
preserves the ordinary core guess or an already available warm density.

```python
from generativeqc import Calculator, GridSpec, InitialGuessSpec, KsOptions

calc = Calculator(
    "pbe0-rks",
    basis="6-31g",
    device="cpu",
    ks_options=KsOptions(grid=GridSpec(24, 12, 24)),
    initial_guess=InitialGuessSpec("hf"),
)
result = calc.singlepoint(atoms, properties=("energy",))
print(result.energy, result.initial_guess)
```

HF and coarse-LDA preparation support CPU, FP64, all-electron, restricted
closed-shell RHF/RKS **energy** calculations with exact two-electron providers.
The projected MINAO provider additionally supports FP64 CUDA restricted KS
energy/force execution for all-electron H-Ar systems. Density-fitted,
unrestricted, ECP and second-order requests remain rejected rather than changing
the backend or scientific model. Without the explicit option, existing
capabilities are unchanged.

## Providers and controls

- `InitialGuessSpec("hf")` runs an ordinary restricted Hartree–Fock preparation
  in the target AO basis. It has no DFT grid
- `InitialGuessSpec("lda")` runs the existing LDA provider with an independent
  coarse `GridSpec(8, 6, 12)`; its AO-grid provider supports through f. For a
  higher-angular-momentum RHF target, this provider declines before allocating
  an extra integral owner and the ordinary core guess is retained
- `InitialGuessSpec("minao")` constructs the occupied H-Ar ANO/MINAO atomic
  reference, projects it into the target AO metric with the native rectangular
  cross-overlap, and forms the density without a preliminary J/K/XC/Fock build.
  The raw projection follows PySCF's MINAO construction. At the native target
  seed boundary it is symmetrized and normalized to the requested electron
  count using the existing density-admission contract

HF/LDA use at most 32 preliminary iterations by default, DIIS history 8, energy
tolerance `1e-6` and density tolerance `1e-4`. These are **preparation controls**.
The target calculation retains its requested method, basis, quadrature,
precision, charge, multiplicity and convergence tolerances. No orbitals or DIIS
history are transported: only a validated same-AO density is proposed.

The HF/LDA iteration limit can be set from 1 to 64 and DIIS history from 1 to
16. MINAO performs zero preliminary iterations, so those SCF controls are
accepted for schema compatibility but do not cause an SCF solve. Preliminary
LDA accepts a v1 equal-radius grid with 2–32 radial, 2–16 polar and
4–32 azimuthal points. Its partition and coincidence rules stay fixed. This
preparation grid does not replace the target grid.

```python
policy = InitialGuessSpec(
    "lda",
    max_iterations=24,
    maximum_numeric_bytes=256 << 20,
    grid=GridSpec(8, 6, 12),
)
```

An explicit/imported density or a retained warm density always takes precedence.
Prepared-batch clearing, freezing, checkpoint import and last-good publication
keep their existing behavior. A cold call may use preliminary SCF; a successful
warm replay skips it. The policy is execution-only and is not part of the
checkpoint's scientific model. The destination calculator selects the policy.
Changing the policy of an existing prepared batch requires preparing a new batch.

## Bounds and fallback

`maximum_numeric_bytes` bounds the preliminary phase's numeric payload using a
conservative preflight. Its 256 MiB default is an **additional-phase** limit,
not a process-RSS or whole-calculation budget. Object headers, allocator/BLAS
storage and the retained target owner are outside that narrow cap.

For a whole-endpoint host budget, use the ordinary `ResourceBudget`. For
HF/LDA the planner conservatively composes the complete preparation inventory
with the target inventory. For MINAO it charges all retained cold seed matrices
plus the largest serialized cross-basis projection workspace. It never adds an
uncharged workspace allowance. This can overestimate serialized lifetime overlap. An
infeasible global plan is rejected before execution, rather than silently
changing a requested strategy to make the plan fit.

If the preliminary numeric cap is exceeded, preparation is skipped and the
normal core guess is used. A nonconverged, invalid or non-allocation-failed
preparation also keeps the core guess. A seeded target that does not converge,
or raises a numerical runtime error, gets **one** fresh core-guess retry with
the original target settings. Existing warm failures go directly to their core
retry without reactivating preliminary SCF; with this policy enabled, that retry
is bounded to one attempt too. Allocation failures propagate. The optional
strategy never raises the target iteration limit or relaxes its tolerances.

`result.initial_guess` reports the requested provider, outcome, preparation
iterations/Fock builds/time, estimated preparation numeric capacity, target
attempts and discarded-target work. `work_counters_complete=False` marks an
exception that prevented a complete work census; zero counts then do not imply
zero work. Ordinary result iteration counts describe the returned target
attempt, not the sum of preparation and discarded attempts.

## Interpreting performance

Compare complete endpoint cost, including preparation and any retry. A smaller
target iteration count does not guarantee a faster calculation. A seed can also
lead to a different stationary SCF solution in a multi-solution system, even
with unchanged charge and spin. Check final energy, density and relevant state
properties against the intended solution.

There is no `auto` provider, molecule-size heuristic, or default promotion in
this interface. MINAO is an opt-in implementation candidate for the CUDA hybrid
cold-start qualification in issue #2052; complete cold endpoint evidence is
required before any default change. A three-stage HF → LDA → target pipeline is
an experiment, not a supported production option.
