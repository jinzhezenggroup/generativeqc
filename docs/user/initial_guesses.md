# Cold-start initial guesses

`Calculator` defaults to `initial_guess="auto"`. It selects projected MINAO for
FP64, all-electron H-Ar, restricted exact **CPU HF/KS and CUDA KS** cold starts.
Other domains keep the ordinary Hcore guess. An already available explicit,
imported or warm density always takes precedence. Use `initial_guess=None` to
restore Hcore explicitly, or an `InitialGuessSpec` to request a provider and its
preparation limits. The low-level C API remains explicit-policy driven.
Automatic selection also requires the supported preliminary-guess schema and a
positive MINAO bit from `generativeqc_initial_guess_capabilities_v1()`. Missing
queries, unknown schemas and HF/LDA-only libraries retain Hcore. Schema 1 alone
does not identify MINAO support. Explicit `InitialGuessSpec` requests retain
their existing schema and native provider/domain checks.

Automatic admission is conservative at batch scope: if any item contains an
unsupported element or ECP, the whole batch keeps Hcore, and resource planning
does not charge an unused MINAO phase. Where the target has a qualified resource
plan, eligible batches include the same MINAO numeric inventory as an explicit
request. Automatic selection does not qualify additional resource-planner
domains (in particular, CUDA global-hybrid plans remain unsupported).
`calc.initial_guess` exposes the
calculator-level candidate; each result reports whether preparation actually ran.

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
energy/force execution for all-electron H-Ar systems. Explicit preliminary
requests reject density-fitted, unrestricted, ECP and second-order domains rather
than changing the backend or scientific model. Automatic selection leaves
unsupported domains on Hcore and preserves existing endpoint capabilities,
including qualified CPU second-order endpoints.

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
  seed boundary a separate MINAO-only construction symmetrizes and normalizes
  it, then projects its orthogonal-metric occupations onto `0 <= f <= 2` with
  their sum equal to the requested electron count. It preserves the metric
  eigenvectors and minimizes the occupation change in Euclidean norm before
  AO reconstruction and strict shared seed validation. The admitted seed may
  therefore differ from the raw PySCF MINAO seed; explicit/imported densities
  still undergo the unchanged validation without this repair

For CUDA MINAO, overlap and occupation decompositions, including strict target
seed admission, use the prepared target's GPU eigensolver and existing charged
workspace. Native cross-overlap construction, matrix assembly and occupation
projection remain on the host; bounded synchronous transfers are part of the
charged cold endpoint. This is not a fully device-resident MINAO implementation.
CPU targets retain the independent reference decompositions. A failing GPU
preparation keeps the existing bounded Hcore fallback, not a hidden CPU MINAO
retry.

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
plus the largest serialized projection, occupation-construction and strict
seed-validation workspace (including eigensolver copies). It never adds an
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

`"auto"` is a calculator admission policy, not an `InitialGuessSpec` provider or
a molecule-size profitability heuristic. The CPU and CUDA paths use the same
MINAO construction and scientific admission, but backend-dependent preparation
cost means the CUDA timing evidence is not a CPU speedup claim. HF/LDA remain
explicit opt-ins. A three-stage HF → LDA → target pipeline is an experiment,
not a supported production option.

## Native resource-planner authority

For installed libraries with `generativeqc_resource_minao_numeric_capacity_v1`,
the Python MINAO planner queries the same C++ numeric-payload bound used by
actual preparation. The query takes target AO count and atomic numbers and
performs no SCF, integral evaluation or CUDA initialization. It rejects invalid
atomic domains and numeric overflow without publishing a partial output.
Retained target and source workspace remain separately charged by the global
resource planner; the bound does **not** represent allocator overhead or process
RSS. Older native libraries lacking this additive private query keep the
previous conservative Python compatibility estimate.

