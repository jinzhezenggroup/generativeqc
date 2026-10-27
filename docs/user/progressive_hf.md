# Deterministic progressive HF

GenerativeQC exposes a bounded two-stage Hartree-Fock controller for using a cheaper
HF calculation only as an initial-state proposal for an explicitly requested
target. The controller never treats the source energy, residual, basis or
provider as the target result.

```python
from generativeqc import (
    Calculator,
    ObservableTarget,
    ProgressiveBudget,
    TargetAccuracy,
    TargetProblem,
    make_deterministic_hf_plan,
    run_progressive_hf,
)

atoms = (("H", (0, 0, -0.7)), ("H", (0.1, 0, 0.7)))
accuracy = TargetAccuracy(
    (ObservableTarget("energy", "absolute", "Eh", absolute=1e-8),)
)
source = Calculator(
    basis="sto-3g",
    energy_tolerance=1e-7,
    density_tolerance=1e-6,
    max_iterations=20,
)
target = Calculator(
    basis="def2-svp",
    energy_tolerance=1e-12,
    density_tolerance=1e-10,
    target_accuracy=accuracy,
)

problem = TargetProblem.from_calculator(target, atoms)
plan = make_deterministic_hf_plan(
    problem,
    source,
    target,
    atoms,
    budget=ProgressiveBudget(
        maximum_source_iterations=20,
        maximum_total_iterations=120,
    ),
)
run = run_progressive_hf(plan, source, target, atoms)
print(run.target.energy, run.verification.status)
```

CPU all-electron RHF/UHF initialization with the built-in projection policy uses
one native system-projection owner for overlap preparation and all spin channels.
It validates identical ordered nuclei, geometry and electron/spin state, then
publishes a seed only after every channel passes. `projection_owners` records
`native_cpu_system` for that path. Old libraries, custom policies and unsupported
backend/model domains retain explicit preparation compatibility; an admitted
native error never retries it. The native matrix-capacity query adds live metrics
and detached spin results to the existing projection workspace. Prepared fleets,
caller buffers and opaque provider overhead are separate; this is not an RSS bound.
Python still owns per-item eligibility/reporting and transactional seed installation.

## Immutable target and typed plan

`TargetProblem` snapshots the resolved scientific model, prepared-provider
identity, basis mathematical/source hashes, observable accuracy requirements,
solver tolerances, requested atom count and the final physical-residual gate.
Grid and local-policy identities are explicit fields and must be absent in the
first HF contract. This prevents a later adapter from silently changing those parts
of the target.

`StagePlan` separately records the resolved model, provider, arithmetic policy,
solve tolerance, transfer operation, optional cost estimate, typed error
evidence and allowed successor stages. `DeterministicHFPlan` currently accepts
exactly:

1. an RHF/UHF initialization compatible in geometry, electron count, spin and
   Hamiltonian, followed by metric density projection; and
2. the exact target model/provider with an FP64 final state. `precision="auto"`
   is admitted only when strict refinement is required and later observed.

Different basis or fitted-operator stages are staged warm starts, not a claim of
one continuous homotopy. No orbitals, virtual spaces or correlated amplitudes
are transported by this controller.

## Failure and budget behavior

Stage count, source iterations, total configured iterations, host projection
workspace, optional Fock builds and optional estimated cost have explicit
budgets. A plan whose configured maximum work exceeds its budget is rejected
before execution. Projection and final-verification host work have separate
bounds. Before creating the verification AO/Fock owners, the controller reserves
its dense audit matrices/copies and checks the existing HF provider inventory
against the remaining host capacity. The full HF inventory is a conservative
admission bound for this fixed-density operation; it is not a measured process
or driver-memory peak. An unavailable inventory fails closed. Exhausting either
part returns `budget_exhausted` without starting the verification Fock rebuild.
Failed source work remains in the execution record and total work count.
Diagnostics separately report source/target setup and execution, projection,
final verification and context cleanup, while `total_seconds` covers the complete
endpoint.

A failed source solve or rejected/out-of-budget projection skips the proposal
and executes the unchanged target from its ordinary cold guess. The target stage
is never replaced by a source result. Target failure is returned in
`FinalVerification`; the public runner does not manufacture convergence by
relaxing tolerances.

## Final verification

`FinalVerification` checks the actual resolved target identity, prepared-provider
identity, basis mathematical/source hashes, native success/convergence,
energy/density gates, physical SCF residual, requested energy/force capabilities,
effective final FP64 arithmetic and complete execution budgets.

The shared native `generativeqc_hf_verify_final_v1` policy owns scalar acceptance,
strict-cleanup requirements and budget precedence. A running controller uses its
selected target library; standalone verification uses the first configured
library. Python validates typed identities and observable array shape and
presents the native decision. Missing older-library protocols and Python integer
metadata outside the fixed-width ABI retain explicit compatibility; a native
error never triggers another verifier. This metadata policy does not rebuild the
physical target or certify an empirical accuracy estimate.

A force capability is established only by a finite, real numerical array with
shape exactly `(TargetProblem.atom_count, 3)`. Empty, partial, extra-atom,
complex or nonnumeric arrays fail closed as an omitted requested observable.

Legacy HF results do not export the optional public physical-residual field. The
controller therefore performs one explicit fixed-density rebuild with the exact
target basis, direct/DF operator, screening, metric and arithmetic policy. It
records the Fock execution identity, density/overlap hashes, RMS and maximum
`FDS-SDF`, and fixed-density energy agreement. This audit is included in target
time and counts as one target Fock build; its failure makes target verification
fail closed. It is a state audit, not an observable-error certificate.

For all-electron CPU RHF/UHF Direct/DF inputs within the native through-g domain,
the fixed-density audit uses one native scope for preparation, Fock construction,
overlap, residual reduction and cleanup. Its additive C ABI exposes a capacity
query and transactional execution for SDK callers. The existing complete HF
resource preflight remains an additional frontend gate; native execution also
checks the conservative numeric envelope before constructing a provider. This
does not establish complete allocator/provider/RSS accounting.

The native receipt keeps the maximum and RMS distinct, counts one physical Fock
build, and binds the density/overlap bytes and versioned operator/execution
identity. It creates no AO-grid packing owner. Missing protocols, non-CPU and
unsupported metadata domains retain explicit compatibility; an admitted native
failure never retries Python or changes the requested model or budget.

Final disposition distinguishes:

- `verified`: exact target gates passed and an independent assessment has
  status `observed_met`;
- `unverified`: the exact target is established, but observable accuracy has no
  independent reference audit;
- `unmet`: a target identity, state, arithmetic, observable or accuracy gate
  failed; and
- `budget_exhausted`: actual work exceeded a bound or a requested work counter
  was unavailable.

`target_established=True` therefore does not imply that an energy or force error
target is certified. A small SCF residual cannot stand in for observable error
evidence. See the [architecture decision](../../.agents/notes/implemented/architecture/2026-09-22-progressive-hf-target-boundary.md)
for the rationale and adapter boundary.
