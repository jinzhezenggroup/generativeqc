# Cold-start initial guesses

`Calculator` defaults to `initial_guess="auto"`. It selects native Modified
Hückel for FP64, all-electron H-Ar, restricted **CPU/CUDA HF/KS** cold starts,
including density-fitted (DF) targets.
Other domains keep the ordinary Hcore guess. An already available explicit,
imported or warm density always takes precedence. Use `initial_guess=None` to
restore Hcore explicitly, or an `InitialGuessSpec` to request a provider and its
preparation limits. The low-level C API remains explicit-policy driven.
Automatic selection also requires the supported preliminary-guess schema and a
positive Hückel provider bit from `generativeqc_initial_guess_capabilities_v1()`.
CUDA RHF and DF additionally require their respective target-domain capability
bits; an older provider bit alone does not imply those integrations. Libraries
without the new provider retain their previous exact-domain MINAO default when
advertised, otherwise Hcore. Missing queries and unknown schemas retain Hcore.
Explicit `InitialGuessSpec` requests retain native provider/domain checks.

Native CPU automatic admission resolves each item's element, ECP and population
independently: unsupported items keep Hcore without disabling supported neighbors.
Resource planning charges only admitted seeds. CUDA and older-library admission
remains conservative at batch scope: an unsupported element or ECP keeps that
whole batch on Hcore. Where the target has a qualified resource plan, eligible
items include the same Hückel numeric inventory as an explicit request.
Automatic selection does not qualify additional resource-planner
domains (in particular, CUDA global-hybrid plans remain unsupported).
`calc.initial_guess` exposes the
calculator-level candidate; each result reports whether preparation actually ran.
The native v1 policy queries and `AUTO_CPU` kind retain their published MINAO
semantics. New v2 queries and `AUTO_HUCKEL_CPU` advertise the Hückel policy;
existing consumers never receive an unknown provider from a v1 query.

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
energy/force execution for all-electron H-Ar systems. Native Modified Hückel
additionally supports FP64 CUDA RHF and restricted CPU/CUDA DF HF/KS energy/force execution and is available
with `InitialGuessSpec("huckel")` for every supported restricted HF/KS functional,
not just PBE0. CUDA HF MINAO remains unsupported. HF/LDA/MINAO continue to
reject DF. All explicit preliminary requests reject unrestricted, ECP and second-order domains rather
than changing the backend or scientific model. Automatic selection leaves
unsupported domains on Hcore and preserves existing endpoint capabilities,
including qualified CPU second-order endpoints.

For DF targets, only the initial density uses native exact neutral-atomic RHF.
The molecular target retains its requested fitted Coulomb/exchange operators,
auxiliary basis, metric threshold, backend and response provider. CUDA DF RHF
borrows its prepared ordinary eigensystem; CUDA DF KS borrows the KS owner.
An admitted cold seed is published directly, without warm-density normalization.
Preparation failure keeps Hcore; seeded-target failure permits one ordinary
DF Hcore attempt after releasing the failed arena, not an exact-HF retry.

## Frozen-VV10 density proposals

An admitted cold CUDA restricted VV10 solve can follow the initial guess with
a semilocal initializer and bounded frozen-potential density proposals. This
schedule changes neither the requested functional nor the final convergence
criteria. It is separate from the preliminary atomic guess described above.
Each complete target refresh evaluates VV10; intermediate frozen-potential
steps only propose the next density.

For strict solves, the inner stopping threshold is
`clip(0.01 * maximum_residual, min(1e-12, 0.1 * min(1e-9, density_tolerance)), 1e-5)`,
where `maximum_residual` is the latest complete target's **AO maximum** residual,
not the normalized history diagnostic. The initial semilocal stopping rule,
finite iteration limits and reserved ordinary full-target tail are unchanged.
Only fresh complete-target Focks satisfying the caller's energy, density and
physical-residual gates can authorize publication. All strict arithmetic stays
FP64; this threshold does not select mixed precision. Budget-selected proposals
retain their separate [application-precision policy](application_precision.md).

Set both `GENERATIVEQC_VV10_FROZEN_POTENTIAL_SEED=0` and
`GENERATIVEQC_VV10_NATIVE_SEED=0` to retain the ordinary full-target schedule.
This disables the provisional schedule, **not VV10**. Warm/imported densities
retain their existing precedence and admission rules.

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
- `InitialGuessSpec("huckel")` solves neutral spherical-average atomic RHF in
  native CPU FP64, then diagonalizes the updated Wolfsberg–Helmholtz Hamiltonian
  in the occupied atomic orbital subspace and constructs a closed-shell target
  density. Hydrogen uses the one-electron atomic Hamiltonian. There is no
  PySCF runtime dependency, imported reference density, occupation clipping or
  charge repair. A target with too many electrons for this subspace falls back
  to Hcore. Atomic RHF retains all radial functions in occupied angular channels;
  empty angular channels decouple from spherical density and are omitted before
  atomic ERI construction. Identical element and normalized ordered basis
  problems are solved only once per preparation, including when atom shell
  groups are interleaved; different bases on the same element are never aliased

Hückel's atomic solves and minimal-subspace decompositions run on the native CPU
production provider even for CUDA targets. Final strict same-AO admission uses
the target's borrowed GPU eigensolver for CUDA KS and resident/bucketed CUDA RHF.
The independent CUDA HF Fock route already owns host SCF algebra; it uses the
shared native scalar HF eigensolver for preparation and target iteration while
J/K remains on the GPU. Neither route silently retries a failed admission with
a different eigensolver. This is an explicit hybrid preparation contract.
All of this work is included in
the complete endpoint. Atomic convergence requires at least `1e-10` energy,
`1e-8` density and `1e-7` physical residual accuracy; tighter requested controls
are respected. The iteration limit bounds each unique atomic problem.

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
uncharged workspace allowance. Hückel similarly charges all retained seed
matrices, molecular subspace/admission storage, invocation-local atomic frames
and the largest serialized atomic integral/DIIS owner. It never adds an
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
For Hückel, preparation iteration/Fock counters sum the unique atomic RHF
problems, not molecular preliminary Focks. Hydrogen performs no J/K build.

Resident CUDA RHF borrows idle target buffers for cold admission and writes the
accepted density directly, without warm-density normalization or a second GPU
arena. A failed seeded or imported-warm attempt releases its arena before the
single ordinary core retry; successful batch neighbors are not rerun. Target
and discarded-target Fock counts come from the existing device precision ledger.
The seed carries no matching orbital frame, so a used Hückel density retains
ordinary force canonicalization rather than claiming core-seed frame provenance.

## Interpreting performance

Compare complete endpoint cost, including preparation and any retry. A smaller
target iteration count does not guarantee a faster calculation. A seed can also
lead to a different stationary SCF solution in a multi-solution system, even
with unchanged charge and spin. Check final energy, density and relevant state
properties against the intended solution.

`"auto"` is a calculator admission policy, not an `InitialGuessSpec` provider or
a molecule-size profitability heuristic. The CPU and CUDA paths use the same
Hückel construction and scientific admission, but backend-dependent preparation
cost means the CUDA timing evidence is not a CPU speedup claim. HF/LDA remain
explicit opt-ins. A three-stage HF → LDA → target pipeline is an experiment,
not a supported production option.

Automatic selection does not infer profitability from atom count. Hückel is a
numerically guarded default, not a guarantee that every molecule becomes faster.
Atomic preparation is included in complete endpoint timings; small systems or
already inexpensive DF iterations may not amortize it. Use `initial_guess=None`
for rollback and compare the actual workload, including forces and retries.
The superseding default/DF decision and qualification boundaries are recorded in
[the Agent Note](../../.agents/notes/implemented/numerics/2026-10-12-default-huckel-df.md).

## Screening additional providers

`benchmarks/pbe0_initial_guesses.py` screens SAP, Modified Hückel and native
MINAO on the existing 48/96-atom PBE0/def2-SVP water cases. SAP and the
`--guess mod-huckel` arm use **explicit CPU/PySCF experimental preparation**.
The `--guess huckel` arm instead uses `InitialGuessSpec("huckel")` with native
atomic preparation. Summarize a fresh core/MINAO/native-Hückel campaign with
`--native-huckel-only`. No experimental arm enters `initial_guess="auto"` through
this benchmark. The experiment imports only a same-AO density into a fresh
target, and counts preparation, native seed admission, target retries and the
complete synchronized energy/analytic-force endpoint.

CUDA seed admission restores ascending eigenpair presentation when a successful
device solve returns unordered values, moving coefficient columns together
with their eigenvalues. It does not clamp occupations or repair the imported
density. Solver-status, finiteness, eigen-residual, orthogonality and physical
occupation checks remain mandatory; reordering is not numerical acceptance.

Use the existing spherical offline basis, unpruned `GridSpec(48, 16, 32)`,
FP64, exact J/K and unchanged convergence controls. Generate an independent
GPU4PySCF reference for each case. Run at least three fresh-process samples
per arm, rotating arm order rather than running all samples of one provider
together. Preserve the existing compiler/JIT caches and Slurm-assigned device
visibility. For example, inside a finite GPU allocation:

```bash
out=.artifacts/pbe0-initial-guesses
mkdir -p "$out"
for atoms in 48 96; do
  python -m benchmarks.pbe0_initial_guesses reference --atoms "$atoms" \
    --output "$out/reference-$atoms.json"
  guesses=(core sap mod-huckel minao)
  for repeat in 0 1 2; do
    for offset in 0 1 2 3; do
      guess=${guesses[$(((repeat + offset) % 4))]}
      python -m benchmarks.pbe0_initial_guesses native --atoms "$atoms" \
        --guess "$guess" --repeat "$repeat" \
        --reference "$out/reference-$atoms.json" \
        --output "$out/$guess-$atoms-$repeat.json"
    done
  done
done
python -m benchmarks.pbe0_initial_guesses summarize --directory "$out" \
  --output "$out/summary.json"
```

Wrap GPU execution in `srun --partition=main --gres=gpu:5090:1 --nodes=1
--ntasks=1 --cpus-per-task=8 --time=01:00:00 bash -lc '<commands>'` on an
available supported node; do not use the maintenance node. The summary itself
and its unit tests are host-only.

Every observation must converge and agree with the independent reference to
`1e-8 Hartree` in energy and `1e-7 Hartree/Bohr` in force components, with
physical residual RMS at most `1e-10`. Paired final native AO densities must
agree with Hcore to `1e-8`. Failed seed admission, incomplete work counters,
unexpected warm reuse or a target retry disqualifies the candidate, even if the
returned fallback endpoint is numerically correct.

The performance gate is a **strictly greater than 5% complete-endpoint median
wall reduction**, not a reduction in SCF iterations or a preparation-only
speedup. The report compares both Hcore and the existing MINAO default. An
alternative must beat the current default, not merely beat Hcore. Exactly 5%
does not pass. Oracle-backed screening success still requires a separately
implemented and qualified bounded native provider before automatic admission;
frozen-source measurements do not qualify another source revision or broader
molecular/backend domains. The benchmark itself does not change automatic policy.
