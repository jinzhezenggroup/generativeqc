# Equal-spin CUDA XC qualification

The native RKS XC point producer automatically shares scalar work when its
spin-feature producer proves exact equality. It is **on by default**;
`GENERATIVEQC_CUDA_XC_EQUAL_SPIN=0` restores the incumbent point algebra,
while `1` explicitly enables sharing. Other values are rejected. The selector applies to **every
registered native CUDA semilocal consumer**, including LDA/PBE and generated
split-global-hybrid consumers, and the structurally registered automatic Libxc
CUDA AOT point programs when that internal bundle is enabled. Coverage follows the registry, not a manually
maintained method whitelist. It does not enable a new method or expand a
method's existing backend admission.

UKS and response retain their existing point algebra. The RKS producer binds
both spins to the same half-density, gradient
and kinetic-density lanes; method labels or approximate equality do not select
the candidate. Single-tile and batched point launches use the same contract.
There is no additional device buffer, reduced precision, altered quadrature,
changed stopping criterion or CPU/reference dependency in this candidate.

The candidate also binds the already-admitted functional code on the host before
instantiating its device point kernel. This prunes unrelated GGA/MGGA algebra
before register allocation without changing the scalar program. The dispatcher
checks the producer's spin layout and exact feature width before launch. Curated
and split-hybrid launch metadata are generated from their existing authoritative
registries; DFT generation does not import the XC scientific implementation.
Feature width is generated from rho/gradient/tau requirements. Physical GGA
launches use 32 threads and LDA/MGGA launches use 128 threads, consistently for
curated, split-hybrid, single-tile and batched consumers; response keeps 128.
This ingredient-based launch policy applies to both selector arms. No method
label or numeric-code range controls eligibility or launch width.
Automatic providers retain their existing 128-thread schedule in both arms;
availability of a restricted scalar body does not qualify a new launch width.
The incumbent curated kernel keeps its original three template arguments and
resource classification. Its separate restricted entry and the automatic
provider sibling reuse the same physical-feature traversal; neither supplies
a response entry or inherits an incumbent resource receipt.
Prepared stationary AO derivative order likewise follows the registered gradient
and kinetic-density requirements, rather than treating one numeric code as LDA.
Unclassified legacy consumers retain conservative first-derivative storage.

## Stationary point coverage

The same default-on, capability-based policy makes every registered unpolarized
stationary point lowering eligible for the precomputed geometry-point producer.
`GENERATIVEQC_STATIONARY_RESTRICTED_POINT=off|on` controls its request and takes
precedence over the retained `GENERATIVEQC_STATIONARY_PBE0_RESTRICTED_POINT`
compatibility spelling. The request defaults to `on` and registered,
qualified consumers are selected without a functional-name whitelist.
`GENERATIVEQC_CUDA_XC_EQUAL_SPIN=0` disables the restricted scalar schedule
for all consumers, including PBE0. PBE0's established specialized mathematics
and subnormal continuation remain unchanged.

Execution still requires a current generation-bound equal-feature witness,
phased storage with sufficient scratch, and no external/nonlocal point seed.
UKS, unproven or stale producers, legacy artifacts, and bounded/nonphased
routes retain their explicit general-point fallback. The candidate reuses
each consumer's original screening, work-input and continuation policies.
The v1 metrics ABI retains its old capability spelling;
`restricted_point_capable` is the method-independent runtime metric alias.

## Stationary component packing

Stationary Cartesian component pages use bounded `bulk` packing by default.
Set `GENERATIVEQC_STATIONARY_COMPONENT_PACKING=scalar` to select the old
per-descriptor producer; an explicit `bulk` also works. Other values are
rejected. This separate selector does not depend on equal-spin XC. It
vectorizes descriptor construction for already-admitted s/p/d integral roots,
without replacing force kernels, changing primitive work or widening method
admission. Ragged expansions preserve scalar descriptor/FP64 coefficient order
and both task-capacity and primitive-work flush boundaries. Host admission
includes bounded mixed-radix scratch and linear AO topology in both modes;
there is no retained quartet inventory or fanout-amplified scratch allocation.
Noncomponent pages retain their existing bulk producer. Validate complete
cold/warm/moved timings, descriptor/primitive counts and independent molecular
forces separately from the XC point ablation. See the
[component-page rationale](../../.agents/notes/implemented/performance/2026-10-11-stationary-component-page-packing.md).

## Direct derivative ownership

For s/p/d Direct J/K plans, retained shell derivatives now automatically
attempt the bounded generated derivative owner when the derivative order,
producer and optional device-memory budget permit it. Set
`GENERATIVEQC_DIRECT_SHELL_DERIVATIVE_FALLBACK=0` to disable this attempt
(`1` also enables it). Failed optional preparation keeps the existing
bounded derivative route, and uncovered bounded SCF **values** remain
unqualified and disabled independently of derivative ownership.

## Scalar contract

The compiler differentiates the original polarized energy before binding
`rho_b=rho_a`, `sigma_bb=sigma_aa` and `tau_b=tau_a`. Structural CSE then shares
identical subgraphs without reassociating the remaining arithmetic or removing
either spin's derivative outputs. `sigma_ab` stays independent: Libxc's
diagonal-sigma floors can make it differ even when the original RKS gradients
are identical. Component-specific density screens, sigma/tau floors, Fermi-hole
constraints and original-density energy rescaling are unchanged.

Run the strict host scalar and selector checks with the optional pinned
reference dependencies installed:

```bash
PYTHONPATH=python:. OMP_NUM_THREADS=1 python -m pytest -q \
  tests/python/test_automatic_equal_spin_cuda.py \
  tests/python/test_equal_spin_xc_codegen.py \
  tests/python/test_functional_independent_xc_optimization.py \
  tests/python/test_split_hybrid_work_policy.py \
  tests/python/test_r2scan_boundary_codegen.py
```

The equal-spin tests compile the actual generated CUDA scalar bodies as strict
host C++, compare all polarized partials to the incumbent, exercise work-input
boundaries and compare ordinary physical points to independent Libxc. Passing
these tests does not establish molecular endpoint correctness or performance.
They also execute the actual emitted host dispatch against launch stubs, checking
the exact functional, single/batched specialization and incompatible inputs.
Automatic checks generate a sibling for every structural registration and
compile representative LDA/GGA/MGGA scalar bodies as strict host C++ against
polarized Libxc partials. This is generation/point qualification, not public
automatic CUDA KS/force admission. The internal AOT option and its remaining
production gates are documented in
[native CUDA XC](../developer/xc_native_cuda.md).
Check the generated split-hybrid binding projection and compiler boundaries with:

```bash
PYTHONPATH=python:. python tools/generate_xc_split_hybrid_registry.py --check-python
PYTHONPATH=python:. python tools/check_compiler_structure.py
```

Device qualification must exercise the native coefficient kernels as well as
the isolated scalar siblings. Cover both channel-major single-tile and compact
batched layouts, including a final partial tile, compare every coefficient and
weighted point-total lane to independent Libxc, and retain separate memcheck and
initcheck runs. Static ptxas stack/spill reports are resource diagnostics, not
dynamic memory-work counts or endpoint speedups.

## Representative regressions and performance claims

Default eligibility depends on ingredient/spin contracts, exact producer
equality and lineage, supported native capabilities, and bounded resource
admission—not individual method names or one-off benchmark approvals.
Routine Python CI exercises default-on/explicit-off selection, UKS/response
exclusions, component descriptor parity, and derivative fallback under limited
coverage and budgets. The generic routes remain available as reference controls.

For claims of **measured whole-method speedups**, follow the
[performance engineering checklist](performance_engineering.md):

- Compare explicit `0` and `1` in the same authenticated native binary, with
  identical method, basis, grid, precision, screening and convergence settings.
- Include cold SCF, warm and changed-geometry energy-plus-force calls, with at
  least five interleaved observations per arm and a larger planner-sensitive
  system. Synchronization and host return belong inside the measured endpoint.
- Retain actual SCF/Fock counts, grid/XC points, AO work, force work, memory
  admission and the executed specialization. Point-kernel timings alone cannot
  support a whole-method speed claim.
- Validate every observation against an independent molecular reference using
  unchanged method-appropriate energy and force gates. Record route fallbacks,
  UKS and response regressions separately rather than silently excluding them.
- Monitor representative LDA, GGA, meta-GGA, hybrid/range-separated,
  RKS and UKS cases in CI or finite Slurm runs, including compile resources,
  fallback rate and cold/warm/moved endpoints; never rebuild a method whitelist.
- Assess the complete endpoint reduction against
  `max(2%, 2*(relative_MAD_baseline + relative_MAD_candidate))` for a speedup
  claim and preserve failed cohorts. Do not pool changed sources or partial jobs.
- For WB97M-V include range-separated exchange and VV10; for r2SCAN-3c include
  its prescribed basis, D4 and gCP. Optimizing their semilocal point consumers
  is not qualification of these complete compositions or the other wB97 variants.

Run all real-device tests, profilers and sanitizers through Slurm with finite
time limits and the scheduler's unchanged device visibility. The durable design
and current evidence boundary are recorded in the
[equal-spin scalar sharing note](../../.agents/notes/implemented/performance/2026-10-11-equal-spin-xc-scalar-sharing.md).
The additional functional-binding decision is recorded in the
[native point binding note](../../.agents/notes/implemented/performance/2026-10-11-native-xc-functional-binding.md).
The registry-wide eligibility and compatibility boundaries are recorded in the
[functional-independent optimization note](../../.agents/notes/implemented/performance/2026-10-11-functional-independent-xc-optimization.md).
