# Compare independent production defaults

This is a separate **user-facing default performance** measurement. It does not
replace the README's matched-grid numerical-implementation test or the
independent accuracy-qualification procedure in
[performance engineering](performance_engineering.md). No published speedup
may be inferred from a ratio alone.

## Protocol

Shared inputs: water-3/6/12/24/48/96 nested geometry, charge/spin, exact
DFT functional, spherical def2-SVP and direct Coulomb/exchange. Each program
uses **its own** default initial guess, SCF convergence controls, J/K
screening, XC grid, pruning, density filtering and acceleration strategy.

Supported electronic functionals: LDA, PBE, PBE0, r2SCAN and WB97M-V.
The native API currently requires an explicit GridSpec() for PBE0 and
WB97M-V because no qualified automatic hybrid grid exists. This is
recorded as a mandatory API input; it is *not* claimed to be a fully
zero-configuration native hybrid run. r2SCAN-3c is not silently reduced
to r2SCAN: its D4 and gCP corrections require an explicit complete-model
reference and separate provenance before a defaults comparison is valid.

The timed fresh-owner endpoint includes public constructor, grid and integral
preparation, first cold SCF, analytic force and host result export, starting
with an already initialized CUDA context. Each repetition uses a new owner.
It does **not** claim process-startup/compilation cold timing, imported-density
warm timing, or automatically matched Fock/JK work. Measurements run in
separate engine processes; CPU/GPU version, grid details, source identity,
SCF iterations and failure status are recorded, never reconstructed.

## Commands

Run both commands on the **same GPU**, separately inside finite Slurm
allocations; repeat with independent process launches to assess cold-context
variance if needed:

    srun ... python -m benchmarks.production_default_compare generativeqc --method pbe0 --atoms 96 --repeats 3 --output .artifacts/benchmarks/native-default.json
    srun ... python -m benchmarks.production_default_compare gpu4pyscf --method pbe0 --atoms 96 --repeats 3 --output .artifacts/benchmarks/gpu4-default.json

Combine on any host (no GPU needed):

    python -m benchmarks.production_default_compare compare --native .artifacts/benchmarks/native-default.json --reference .artifacts/benchmarks/gpu4-default.json --output .artifacts/benchmarks/default-ratio.json

For true software defaults, the force policy is engine-default.
Native returns full moving-grid-response analytic forces, while GPU4PySCF
normally leaves grid_response disabled. Thus the default E+F ratio is a
**user-workflow runtime comparison for different force definitions**, not
a speed comparison of equal derivatives. GPU4PySCF's actual gradient
grid_response value is captured for every sample, rather than assumed.

To compare the cost of *complete moving-grid forces*, run both engine
commands with --force-policy full-moving-grid. This explicitly overrides
GPU4PySCF's usual gradient response setting, but never copies SCF defaults
or the native quadrature. Keep the two force policies in separate tables.

An independent grid-converged oracle should separately verify total energy,
maximum force component and moved-geometry force behavior for each program.
The summary is unconditionally marked **accuracy unverified** until such
evidence is supplied. Default-vs-default energy disagreement is not an
independent oracle. Mismatched scientific inputs, missing repetitions, or
failed SCF prohibit a timing comparison. No pre-existing matched-grid
96-atom result is relabelled as a production-default measurement.

Agent: ChatGPT
Model: GPT-6
