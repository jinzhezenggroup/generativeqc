# Production-default three-way benchmark

This is the standalone performance benchmark for **three actual user-facing
modes**, not a numerical-alignment campaign. It does not change production
settings or reinterpret historical matched-grid README results.

The only scientific inputs shared between all three arms are the molecule,
charge/spin, functional, exact orbital basis and direct/non-DF calculation.
The programs decide their own SCF convergence, initial guesses, screening,
quadrature, pruning, VV10 point domains and hardware execution optimizations.

| Engine | Public API | Numerical settings passed by benchmark |
| --- | --- | --- |
| GPU4PySCF default | gpu4pyscf.dft.rks.RKS(mol) | None (functional selected) |
| GenerativeQC Standard | generativeqc.Calculator(..., device="cuda") | None |
| GenerativeQC Fast | generativeqc.PrecisionCalculator(policy="fast", ...) | No manually chosen budget or SCF knobs |

Fast uses its **own published default** application error budget and may
execute mixed precision, strict FP64 or a measured strict fallback/audit.
The *entire* public Fast call is timed, and its validation status, internal
solve count, requested precision and selected controls are recorded. Calling
Calculator(precision="auto") would **not** represent this Fast mode.

The native hybrid/range-separated public API currently requires an explicit
KsOptions(grid=GridSpec()) because an implicit hybrid production grid policy
is not published. Both Standard and Fast receive that *native default
GridSpec()*; GPU4PySCF receives its own completely unmodified grid.
This exception is prominently recorded and is not represented as a pure
zero-argument native hybrid run.

Supported methods: LDA, PBE, PBE0, r2SCAN, **r2SCAN-3c** and WB97M-V.

For r2SCAN-3c, native Standard and Fast use the canonical selector, which
loads the pinned **def2-mTZVPP** basis, native D4 dispersion and gCP.
GPU4PySCF does not expose an ordinary built-in r2SCAN-3c method: this
benchmark uses its **unmodified default** R2SCAN electronic SCF/gradient
with the same canonical def2-mTZVPP basis, then adds independent CPU D4
and gCP energies/analytic gradients. Those corrections and their
preparation are included in the complete timed GPU4 E+F endpoint.
The exact independent reference was already used in
`benchmarks/readme_omol25.py::reference_correction`.

Therefore the r2SCAN-3c comparison is labeled "GPU4PySCF r2SCAN +
independent CPU D4/gCP", never a native GPU4PySCF r2SCAN-3c selector.
Only method-defined scientific inputs are shared: neither program
receives the other's grid, initial guess, density or SCF tolerances.

## Scope

All timings are **fresh-owner cold E+F**: after CUDA context initialization,
measure constructor, complete SCF energy, complete analytic force and
host export, with one new calculator per repetition. All three engines
run in separate child processes on the same assigned GPU. Differences in
Fock iteration counts and internal budgets are part of their defaults,
not normalized away. No unmeasured reference density enters a timing.

No default is forcibly changed, including GPU4PySCF's force
grid_response setting. GenerativeQC Standard/Fast produce moving-grid
responsive forces while GPU4PySCF normally leaves that response disabled.
This makes the main table a **stock end-user E+F latency comparison**,
not equivalent-derivative FLOP/precision evidence. The exact executed
force response setting is retained with the raw receipt.

Numerical/analytic force gates must be evaluated independently against
grid-converged references before claiming scientific agreement.
The performance summary is explicitly marked accuracy-unverified.
Failed SCF, partial force vectors, nonfinite energy/forces/timing, missing
repetitions, mismatched scientific inputs, incompatible force definitions
between native Standard/Fast, or mismatched GPU hardware are rejected.
The Fast summary includes its internal solve count, so a fallback is not
misreported as successful FP32 arithmetic. Do not reuse older matched-grid
96-atom numbers as production-default data.

## One-command run

Execute inside a finite Slurm GPU allocation, using the same device
and scientific case for all three arms:

    srun ... python -m benchmarks.production_default_compare all --method r2scan-3c --atoms 96 --repeats 3 --output .artifacts/benchmarks/r2scan3c-96-defaults.json

This launches three separate child processes, writes three JSON receipts
(next to the combined JSON) and prints **one performance main table** with
GPU4PySCF default, GenerativeQC Standard, GenerativeQC Fast and two speedups.

Or execute and combine individual arms:

    srun ... python -m benchmarks.production_default_compare gpu4pyscf --method pbe0 --atoms 96 --repeats 3 --output .artifacts/benchmarks/gpu4.json
    srun ... python -m benchmarks.production_default_compare standard --method pbe0 --atoms 96 --repeats 3 --output .artifacts/benchmarks/standard.json
    srun ... python -m benchmarks.production_default_compare fast --method pbe0 --atoms 96 --repeats 3 --output .artifacts/benchmarks/fast.json
    python -m benchmarks.production_default_compare compare --gpu4pyscf .artifacts/benchmarks/gpu4.json --standard .artifacts/benchmarks/standard.json --fast .artifacts/benchmarks/fast.json --output .artifacts/benchmarks/defaults.json

Gather individual method/atom comparisons into **the same single table**:

    python -m benchmarks.production_default_compare render --comparisons .artifacts/benchmarks/defaults.json --output .artifacts/benchmarks/main-table.md

This benchmark reports measured wall time, not a claim of identical
numerical tolerances, identical force definitions or identical Fock work.
The existing common-grid validation tool remains independent.

Agent: ChatGPT
Model: GPT-6
