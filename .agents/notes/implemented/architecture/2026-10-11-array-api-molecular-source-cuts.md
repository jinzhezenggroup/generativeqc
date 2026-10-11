# Decision: bind complete native RHF endpoints into first-order Array API tapes

Status: implemented
Date: 2026-10-11

## Problem

The [native Array API frontend](2026-10-11-array-api-native-autodiff.md) qualified
tensor equations and density derivatives, not a complete molecular endpoint.
The next acceptance cut is coordinates -> converged energy + nuclear gradient,
reusing native scientific operators instead of duplicating HF in generic code.

## Decision

`RHF` is an immutable runtime configuration and explicit source boundary for the
existing native all-electron exact-J/K closed-shell driver. This is orchestration,
not a second MethodIR or a new set of HF equations. Native `ResolvedModel` and
the existing selector retain scientific method/basis ownership. The MethodIR
name catalog does not currently resolve `hf`/`rhf`; do not fabricate a MethodIR
lowering receipt for the native HF selector.

Capture binds the symbolic geometry Program and a first-order jet
`E0 + <g0, q-q0>` using canonical TensorIR inputs/primitives, with E0/g0/q0
nondifferentiable snapshots supplied by native execution. Canonical AD acts
before runtime binding; at execution the canonical DAG rebuilder substitutes
exact zero for the jet displacement at q=q0. Thus the actual native value E0 is
preserved even in nonlinear surrounding expressions, without recomputing a
potentially differently rounded displacement. The g0*dq chain-rule terms remain
in derivative graphs. No custom TensorIR primitive, scientific AD rule registry,
reference evaluator or differentiated SCF history is introduced.

Geometry may be a compiled tensor expression of original user inputs, but may
not depend on another molecular source. Identical source/config/geometry graphs
within one trace share one invocation. Different geometries always execute the
method; persistent caching is limited to native workspace and compiled algebra.
VJP records the forward geometry, native energy and analytic gradient, and its
later pullback never runs SCF again. Resource and dtype validation precedes source
execution; failed native calls publish no source diagnostics or successful result.

## Invariants and rejected alternatives

- A thin energy-only wrapper with finite-difference forces is not analytic AD.
- Density AD/Fock matrices cannot stand in for Pulay/nuclear derivatives.
- Reimplementing SCF/J/K/integrals in the public namespace would fork scientific
  ownership and forfeit existing optimized consumers.
- A local jet must not be evaluated as a surrogate at another geometry, or
  differentiated a second time while silently freezing its native gradient.
- Plain lowering must not execute SCF, and a lowered source tape is not a
  standalone molecular executable outside its explicit owner.
- Source backend/device is checked during capture. No CUDA-to-CPU/reference
  fallback is legal. The CPU native ABI's `cpu_reference` label is distinct from
  the explicit Array API reference interpreter.
- Scientifically annotated coordinates require complete, spin-independent
  atom/cartesian spaces. Equal AO/orbital extents cannot be rebound as nuclei;
  the coordinate domains survive native snapshots and first-order AD.
- TensorIR work/byte budgets do not account for native SCF/J/K work or resident
  memory; native ResourceBudget and finite iteration policy are independent.
- Retain at most 16 distinct sources per capture and 256 atoms per declaration.
  Source diagnostics store counters/last call, not an unbounded event history.

## Evidence

- Built the actual CPU library through verified ccache 4.5.1. Build log:
  `/tmp/gqc-array-api-rhf-build.log`; cache before/after statistics retained.
- `test_array_api_molecular.py`: 15 passed, one explicitly scheduled GPU skip at
  the first native gate. Covers no-native lowering, immutable settings, native
  algebra without reference interpretation, source deduplication, geometry
  transforms, nonlinear chain rules, JVP, mutation-safe reusable VJP, owner
  cleanup, inactive sources, finite/budget admission and native nonconvergence.
- H2/STO-3G, HeH+/def2-SVP and water/STO-3G complete native molecular gates agree
  with the existing endpoint (1e-11 energy / 1e-10 gradient), translational
  invariance (2e-9), and independent central energy differences (5e-7 absolute
  directional nuclear-gradient tolerance). H2 energy also agrees with the
  retained native suite oracle (-1.11671432506255 Hartree within 2e-9).
- CPU water complete endpoint, ten alternating warm repetitions: native median
  0.419341292 s, Array API median 0.420001372 s (ratio 1.00157409), zero reported
  energy/gradient difference, exactly twelve SCF iterations on every path.
  `/tmp/gqc-array-api-rhf-cpu-endpoint.json` retains cold time and raw samples.
  This is a bounded CPU observation, not a GPU or universal 5% guarantee.
- No allowed GPU was available: Slurm exposed only drained node3; no actual
  GPU test/benchmark ran. Opt-in tests and a scheduler-guarded complete endpoint
  benchmark are provided. Peak memory and complete exact-J/K work counters remain
  unmeasured; the script explicitly reports these qualification gaps.

## Qualification after master synchronization

The original #2268 was squash-merged while this follow-up was in progress. Rebase
only this new feature onto master `140160231f23d61cea6ce103ec5a46a0da915b31`,
retaining the merged wrapper-ownership/inactive-gradient fixes and structured
tensor routes. No remote history was force-pushed.

- Final molecular gate: 26 passed, one explicit real-CUDA skip.
- Broad native/frontend gate: 897 passed, three scheduled CUDA skips and 71
  subtests; the single warning is the existing eager empty-mean NumPy warning.
- Python 3.11.14 / NumPy 2.1.3 first-order native/molecular/boundary gate:
  46 passed, one scheduled CUDA skip.
- Official unfiltered upstream conformance remains 1377 passed / five upstream
  flaky skips on Python 3.13.9 / NumPy 2.5.1 and 1381 passed / the same five
  upstream skips on the minimum environment. Both use compat 1.15.0 and the
  previously pinned suite/spec commits, without library skip/xfail exclusions.
- Compiler structure: 521 modules, zero dependency errors. Public API contracts,
  Ruff lint/format, whitespace and Sphinx `-W --keep-going` pass.
- Rebuilt the synced native library with verified ccache and compiler commands
  showing its launcher. Synced cache statistics and build evidence are retained
  at `/tmp/gqc-array-api-rhf-synced-cache-{before,after}.txt` and
  `/tmp/gqc-array-api-rhf-synced-build.log`.
- Synced CPU water benchmark: native median 0.420349291 s, Array API median
  0.422072584 s, ratio 1.00409967 (approximately 0.41% overhead), zero reported
  energy/gradient differences, twelve iterations per call on both paths.
  The eleven generated calls (one cold plus ten warm) perform eleven force
  evaluations and 132 SCF iterations. Full samples, native source/model identity,
  precision diagnostics and qualification gaps are retained in
  `/tmp/gqc-array-api-rhf-synced-cpu-endpoint.json`.
- Final test/standard/doc logs use the `/tmp/gqc-array-api-rhf-qualified-*`
  prefix. No GPU ran; only drained node3 remains visible through Slurm.

## Revisit when

Add other native method sources only with independently qualified complete
derivatives and explicit contracts. Add resident GPU source/tensor ownership,
stationary implicit responses/higher derivatives and GPU timing/memory/work
evidence as separate steps. This does not yet let a user replace HF's electronic
functional or invent an arbitrary SCF/CC algorithm solely with Array API syntax.
