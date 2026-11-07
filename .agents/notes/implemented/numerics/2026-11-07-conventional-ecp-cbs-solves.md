# Decision: safeguard cold physical references and condition the physical Z solve

Status: implemented
Date: 2026-11-07

## Problem

Issue #3034 supplies three small, closed-shell iodine-ECP examples. Two exhaust
the conventional CUDA correlated HF reference and one fails the CCSD(T)
physical orbital response. Independent RHF convergence does not qualify CC
energies or forces. The original input/output bundle is pinned under
`tests/data/conventional_ecp_cbs/` without basis/ECP truncation.

The physical-reference CUDA DIIS owner previously used an unnormalized Gram
matrix and an absolute `1e-14` elimination pivot cutoff. Small commutator errors
therefore became numerically "singular" even when their relative directions
still carried useful information. Falling back to the current Fock could leave
energy stationary while density and physical residuals remained unconverged.
For example, a source-resident diagnostic replay of `hf_dz` exhausted 200
iterations at energy `-442.99474918122735`, energy change `2.27e-12`, density RMS
`2.01e-7`, commutator RMS `2.54e-7`, and maximum commutator `2.60e-6`.

Normalization alone did not fix the second HF case (`hf_tz`). Its large initial
density moves led to a persistent two-cycle between roughly `112` and `-240`
hartree. Replaying the native normalized DIIS with independent libcint/PySCF
integrals reproduced that behavior: it was not an iodine integral error.
Bounded convex startup mixing, with dependent history retired on each mixed
step, resolved the cold-reference charge sloshing in the independent replay.

The conventional physical Z owner built and independently checked a dense
positive-definite orbital Hessian, but fed the original generated action to
restarted GMRES without preconditioning. A wide core/valence excitation scale
can exhaust that bounded solve despite an admissible stable reference.

Failure reporting also discarded scalar SCF diagnostics before reference
allocation and reduced response failures to one generic exception.

## Decision

- Physical-reference CUDA DIIS uses the existing scale-normalized affine solver
  and chronological dependent-history retirement. Ordinary HF keeps its
  existing schedule. This is a reference-owner policy, not iodine special-casing.
- Cold exported restricted physical references half-mix proposals whose undamped
  density RMS exceeds `0.1`, during at most the first 32 iterations. Mixing
  preserves trace and positivity and retires the affected system's DIIS history.
  Reported changes and all acceptance gates use the original undamped proposal;
  warm, ordinary, fitted and unrestricted references retain their schedules.
- Reuse the dead physical-Hessian column-construction vector for its diagonal
  and pass it as a right preconditioner. The already checked positive minimum
  curvature guarantees positive diagonals; no new inversion, dense factor,
  iteration-limit increase or response allocation is introduced.
- Retain scalar SCF diagnostics even when convergence fails, before allocating
  any dense reference matrices. Reduce the physical residual into a dead DIIS
  coefficient slot, using a scaled norm to avoid overflow/underflow.
- Preserve numerical/device failure status and report explicit solver stopping
  diagnostics. Rejected references/response solutions remain unpublished.

## Invariants

Energy, density, physical-Fock, canonical-reference, curvature, Lambda and
independent Z-residual checks retain their existing thresholds. GMRES still
accepts only its true **unpreconditioned** residual. The dense physical-action
oracle remains independent of the Krylov implementation. No CPU/PySCF production
fallback, shell truncation, ECP substitution or extra scalar-work allowance is
introduced. Failed asynchronous downloads drain before staging storage dies.

## Rejected alternatives

- Increasing iteration limits or relaxing density/response tolerances would
  hide numerical conditioning instead of resolving it.
- Treating tiny energy change as HF convergence would publish nonstationary
  orbitals. The recorded `hf_dz` plateau is a concrete counterexample.
- Level shifts of `0.5` and `1.0` hartree did not resolve the independently
  reproduced `hf_tz` cycle. Permanent damping unnecessarily penalizes the
  stationary tail; convex startup damping is bounded in both size and duration.
- A dense inverse is unnecessary when the existing Hessian diagonal resolves
  the scale separation, and would require a new numerical/resource contract.
- A sodium/boron auxiliary-g or DF-HF change cannot fix a conventional iodine
  ECP calculation; integral/fitting model boundaries must not be conflated.

## Evidence and qualification

Host tests cover failed-reference scalar retention, distinct failure status,
staging cleanup, and bounded true-residual diagonal-preconditioned GMRES.
Real-device contracts cover DIIS scale invariance and chronological retirement,
plus finite/extreme/nonfinite physical-residual diagnostics.

The independent-integral `hf_tz` replay with native DIIS and the bounded startup
policy converged in 23 iterations at `-370.79058492615263` hartree, density RMS
`8.57e-12` and maximum physical commutator `1.47e-10`; independent PySCF gave
`-370.7905849261543` hartree. RMS thresholds `0.05`, `0.2` and `0.3` also
converged (24, 29 and 29 iterations). These are diagnosis, not full native
endpoint qualification. The original physical orbital-response replay exhausted
200 iterations at true residual `7.6590096081880106e-5` against the requested
`2.3827302298009143e-9` absolute tolerance.

The opt-in issue regression isolates both reported HF steps and qualifies the
complete conventional CBS energies/forces for all three geometries against
independent PySCF component energies and two-step directional differences.
The archived RHF numbers alone are never treated as CC/force evidence. Real
device qualification must run under finite Slurm allocations; diagnostic
source-residency overrides must be identified separately from default routing.

### Native replay identity and completed observations

The qualification build is based on `9306f5f4c3928a571a9de2f4ace96d45a8065e1a`
plus this change. Its `libgenerativeqc.so` SHA-256 is
`2247f71761c0a6a9256ccacd42e0fe26a2f1942dfd8f14bff426668109930867`.
The following observations use finite Slurm allocations on RTX 5090 devices;
they are correctness evidence, not complete-endpoint performance claims:

- Both reported HF-stage tests pass with optional exact source residency enabled.
  Both default bounded-source replays also pass, retaining the original
  density/physical-reference gates and independent archived energy comparison.
- The `response_dz` native DZ CCSD(T) physical force stage completes. Its checked
  Hessian dimension is 1000, minimum curvature is `0.037776401801083014`, and the
  right-preconditioned physical orbital solve takes 16 iterations. This is not
  yet an independent comparison of its forces or a completed CBS endpoint.
- Nine existing real-CUDA independent-oracle regressions pass: H2 and water
  complete conventional CBS components/composites, and restricted/unrestricted
  Na-ECP conventional/fitted analytic components. This is not blanket coverage
  of all elements or all basis/ECP combinations.
- Focused host suites pass (255 tests with seven opt-in/optional skips, plus 315
  additional tests), as do the native GMRES, reference-export and method-execution
  suites, the real-device convergence/DIIS/startup contracts, and applicable
  repository pre-commit checks.

### Independent-reference qualification

The initial standalone native `response_dz` CCSD(T) energy/force endpoint returned
success under default bounded-source controls, but its subsequent independent
comparison stopped because PySCF's CCSD reference did not converge in 200 cycles.
At cycle 200 its amplitude-update norm was about `4.66e-8`, so a small energy
change was not accepted as a replacement for the `1e-10` amplitude gate.
The initial full-CBS replays were stopped after diagnosing this invalid oracle
configuration; they are not counted as completed energy/force qualification.

For the same pinned Bohr geometry, PySCF 2.14 converges in 915 cycles at unchanged
`1e-12` energy and `1e-10` amplitude-update thresholds, with CCSD total energy
`-793.5084509197476` hartree (before triples). The initial oracle was therefore
given a bound of 1200 cycles, with an independent replay of the final raw
amplitude update before using its energy. Native HF's 200 and
CCSD's 150 iteration limits, error budgets and acceptance gates are unchanged.
Shorter DIIS history did not solve the strict reference tail, and an undamped
fixed-point restart diverged; neither is used to manufacture a reference.

The first two-step finite-difference comparison was inconsistent: the `2e-4`
Bohr step agreed, but `6e-5` differed by `1.80e-6` hartree/Bohr. Do not loosen
the `3e-7` force gate to accept this. The original independent RHF orbitals had
true gradient norms of `5e-9` to `8e-9`; stable RHF energies alone did not make
them accurate enough for the nonvariational post-HF finite difference. Merely
calling PySCF's Newton driver did not refine these tiny gradients adequately,
even with a smaller augmented-Hessian dependence threshold.

An independent test-only refinement using PySCF orbital-Hessian actions and
normalized-right-hand-side SciPy GMRES reduces the true RHF gradients below
`1e-12`. With `1e-12` raw CCSD amplitude-update gates and scale-normalized
PySCF DIIS, this gives the same retained native physical CCSD(T) energy within
`9.22e-9` hartree. Force-projection errors are `1.34e-7` and `1.43e-7`
hartree/Bohr at `2e-4` and `6e-5` Bohr respectively, both below the unchanged
`3e-7` gate. The native net-force maximum is below `1.5e-13` hartree/Bohr.
This reuses the actual successful default bounded-source endpoint, library
SHA-256 `2247f71761c0a6a9256ccacd42e0fe26a2f1942dfd8f14bff426668109930867`,
not reconstructed or reference-injected native forces.

The permanent independent helper instead uses dimensionless affine PySCF DIIS
for both HF and CCSD: a common Gram scale preserves the affine equations and
avoids PySCF's absolute small-eigenvalue cutoff. It checks the final unscaled
HF gradient (`1e-11`) and raw CCSD update (`1e-12`) explicitly. The slightly
larger HF gate allows the few-`1e-12` double-precision floor in larger TZ
references; it is still 1000 times tighter than the initial oracle. Integral
screening is tightened to `1e-16`, and CCSD energy convergence to `1e-13`.
All changes are test-only; production does not call PySCF or SciPy.

The full pinned three-geometry CBS replays remain unqualified at this
checkpoint. Do not treat the qualified failed physical-response stage as
qualification of the larger complete endpoints.

### Optimized generic qualification build, November 8

The subsequent CUDA 12.9.1 / `120-real` build disables fast compilation while
retaining the generic integral/force consumers (shell and stationary AOT
products remain disabled). Its library SHA-256 is
`018c990f1ab7bced13328e5843211e155ef5f78c5dca4dec9f92907f60a8dead`.
The selected launcher is verified sccache 0.16.0, with before/after statistics
retained separately; cache use does not replace source provenance or device tests.
This is correctness evidence, not a release/performance qualification.

This binary passes the three native host suites, the real-device physical
convergence/startup contracts, and all nine existing independent-oracle CUDA
CBS/ECP regressions described above. Final-source focused Python tests pass
38 cases with six explicit GPU skips. The strict independent reference
preflight completes all three geometries, five displacement points and both
cardinals, including the explicit unscaled HF and CCSD update gates.
The three complete pinned CBS tests run under finite eight-hour Slurm
allocations, with unchanged 43-GiB caller budgets and no overridden device
visibility; their completion is not implied by these shorter checks.

All new permanent energy and force comparisons explicitly disable relative
tolerance. Pytest's default relative tolerance would otherwise dominate the
intended absolute energy gates for these large negative total energies. The
retained-output comparator already uses strict absolute differences and also
checks the net force of the complete CBS composite independently. The test-only
gate correction does not change the candidate CUDA library or the numerical
controls used by the running complete endpoints.

The first completed final-binary `hf_dz` replay on an RTX 5090 with 32607 MiB
device memory fails later in the TZ MP2 MO transformation with `std::bad_alloc`,
after the DZ group and both physical HF references complete. Optional source
residency is released and the bounded provider is attempted, but the complete
endpoint still cannot return on this allocation. Its caller's 43-GiB combined
numeric budget is not a reservation of 43 GiB of physical device memory. This
is an unsuccessful complete-endpoint qualification, not an SCF/response failure
or evidence of energy/force parity. Larger-memory RTX PRO 6000 replays retain
the same 43-GiB caller budget and scientific gates; default source selection
and explicit optional resident-source selection are recorded separately.

The final optimized generic library also passes the permanent isolated
`test_reported_physical_ccsdt_force_step` with the corrected strict absolute
gates and the permanent independent reference helper. This is a completed
real-device pytest node, not only posthoc reuse of an earlier endpoint.
Native energy is `-793.5709080104523` hartree versus the independent
`-793.5709080196682`; the native force projection is
`0.037302009445823237` hartree/Bohr. Independent projections are
`0.037301877853224141` and `0.037301878326919301` at the two original steps,
giving maximum absolute force error below `1.32e-7` hartree/Bohr. The
test also checks convergence, CUDA backend selection and zero net force.
This qualifies the reported physical response endpoint, not the pending
complete CBS replays.

Both permanent physical HF regression nodes also pass on this final library
with default bounded-source selection and strict absolute energy checks.
`hf_dz` takes 48 SCF iterations and `hf_tz` takes 27, within the original
200-iteration bound. Their tests explicitly retain the `1e-11` density and
`1e-8` physical-stationarity gates. The probe compiles through verified
sccache 0.16.0, supplied via the existing host-test fixture's `CCACHE` launcher
setting; an earlier run that lacked that setting skipped both probes and is
not counted as a pass.

### Final complete-endpoint qualification boundary, November 8

The full pinned CBS regressions are not qualified by this change's local
evidence. The final-binary RTX 5090 `hf_dz` run fails with the later MP2
allocation error described above. The other two 5090 full endpoints, and the
default-source and explicit-resident-source `hf_tz` replays on the RTX PRO 6000,
are deliberately stopped at `2026-11-08T05:02:48+08:00` during repeated TZ MP2
energy-provider scans. They did not reach their Slurm allocation timeouts and
must not be reported as passes, timeout failures, or new convergence failures.
The larger-device serial job never reaches its other two cases.

For example, completed larger-device provider calls each scan all `88529281`
AO values in 16 device-source reads, returning only 125952 output bytes. Those
individual calls take approximately 829--840 seconds and repeat while still
inside the MP2 energy phase. These are semantic work observations explaining
why the long qualifications are stopped, not complete-endpoint performance
results. The 5090 bounded-provider replays also remain in this phase without
returning a native complete-endpoint result. No MP2 performance, source-tiling
or resource-fallback policy is changed to manufacture endpoint qualification.

All three originally failing physical steps pass their permanent real-device
regressions on the final production library, including independent physical
CCSD(T) energy and two-step force checks. The complete CBS tests remain opt-in
and retain all scientific gates. Their physical HF probe now requires its
cached compiler when qualification is requested, so a missing launcher cannot
silently turn either HF regression into a skip. Raw logs and exact native
binaries are retained locally under ignored `.artifacts/issue-3034/`. No
complete-endpoint success or broader element coverage is implied.

### PR integration with current master, November 8

The independent issue branch is based on master
`d52d2ae2a` rather than retaining the already-merged auxiliary-g branch's
unsquashed history. The convergence safeguard composes with master's optional
256-thread restricted reduction: both its copy and damping decisions are
broadcast across the entire block. The existing 32-thread default and
incremental full-energy refinement transitions remain intact. Executable
startup tests cover both reduction widths, retained and advanced densities,
inactive neighbors, the finite startup boundary, and matrices spanning more
than one block-stride iteration. Existing host fault-injection fixtures are
adapted for the new diagnostic helper and physical-reference DIIS option;
their ownership and dispatch assertions remain in force.

On this integrated source, the focused host selection passes 121 tests with
18 explicit opt-in skips, and the source/ownership checks pass. The CUDA
ownership report records a net increase of 54 noncomment runtime lines and
no increase in handwritten scientific CUDA lines, with no new CUDA file or
semantic ownership reclassification. Real-device qualification of the
integrated source is distinct from the earlier binary's evidence above.

The first PR CI attempt also identifies the native Libxc catalog's checked
source-input digests as stale after these native-source edits. Regenerate
only that current identity/report: the retained campaign, measured binary,
scientific capability summary, acknowledgements, and already-stale admission
state remain unchanged. This is not a new Libxc numerical qualification.

## Revisit when

Matrix-free curvature admission replaces the dense physical Hessian, or a new
response preconditioner is independently qualified. Retain an explicit bounded
method and the original physical residual gates when changing that ownership.
