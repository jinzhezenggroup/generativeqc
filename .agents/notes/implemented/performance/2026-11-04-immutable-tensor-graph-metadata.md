# Decision: reuse immutable TensorIR graph metadata

Status: implemented
Date: 2026-11-04

## Problem and evidence

After bounded stationary cotangents fixed the Ni integral-block admission
failure, the complete low-level explicit-grid HVP on eb051d0d1 passed the frozen
independent moving-gradient gate. Its assembly nevertheless took 1875.44 s,
including 350.59 s for Coulomb moving-weight/first-integral contraction. Memory
boundedness alone does not bound aggregate work or establish useful throughput.

A bounded synthetic replay of 2000 signed shell cotangents took 4.177 s under
cProfile. Repeated graph order/content hashing consumed 2.134 s cumulatively;
debug naming consumed 1.831 s and liveness 1.171 s, with overlapping attribution.
These are synthetic diagnostics, not a molecular speedup measurement. Receipts
are `/tmp/gqc-2843-2849/weight-replay-uncached-metadata.{txt,prof}`.
The same profiled replay with retained metadata takes 1.092 s with the exact
same output checksum (-40.29706614632586). This still is not an endpoint timing.
TensorIR legality, replay, interpreter, canonicalization, forward/reverse AD,
precision and independent prepared-cotangent tests pass 211 cases in 43.69 s.
Compiled CPU tensor/Boolean/where/bundle/preparation contracts pass 88 cases in
64.26 s. Shared stationary/compiler/gradient integration passes 129 cases in
51.28 s; complete global-hybrid and WB97M-V direct/full-rank-DF RKS/UKS molecular
files pass 52 cases in 344.92 s with eight JUnit warnings, using the retained
warm cache outside pytest retention. These 480 source-checkout cases include
no real-GPU execution. The compiler structure check covers 586 modules with no
dependency errors. Complete Ni HVP/raw-matrix replay is evaluated separately.

## Subsequent complete spherical Ni HVP and lifecycle evidence

On ddb41feab the same explicit 5760-point NiH4(2-) singlet HVP is bitwise
identical to the accepted eb051d0d1 result. Normalized maximum errors against
three independent moving-grid gradient steps remain 2.6253e-8, 7.2917e-9 and
1.0926e-8; three independent raw-reference matrix actions give 3.5979e-7,
8.9713e-8 and 3.1502e-8. The frozen gate is 1e-5 Hartree/Bohr squared.
All eight physical sources occur exactly once, with one native response solve,
17 iterations / 34 actions and true residual 8.1108e-12. Assembly takes
1296.01 s. The 1316.24 s probe and 1048544 KiB peak include independent
reference setup; this is not an isolated production endpoint timing or a
compiler-cold measurement. An independent replay retaining the native source
arrays is also bitwise identical (1280.67 s assembly, 1300.98 s whole probe,
1055348 KiB peak including independent setup).

A fresh native state at a finite pi/6 rotation about z has energy parity
9.1e-13 Hartree and an explicitly verified bijection of this quadrature's point
set (maximum point error 1.37e-13 Bohr). The normalized native HVP covariance
error is 1.04e-13 and the three independently rotated gradient-reference errors
are at most 2.63e-8. This rotation is a discrete symmetry of the chosen
12-azimuth quadrature, not evidence for arbitrary grid orientation or rotational
zero modes at a nonstationary nuclear geometry. The full rotation probe takes
1306.41 s and 987840 KiB including independent reference setup. Two failed
atomic-guess reference attempts remain retained; the successful independent
minao-guess state uses the original convergence controls, not native density
seeding or relaxed gates. Stale geometry and the closed rotated owner reject.

Six executable B3LYP/PBE0 spherical lifecycle cases also pass in 12.11 s.
They cover same-AO-count basis changes, stale geometry, actual batch coordinate
replay and old-generation rejection, grid/output/response workspace failures,
injected late-source failure followed by clean replay, and exhausted SCF
rejection. Restoring the original geometry recovers its detached HVP within
2e-11. These tests preserve observed behavior, not internal source call counts.
Receipts are under `/tmp/gqc-2843-2849/`: the
`ni-hvp-graph-metadata-acceptance.json`, `ni-native-components.log`,
`ni-rotation-hvp-v3.log` and `global-hybrid-lifecycle-v4.{log,xml}` files.
Independent PySCF analytic-gradient differences also check six Ni source groups
separately: one electron, Coulomb, exact exchange, overlap/Pulay, combined XC
and nuclei. At steps 3e-3, 1e-3 and 3e-4, every normalized group error passes
the unchanged 1e-5 gate; the largest error is 2.57e-6. The combined XC check
does not independently distinguish its AO, grid-motion and weight subgroups.
At each displaced state the independent source gradients sum to the engine's
complete analytic gradient within 2e-10. All states converge independently;
only an independently solved center density seeds displaced states. Two failed
reference runs remain retained. The third uses DIIS space 12 instead of 8,
with unchanged energy/orbital-gradient tolerances and iteration limits; it
finishes in 31.80 s with 161972 KiB peak. No native density seeds or oracle
values enter production. The three-step source errors are retained in
`ni-components-acceptance.json`; finer-step component errors need not decrease
monotonically when independently converged SCF roundoff is differentiated.

The separate all-column native replay reaches its original 10800 s deadline
with 14 of 15 columns independently accepted, then exits 124. Across these
columns the maximum errors at the three reference steps are 2.1983e-6,
5.5001e-7 and 1.9838e-7; the maximum response residual is 2.0992e-10.
The 1095768 KiB process peak includes independent setup. The retained partial
array explicitly has `complete=False`; no complete native matrix is published.
Do not infer the missing column from symmetry, translation or the already
accepted HVP, or count this timeout as a complete raw-Hessian endpoint pass.

None of these receipts qualifies a complete native raw matrix, a default-grid
derivative, an installed/public Ni endpoint, or the 25-40-atom production gate.

After retaining the remote master integration 89c460644 without overwriting
its policy/evidence changes, rebuild the CPU library with verified sccache
0.16.0 and explicit four-worker compilation. The generated compiler commands
use that launcher; before/after statistics remain retained. The integrated
global-hybrid lifecycle/molecular and WB97M-V direct/full-rank-DF RKS/UKS suite
passes all 58 cases in 633.70 s (eight JUnit warnings), using the normal retained
cache with invalidated native-header artifacts rebuilt. All applicable hooks
pass, and 586 compiler modules have no dependency errors. This rerun does not
repeat the large Ni sweep or promote its preceding partial result.

## Decision

`Program` already freezes outputs, definitions, provenance and all Node inputs
and attributes, and already retains its logical hash. Retain its deterministic
node order, live-node tuple and immutable debug-name map for the same lifetime.
The public name property still returns a detached dictionary so caller edits
cannot affect later validation, precision naming or execution. Each new Program
gets independent metadata; no numeric feed, tensor value, geometry, shell
weight, live owner or compiled artifact is reused by this change.

The interpreter still validates every action's feeds, dtypes, shapes, finite
values and explicit retained-byte budget. Serialization, equation identity,
duplicate-node occurrence naming, dead-definition inspection and execution
order remain unchanged. Compiler provenance and native cache checks are not
overridden. Tests preserve observable replay, changed-feed behavior, detached
names and output independence rather than pinning internal cache call counts.

## Limits and rejected alternatives

Do not cache evaluated tensors or whole-molecule rank-four cotangents to hide
the metadata overhead. Do not omit per-action numeric validation. This change
does not remove scalar second-source lowering or primitive execution work and
does not qualify the installed/public/default-grid or production-scale endpoint.
The old raw-matrix probe passed its first independently checked column but was
intentionally interrupted before completion to avoid repeating known metadata
work; its partial columns and failed/limited earlier attempts remain retained.
