# Decision: compose native host CPKS from the shared spin action and SCF source

Status: implemented (optional fixed-geometry response only; no endpoint admission)
Date: 2026-11-12

## Problem

The linked scalar and bounded XC consumers do not solve coupled orbital
response. A Python-free CPU stationary owner needs native CPKS, but another
named-B3LYP Jacobian or a reference-engine production fallback would split the
scientific ownership already present in the compiler and shared response code.
The earlier XC consumer also rejected exact vacuum points even though its
linked production graph already has a qualified lazy vacuum branch.

## Decision

Add `HostKsResponse`, separately linked to the ordinary native CPU core and the
explicit host leaf package. Borrow its actual immutable `PreparedFockPlan`;
reuse `spin_orbital_problem`, `assemble_fock` and native GMRES. The adapter owns
AO/response panels, not a second integral provider or SCF implementation.

The compiler publishes an exact MethodIR-to-native-model binding only when
both XC and coefficient-response programs are linked and the generated native
family metadata matches the exact semilocal composition and full-range exchange
fraction. A display name is not a binding. Native construction checks CPU FP64
exact J/K, functional/domain/spin, canonical physical hybrid coefficients,
system identity, grid point count/tile identity and finite, resolved orbital
gaps. Range/nonlocal corrections and different models fail before actions.

RKS embeds its one orbital direction in the singlet sector of the common
two-spin problem. Each induced spin density has occupation one; their sum
supplies the ordinary restricted total-density J/K/XC source. The two equal
spin outputs are averaged back into the restricted orbital sector. The
synthetic occupation-one frame deliberately carries no occupation-two weighted
density: the shared orbital operator does not consume W. UKS uses the full
coupled action directly, including unlike-spin Hartree/XC response.

Adapter/XC persistent numeric panels, a conservative shared-action array bound
and GMRES workspace share one explicit admission budget. An action can fit
while a solve cannot. That case returns workspace refusal without allocating
Krylov output, including an exactly zero remaining solve budget. Borrowed
SCF/grid/Fock data, provider/BLAS internals, scalar stack, metadata and allocator
overhead are outside this bound and must be admitted by a future complete owner.

Owner validation authenticates the exact native KS identity and solve epoch
around actions/tile reads and before publication. Frame pointer/shape changes
also fail closed. All published vectors are transactional; nonconverged solves
leave the caller's solution unchanged. Output may alias the input vector but
cannot overlap borrowed S/H/C/epsilon/D/F/W. Serialized invocation guards reject
reentrancy and recover after failures.

## Production vacuum semantics

This extends, rather than relabels, the earlier positive-density XC cohort in
[the fixed-grid note](2026-11-12-bounded-host-xc-response.md). Admit nonnegative
spin densities with compatible zero gradients. Let the **unchanged linked
production Graph** select its lazy vacuum branch, including zero derivatives;
do not replicate its threshold in a C++ method branch. Do not evaluate inactive
inverse powers, add a density/tangent floor, or replace nonfinite non-vacuum
spin-endpoint Hessians with zero. Invalid/nonfinite sampled points still fail
transactionally. The frozen production policy used by the live SiH3 saddle
control is unchanged.

## Evidence

The first complete focused run exited zero: **60 passed / 77.09 s**, retained at
`build/qualification/host-ks-response-20261112/focused-second.{log,xml}`. It
includes actual cold native SCF and shared canonical final-state validation for
H2 RKS and asymmetric H3 doublet UKS, each with two independent normalized s
Gaussians per H and the ordinary default molecular grid. These are development
controls, not representative Ni/ECP or complete Hessian qualification.

Independent PySCF one-electron/ERI and Libxc first-potential finite differences
check the exact exported native frame/gauge/grid. Three direction steps use
`1e-3`, `3e-4`, `1e-4`; the complete independent orbital matrix uses `1e-4`.
Action/matrix gates are `atol=2e-8, rtol=3e-7`; native bilinear symmetry is
`3e-12`. Native GMRES is checked against both native dense and independent
dense solves, with fresh physical residual below `2e-11`. No oracle density,
orbitals or SCF seed enters the native solve.

Executable tests cover model/grid/gap failures, missing identities, incomplete
linked closures, exactly fitting and one-byte-short composed numeric budgets,
workspace refusal, nonconvergence, expired owner/epoch, reader/nonfinite failures,
reentrancy, alias safety and subsequent recovery. Vacuum tests retain signed
weights, exact zero and positive-tail densities and mixed active/vacuum tiles.
The optional CMake target runs both cold native SCF/CPKS controls in a linked
executable with no Python on PATH; missing core/leaves and CUDA configurations
are refused. Compiler use remains verified sccache 0.16.0.

Initial qualification failures are retained separately: test-probe status-enum
spelling, mismatched SCF screening provenance, zero (invalid) user GMRES options,
and a misplaced Python test tail. Their corrected reruns do not rewrite those
logs or older accepted evidence. No scientific gate was relaxed.

### Final source-cohort qualification

The final broader run exited zero: **273 passed, 2 explicit expensive saddle
oracle skips / 101.68 s** in `broad-final.{log,xml}`. This includes independent
cold PySCF SCF on the exact exported grid (no cross-seeding), pointer-replaced
frame rejection, failed-solve transactional publication and recovery, in
addition to the earlier focused controls. Both SCF density and energy comparison
gates are `3e-9`; the RKS/UKS maximum density errors were respectively
`4.5353e-14` / `2.0820e-10`, with energy errors `1.8874e-14` / `1.7764e-15`.
The complete independent orbital-matrix maximum errors were
`1.0924e-11` / `2.9908e-11`. Native relative solve residuals were
`1.4643e-15` / `1.0903e-11`, below the unchanged `2e-11` gate. Exact observations
and semantic work censuses are retained as JUnit suite properties, not pinned
as implementation-specific required-CI call counts.

Final changed-file pre-commit and the repository's actual documentation CI
command (`sphinx -W --keep-going -b html`) exited zero. An additional, non-CI
`-n` nitpick diagnostic reported 331 unresolved type-reference warnings (for
example existing `numpy.ndarray` references); its failed log is retained as
`sphinx.log`, not reclassified as success. No required documentation check was
weakened. Root CPU configuration accepts both optional targets; the root scalar
archive rebuilt successfully with verified sccache invocations. The coupled
CMake test builds its static archive and executes it against the existing CPU
core library with SHA `afb07c88c86bcfa2959cfa35bfcc138266c0d6fed8fe5abe16cac02b10ac0ec9`.
This is explicitly **not** a newly rebuilt/installed complete native-core or
CPU Hessian/TS endpoint qualification. Local receipts/recovery live under
`build/qualification/host-ks-response-20261112/` and
`build/qualification/host-ks-response-recovery-20261112/`.

## Rejected alternatives and boundaries

- Do not introduce a restricted/hybrid-specific Jacobian, hidden SCF, Libxc
  production call, dense molecular finite-difference Hessian or Python bridge.
- Do not relabel existing Cartesian RHF integral AD as a spherical hybrid/ECP
  nuclear-response provider. Its domain and ownership differ.
- Do not claim provider workspace or complete RSS is bounded by an adapter's
  array accounting. Streaming XC still rereads the full grid for every action.
- Do not link this development consumer into public CPU Hessian/TS capability
  admission. Nuclear RHS, metric/Pulay, moving grid/integral/ECP derivatives and
  full stationary assembly remain unresolved. So do the installed/current-source,
  representative Ni/ECP, mixed-element, scale and positive native TS gates.

## Revisit when

The subsequent [metric/field response decision](2026-11-12-native-host-metric-field-response.md)
adds compiler-owned overlap/RHS/D/W composition. It does not supersede this
cohort's frozen evidence or complete the nuclear-source/endpoint requirements.

A native stationary owner composes all derivative/provider resource lifetimes,
an exact nuclear perturbation source and complete endpoint acceptance. Any
extension of point domain or retained response storage needs its own
independent oracle, exact model identity and bounded fallback policy.
