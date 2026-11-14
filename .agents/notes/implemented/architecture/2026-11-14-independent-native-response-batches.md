# Decision: schedule independent native responses in explicit source batches

Status: implemented; complete heavy-element/installed/TS qualification remains separate
Date: 2026-11-14

## Problem

The [private scheduling proposal](../../proposed/2026-11-14-independent-rhs-native-xc-batching.md)
identified repeated AO-product work in independent native XC actions. Component
speed alone did not fix the timed-out whole Hessian. Sharing source execution
requires the actual response controller, orbital equations, nuclear maps and
aggregate admission to agree, rather than wrapping mutable callbacks in threads
or replacing fresh residuals with combinations of previous operator images.

## Decision

`native_gmres.cpp` has one explicit resumable numerical owner for both ordinary
single-RHS solves and cohorts of at most four independent solves. Only physical
operator requests suspend execution. Each RHS retains its own initial guess,
right preconditioner, Arnoldi/reorthogonalization/Givens/backsolve, best iterate,
restart, failure and stagnation state. Every requested true candidate residual
executes a new action. The ordinary `true_residual_every` policy is unchanged;
qualification continues to request one, relative zero and absolute 1e-12.

`LinearResponseProblem` validates the entire action batch before source work.
An ordinary operator retains a sequential fallback. There is no block-GMRES
equation, hidden thread rendezvous or alternate numerical controller. Controller
state is explicit C++ metadata, not an unpriced coroutine frame. All numerical
arrays retain their existing per-solve tracked allocation domains. Admission
includes simultaneous complete Krylov workspaces and retained solution/history/
basis capacities. Completed states release scratch before another cohort.

The shared spin orbital owner now has common density and Fock-projection maps
used by scalar and batch schedules. `HostKsResponse` retains ordinary sequential
prepared J/K builds while sharing bounded fixed-grid XC execution. RKS remains
the same two-spin singlet embedding and averaging; UKS keeps both coupled spin
channels. No new orbital/functional derivative algebra enters the adapter.

The XC owner accepts explicit caller-owned batch workspace. Its packed field
inputs place the current primal D before the independent tangents; every tile
re-packs them after the live reader returns. Pullback SIMD lanes are independent
directions, not reordered reduction terms. Portable scalar paths and masked
two/three-direction tails retain the same AO-pair, point and field orders.
No FMA/reassociation, field cache, density floor or new vacuum policy is added.
Output/workspace/source aliases and invalid capacities fail before source work.

`HostKsPerturbation` uses the same metric/RHS/D/W/physical-residual TensorIR maps
for scalar and batched solves. Additional per-direction map arenas and panels
are admitted explicitly. Metric and final-density actions remain ordinary
physical actions. A linear failure never exposes an unpublished orbital
solution to the D/W maps; a physical failure prevents the entire group's output
publication. Nuclear first sources retain their per-direction limits and exact
source work, with separately retained first-ECP work for each direction.

The full raw-matrix bridge selects a batch width before admitting XC partials
and the AO prefix. Nuclear source/result panels, all independent Krylov states,
provider panels and transactional assembly columns fit the original composed
response/aggregate caps. Assembly uses only existing output/aggregate headroom;
an exact original budget, insufficient batch capacity or optional allocation
failure retains the scalar column schedule. HVPs retain the scalar schedule.
Both schedules use the same physical column assembler, source-energy checks
and raw symmetry gate, with no symmetry/translation/mode repair. ECP and D3
receipts use actual per-direction diagnostic work rather than repeating the
last source's work across a batch.

## Rejected alternatives

- Independent thread stacks around the old callbacks violate borrowed mutable
  action-buffer ownership and hide simultaneous numerical payload.
- A separate batched GMRES implementation would split numerical ownership.
- Coroutines would require a compiler-specific frame admission/lifetime contract
  without improving these small explicit suspension points.
- Cached operator-image combinations cannot certify a freshly evaluated true
  physical residual, even for mathematically linear response equations.
- An AVX2-only/RKS-only production admission would discard required portable and
  coupled-spin behavior demonstrated by the private prototype.

## Evidence and limits

- 1,458 one-off original-versus-refactored scalar trajectories have identical
  solution/history/basis bits, status, iterations, actions and measured peaks.
  This is opt-in migration evidence, not a required source fingerprint test.
- Permanent native controller tests exercise widths one through four, empty and
  tail cohorts, known nonsymmetric dense solutions, independent failures,
  cancellation/recovery, fresh residuals, exact/short aggregate budgets and
  retained allocation lifetimes. The native ASan/UBSan/leak run passes.
- Ordered contraction/XC suites pass 554 tests, including scalar and SIMD paths,
  signed nonsymmetric inputs, extreme normal/subnormal ranges, masks, canaries,
  Libxc finite-difference controls, cross-spin response and cache failures.
- Fixed-geometry molecular action/solve/metric tests pass 26 tests. A new linked
  core also passes 57 public compact-endpoint and fixed-geometry tests, including
  raw matrix/HVP agreement, work caps, D3 receipts and failure recovery.
- Core source identity is
  `4879bbaf9f567b416a7d6ed70772ac8fddb7b059e4f324cba21aaa86fb197198`,
  SHA256 `e2742360141d9dfc1ada652ef5f694f22fdb437d1d7f49d3294726df7208c2d7`.
  Its baseline patch and full-byte roundtrip recovery are retained separately.
- A separate ASan/UBSan/leak executable passes RKS/UKS batches of widths one
  through four, source/output masks and canaries, nuclear D/W, complete ECP/D3
  columns, alias refusal, nonconvergence, reentry and recovery. The changed
  response/controller/column/bridge owners are instrumented; unchanged providers
  and other core objects are not. Loader paths, binary hashes and actual flags
  are retained in `host-independent-rhs-sanitizer-20261114`.
- The actual 81-AO, 87,458-point iodine SN2 stationary fixture completes all 18
  newly solved nuclear responses in 805.106047342 seconds, with maximum fresh
  physical residual 6.067583283002547e-13. Its complete D'/W' banks are bitwise
  equal to the independently accepted serial response banks; no saved bank is
  supplied to production. The original 128 MiB response and 512 MiB aggregate
  caps include simultaneous source/Krylov/assembly reservations and the right
  panels. The receipt records all 556 actions and five groups (4+4+4+4+2).
  This is response-only evidence, not Ni, frozen curvature, a complete Hessian,
  modes, an installed endpoint or a prediction of complete-request timing.
  Reproduction and input/core hashes are in
  `host-independent-rhs-all-coordinate-20261114`.
- Composed nuclear/full-column suites pass 39 tests in 2024.23 seconds,
  including independent displaced-energy controls, RKS/UKS complete raw columns,
  residual ECP and D3, width-one fallback, tails, failure publication and recovery.
  Their duration is a multi-test qualification process, not an endpoint timing.
  No private prototype timing is a timing of this production core. The old
  1800-second whole-matrix failure remains failed.
- A new complete iodine SN2 saved-frame physical request finishes all 18 columns
  in 1605.357080343 seconds under the unchanged 1800-second cap. It recomputes
  first sources, frozen curvature and every response/column without saved-bank
  injection, splicing, SCF/oracle restarts or raw symmetry repair. Independent
  fourth-order raw-H error is 1.0839430530235061e-8; raw asymmetry is
  9.948639134726989e-14 and translation residual 1.0957901253050295e-13.
  All 12 unscaled vibrational modes pass the frozen reference gates, with six
  rigid motions removed by the qualified mode analysis and one physical
  imaginary mode at -256.1859061430339 cm^-1. The native raw matrix is unchanged;
  symmetric eigensolver preparation happens only after its raw symmetry gate.
  This is an authenticated saved-frame Hessian/mode qualification, not a fresh
  installed SCF, optimizer or Ni-containing endpoint.
- Complete execution records 520 GMRES actions plus 36 physical metric/final-
  density actions, 226079504 first and 136320861 curvature ERI primitive visits,
  and all per-direction ECP/D3 work. The first checker wrongly equated solver
  actions with all physical Fock work; its negative receipt is retained.
  `host-independent-rhs-full-matrix-20261114/complete-validation.json` accepts
  the separately corrected accounting audit without changing any scientific
  gate or rerunning production. Earlier pre-column clarifications also retain
  the original checker and distinguish whole-owner fixed storage from the
  response-only cap and last-action from aggregate D3 work. A separately priced
  1 MiB receipt allowance remains inside the original output/aggregate caps.
  Peak process RSS is 532832 KiB; it is not claimed to be the numeric-payload cap.
- A subsequent ordinary CMake build/install links the same scientific source
  into a separate SDK prefix; no qualified core or old in-tree ELF is replaced.
  Only linking is needed, and the verified compiler-cache launcher/commands and
  before/after statistics are retained without claiming nonexistent cache hits.
  The installed core SHA256 is
  `57616f833679bab08a0bc50998be6d12ab5dbeb5ed41873843b5d766c19c49ff`.
  Source identity is unchanged; a 16.2 KiB patch plus all-byte recovery retains
  its exact SDK/RPATH identity separately from the first qualified core.
- Fresh installed Python/compiler-free CsH RKS and CsH(+) UKS CLI requests each
  complete one raw H and two HVPs using the requested canonical spherical
  def2-SVP/Cs46core/default-grid B3LYP-D3(BJ) model. Each command runs its own
  native SCF from the declared geometry; no checkpoint, oracle or old derivative
  is a production input. Whole process H/HVP0/HVP1 wall times are
  27.14/10.03/9.24 seconds for RKS and 54.29/14.52/14.53 for UKS. Peak RSS across
  their three processes is 95700/102996 KiB, respectively. The installed native
  RKS and UKS SCFs take 15/13 iterations and 17/15 Fock builds, respectively.
  Maximum physical response residual is 3.210577903706116e-13 for RKS and
  8.733684852562862e-13 for UKS, with relative zero and absolute 1e-12.
  Three sealed cold-gradient reference matrices, H@v/bilinearity, raw symmetry,
  translation, previous installed API agreement and actual work gates pass.
  References are authenticated existing independent controls, not newly run
  oracles. The initial UKS checker used the RKS follow-up archive layout;
  its KeyError/negative receipt is retained, while a distinct audit selects
  the original pinned UKS v2 archive identified by the existing installed audit.
- This SDK depends on the explicitly configured host OpenBLAS/OpenSSL/archive
  libraries. It is not a relocatable Python wheel, a stationary Cs mode audit,
  mixed/large-molecule qualification, a Ni reaction TS or universal speedup
  evidence. The two public ECP saddle guards are still closed.
  Remaining heavy-element/spin and stationary-mode qualification, remaining
  installed consumers/Python wheels,
  practical 25--40-atom endpoints and a representative native Ni saddle remain
  required.
  Public ECP saddle guards are unchanged.
- A subsequent genuine wheel is built normally with the reviewed bundled
  OpenBLAS configuration, repaired to `manylinux_2_35_x86_64` and installed by
  pip into a fresh minimal venv. This is not an old wheel with a substituted
  SDK core. The repaired wheel SHA256 is
  `a232881d8a8f64aa120a86e61dbd447a6c9e4ec9ad586aea905df0c6cdd892e0`;
  its installed core SHA256 is
  `64baa62a72d0e2cf8be8ba0342200136d141e8e42fe0e0b5717bd34ecae57225`.
  Scientific source identity remains `4879bbaf...`. The ordinary build takes
  1318.85 seconds; verified sccache 0.16.0 records 405 compiler requests,
  51 hits and 354 misses. All three earlier qualified cores remain unchanged.
- A fresh installed stationary-CsH API request completes its cold state,
  raw H, two HVPs and public unscaled mode analysis in 46.741441 seconds,
  below the predeclared 600-second whole-case cap. Its producer remains
  unaccepted; the separate `installed-cs-minimum-validation.json` accepts
  matching occupied roots/density/energy/gradient, stationarity, all three
  sealed independent raw matrices and mode normalization/mass overlaps.
  The single projected stretch is 855.0317758805603 cm^-1 with five geometric
  rigid motions; raw asymmetry/translation are 1.3878e-17/1.6959e-19.
  Installed Python/compiler-free native CLI H/two-HVP controls also pass,
  with whole process times 25.23/8.90/8.91 seconds. The initial qualifier
  incorrectly sent the CLI-only `b3lyp-d3(bj)-rks` name to the MethodIR resolver.
  That pre-SCF failure and executed driver remain retained; a distinct v2 uses
  the existing `B3LYP-D3(BJ)` resolver name and the unchanged physical graph.
- Rigid projection is not a claim that the raw finite-grid matrix has five
  zero eigenvalues. A separate raw rotational-action audit exposes the
  9.5728641e-5 residual and the two negative rotational curvatures before
  projection. Independent frozen stencils reproduce the action with errors
  3.6730e-8/4.0674e-9/3.6373e-10 at the three original steps. No raw repair or
  exact continuum-rotation certificate is implied. Attributing its cause to
  grid orientation would require a separate rotated-grid experiment.
- Three further fresh current-wheel API cases (Cs UKS, iodine RKS and iodine
  UKS) pass separate source/model/per-spin-root/raw-H/two-HVP audits against
  authenticated cold references. Whole cases take 89.657031/78.953891/
  285.884942 seconds, each below its predeclared 1200-second cap. Occupied
  root overlaps exceed 0.99999999999989; maximum raw-H errors across the
  three stencils are 5.2961e-7/1.3929e-6/5.6077e-6. These additional cases
  retain their prior rtol=1e-11/atol=1e-12 response policy and original 2e-10
  physical residual gate; the stationary-Cs API/CLI instead use rtol=0.
  No new oracle SCFs or external optimizers execute. These are discrete
  small installed consumers, not mixed Ni/ECP, practical 25--40-atom or
  Ni reaction-TS qualification. Both public ECP saddle guards remain closed.
- The same installed wheel subsequently completes the canonical 46-AO
  all-electron NiH3 doublet full raw H, two HVPs and supplemental cold orbital
  export in 783.767272 seconds (predeclared cap 3600 seconds). The producer
  stays unaccepted; the existing independent installed-Hessian validator and
  a separate per-spin root/metric audit both accept its retained outputs.
  Native H/HVP0/HVP1 times are 400.283577/161.220848/161.170364 seconds;
  peak process RSS is 234428 KiB. Both occupied-root overlaps exceed
  0.99999999999989. Three authenticated independent-stencil matrix errors
  are 6.2451e-10/2.8448e-9/6.9993e-9. Raw symmetry/translation remain
  4.2626e-11/4.6683e-11 without repair. The 12-column H reports 572 actual
  induced Fock actions and first/curvature ERI visits 13851057/14379765.
  Its retained rtol=1e-11 response has physical residual 1.5456e-10, below
  the original 2e-10 gate, not below the stationary-Cs absolute-only gate.
  This is nonstationary all-electron NiH3, not a stationary mixed Ni/ECP
  molecule, a Ni reaction TS or practical large-molecule qualification.

Numeric budgets still exclude borrowed stationary/provider storage, BLAS
internals, scalar/control metadata, allocator overhead and RSS, as their existing
contracts state. Per-RHS operator timing in a shared batch includes shared wall
time and must not be summed as independent endpoint time.

## Revisit when

Complete source-matched endpoint qualification establishes whether this schedule
is sufficient. Further changes must preserve independent residual certification,
bounded scalar fallback and common scientific/numerical ownership.
