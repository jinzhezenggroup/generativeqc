# Proposal: Device-controlled CPKS in the native semilocal Hessian

Status: proposed
Date: 2026-11-22

## Original boundary

For #1905's Direct all-electron Cartesian FP64 LDA/PBE RKS endpoints, #3132
made the scientific panels resident, not the iteration controller. The linked
GMRES reads `State::code` to the host after every true-residual evaluation.
Moving just that check into a conditional CUDA graph would still be insufficient:

- The matrix-free CPKS action reaches `CudaKsWb97mvResponse::induced_fock`.
  Its checked `apply` synchronizes and has an immediate-output contract. The
  standalone XC grid response reads status after feature generation and after
  assembling the potential. The induced Fock reads both audit flags and the
  Direct work census before publishing an action.
- The retained FP64 operator matvec does not read back tensors or audit scalars,
  but its host callback counter counts enqueues, not graph replay. Automatically
  making all solves depend on this optional matrix is not an acceptable fix: the
  matrix-free fallback is intentional and the matrix can exceed the budget.

## Proposed ordering

1. Give the existing grid and induced Fock owners an enqueue-only composition
   path with the same scientific kernels. Consume finite/symmetry/status gates
   on device before a result can enter the Krylov solve. Move per-action census
   bounds and overflow checks into the device path and account for all replays,
   including exact recalculations of the residual. Preserve the checked public
   action and host-side error diagnostics at their transaction boundaries.
2. Drive the same bounded restarted GMRES and its true, unpreconditioned residual
   gate from the device. Evaluate conditional CUDA graphs with caller-owned
   scratch; use stable action buffers and a bounded graph independent of the
   requested iteration cap. Check capture of both retained and matrix-free
   actions, including the Direct fallback without resident ERI storage. Trace
   transfer bytes rather than inferring them from device pointer types.
3. Authenticate state/grid/provider before and after execution, ensure borrowed
   workspace is drained on failure, and verify failed actions cannot modify
   public outputs or poison valid replay. Reuse the prepared response context in
   dense assembly. Qualify complete mixed/native cold, warm and changed-geometry
   endpoints and their simultaneous live device peaks before claiming promotion.

Unconditional replay to the iteration cap would avoid host reads but can
substantially increase induced-Fock work despite early convergence. Checking only
the projected GMRES residual or suppressing audit transfers without consuming
their flags would change the numerical contract. Neither is an acceptable
substitute for the device-controlled loop.

## Implemented execution slices

The induced Fock now uses the shared grid enqueue path with a private output
panel. The grid's input and output audits remain device-side; a failed audit
poisons the private panel, which the enclosing Fock transaction rejects before
public output can change. Standalone grid calls keep their checked publication
contract. The Fock consumer keeps distinct failure flags for tensor maps and
each Direct source, audits the per-action census on device, and reads aggregate
counts only before outer publication. Work counts come from execution, not graph
recording; rejected candidates cannot advance the accepted counts.

The generic CUDA GMRES owner records one iteration into a conditional WHILE
node with stable provider operand addresses. Each iteration still evaluates the
actual unpreconditioned residual. It only replays if the device says to continue
within the admitted iteration budget. The ordered MGS reduction and updates are
fused within one block to avoid a graph whose node inventory grows with restart
length. It uses CUDA 12.9 whole-program compilation without changing device link
policy. The host still reads one setup status and a terminal status/diagnostic
receipt in the checked call; no iteration transfers are needed by this solver
or its Fock action. Dense Direct Cartesian FP64 RKS instead binds a reusable
unseeded graph to the same private solver arena. A predecessor kernel seeds
the loop condition from the initialized state so a failed or already converged
RHS never enters the WHILE body. The graph is scoped to the dense transaction
and all captured objects drain before it is destroyed. The device receipt
drives true physical residual validation, neutral private response panels on
failure, and one outer column/publication audit. The retained cache has only
its one-time admission checkpoints; matrix-free execution uses the same graph
control without a cache. FP32 proposals keep the checked path.

The left-coordinate geometry and XC mixed contractions reuse the checked
provider arithmetic but enqueue into private device views and scalars.
Their audits feed the enclosing device assembly status. Standalone calls
retain their checked, transactional semantics; a private failure never
authorizes publication or a persistent view.

`CudaKsResidentGridBinding` authenticates the canonical quadrature, basis,
accepted density and native atomic measure/topology. A successful force
publication can also lease its read-only Becke center-pair table. Cold calls
without compatible force storage use bounded recomputation; they never create
a force owner just to populate a cache.

The existing force AO map and panels have no matching Hessian identity:

- `ResidentAoMap` in `cuda_grid.cu` keys the thresholded tile union by point
  span, geometry epoch, derivative order, tile size and force plan identity.
  The force owner requests one spatial derivative for LDA and two for PBE;
  those screened unions do not certify the order-three jet domain used by
  `CudaGridGeometryResponse`. SCF can additionally use a private point
  permutation, whereas the mixed geometry stage uses canonical atom order.
- `GridTaskView` is a borrowed current tile, not a persistent collocation or
  feature census. The producer reuses that storage on the next tile or density
  update, and a force call does not publish a full reference panel. Its
  unperturbed feature layout also cannot supply the left, right, mixed or
  CPKS induced features. The Hessian reuses the authenticated **density matrix**
  and retains/replays its own order-three collocation under its grid budget.

An improved cache must match the derivative order and point/AO layout and
survive the consumer's admitted lifetime. Sharing a smaller-order mask or
retaining the mutable force arena would not satisfy these conditions.

The historical mixed baseline is `rks_hvp()` / `rks_hessian()` with both
integral backends set to CUDA and an explicit compiler adapter. Its shared
plan requires a CPU stationary state; passing a CUDA SCF changes the contract
and is rejected. Complete timings must include each path's actual primal SCF
and preparation. Warm and moved repeats here reuse the process and artifacts,
not an SCF warm seed.

## Evidence

- Slurm `19184` passed a standalone same-stream graph replay with rejected,
  restarted, and zero-iteration RHSs and one final device receipt collection.
  Slurm `2336` passed all 149 staged component cases with the correct binaries;
  its separate build-generation check needed to run from the source tree.
- Slurm `2338` (`cc8259c4eedf16d4…`) passed 11 ledger, nonconvergence/replay,
  cutoff, and optional-cache fallback tests. PBE water-dimer dense response
  D2H fell from 97 calls / 4,600 B to 26 calls / 2,956 B; H2D stayed at 36
  calls / 8,293,300 B. The output changed by at most 6.11e-15 Ha/Bohr² and
  Direct, grid, one-electron and CPKS work counts were unchanged. HVP had
  32 calls / 492 B D2H, versus 32 / 516 B. Standalone GMRES copies just its
  diagnostic prefix, not all internal Krylov scalars. The remaining dense
  reads should be checked against the exact driver-level trace before closing
  the transfer gate.
- Slurm `1803` ran matched LDA water-dimer HVP on `83272daed49f0ef…`.
  Native cold/warm/moved were 1.362/0.327/0.331 s versus mixed
  797.174/303.444/302.865 s, with remaining JIT compilation in the first
  mixed call. Matched values differ by at most 2.64e-12 Ha/Bohr². Mixed used
  28 response iterations and 12/14 target SCF iterations/Fock builds on the
  first geometry, 13/15 on the moved geometry; native used 11 response
  iterations. This is the intermediate HVP comparison, not final dense
  qualification. Slurm `1804` runs matched compact-water dense endpoints for
  LDA and PBE on `cc8259c4eedf16d4…`.
- Slurm `2333` merged the cached CPKS physical and reconstructed-output
  checks into one receipt, preserving rejection before publication and cache
  revocation when the physical residual fails. The full CPKS module (116 tests)
  and nine ledger/cutoff probes passed. In the PBE water-dimer response, dense
  reads fell from 133 to 97 with bytes unchanged (4,600); HVP is unchanged.
  For `f9eaddcc2d51d676…`, outputs changed by at most 6.22e-15 / 8.66e-15
  Ha/Bohr² (HVP/dense) versus `2331`, with equal semantic work counts.
  This does not remove the per-column CPKS/GMRES setup or terminal receipts.
- Slurm `2332` passed four public C ABI HVP/dense cases with a genuinely
  unavailable device XC primitive: a strict, converged CUDA SCF explicitly
  uses host-unfused XC. Both LDA and PBE reject with `NOT_IMPLEMENTED`, leave
  the caller's output/receipt and ledger unchanged, and then publish on the
  valid device-fused state. This is separate from stale-token rejection.
- Slurm `2331` (`99cebbb32a16af6b…`) passed 34 Direct J/K reference and
  fallback checks plus 11 public live-peak and native ledger/replay checks.
  Private frozen Direct J/K submission reduced dense response D2H from 168
  to 133 calls (4,600 B); HVP stayed at 32 calls (516 B). H2D and exact
  primitive work were unchanged (516,132 HVP / 9,290,376 dense primitive
  components). Non-resident exact execution still uses a checked path.
- Slurm `2330` (`4002fbd3…`) deferred dense column-end statuses to one
  device audit and passed nine ledger/replay/cutoff checks and all four
  public LDA/PBE water-dimer oracle and live-peak checks. PBE response calls
  were 37/36 H2D and 32/168 D2H (HVP/dense); directions and final values are
  included. The remaining dense receipts still need to be accounted for.
- Slurm `1802` compared the deferred-right PBE water-dimer HVP to the mixed
  path on the same RTX 5090. Cold/warm/moved end-to-end seconds were
  1.386/0.374/0.359 native versus 810.763/317.967/318.467 mixed. The first
  mixed call includes remaining JIT work, so use warm/moved to characterize
  steady artifact replay. The mixed path used 12 target SCF iterations / 14
  Fock builds and 28 response iterations; native used 11 response iterations.
  Matched values agreed within 3.26e-12 Ha/Bohr² across all three phases.
  This is one PBE HVP fixture on an intermediate build, not a matched gate
  for LDA, dense, or the final implementation.
- Slurm `2323` passed all 27 one-electron checks and all eight Hessian
  budget/replay checks on the deferred-right library (`83272daed49f0ef…`).
  The earlier probe rejected a short *parallel* arena even though the owner
  correctly selected the bounded scalar schedule. The probe now rejects below
  the scalar minimum and separately tests parity with exactly that minimum.
  The same build passed 34 focused moving-grid tests in `2322`. PBE water-dimer
  dense ran in 3.055 / 1.940 / 1.929 s (cold / warm / moved) without tracing;
  these are native-only figures, not a matched performance claim.
- Slurm `2325` used the same library and corrected the profiling wrapper to
  propagate the linked `ctypes` signature. Without it, the dense trace could
  fail ABI argument admission before any CUDA transfer; the untraced endpoint
  and both `2321` molecular oracles were unaffected. Bounded descriptor staging
  reduced H2D calls from 836 to 37 (HVP) and 835 to 36 (dense), with unchanged
  byte counts. Deferred one-electron and right-grid checks reduced D2H to
  34 calls / 524 B (HVP) and 240 calls / 5,088 B (dense). This remains a linked
  CUDA Runtime API census, not proof of transfer closure. Dense still reads
  CPKS phase gates and terminal GMRES receipts per column, checks the exact
  J/K cache during setup, and reads several Hessian assembly statuses.
- Slurm `2326` tested a later assembly build (`870250a2f8d14711…`). A dense unit
  direction no longer needs a separate host check, and the post-CPKS weight
  reduction feeds the final sticky assembly audit. D2H fell further to 33
  calls / 520 B for HVP and 204 calls / 4,944 B for dense. The traced outputs
  changed by at most `7.78e-15` / `8.89e-15` Ha/Bohr² relative to `2325`; all
  eight ledger/replay checks and both independent LDA/PBE water-dimer gates
  passed. Slurm `2328` used Nsight Systems 2025.1.3 with CUDA graph node tracing
  to confirm these H2D/D2H counts and bytes in `CUPTI_ACTIVITY_KIND_MEMCPY`.
  This trace also exposes 92,354,832 B (HVP) / 1,662,424,528 B (dense) of
  device-to-device copy traffic, not device allocation peaks.
- Slurm `2327` folded assembled RHS status into the existing CPKS setup
  checkpoint (`46388b09860f71ce…`), removing a separate transfer per direction.
  It passed 92 native CPKS tests, including poisoned-audit rejection and valid
  same-workspace replay, and eight Hessian ledger/replay tests. The linked
  Runtime census read back 32 calls / 516 B for HVP and 186 calls / 4,872 B for
  dense. Slurm `2329` measured water-dimer aggregate arena-ledger peaks of
  77,073,749 / 77,122,485 B (LDA HVP / dense) and 77,172,053 / 77,220,789 B
  (PBE). Those peaks include the live prepared SCF allocations and both public
  response modes, not CUDA contexts or unrelated GPU allocations.
- Slurm `2316` (RTX 5090) passed 158 combined tests, including both public
  Cartesian water-dimer endpoints and the full generic CPKS and GMRES test
  selections. This build preceded the final grid action-census correction;
  it cannot qualify the corrected receipt. Slurm `2317` uses the exact
  post-correction library and passed both molecular oracle tests.
- Slurm `2318` passed eight C ABI ledger/replay tests. The probe exercises the
  same Direct Cartesian LDA/PBE water
  state across rejection and replay. It checks that an undersized result
  budget does not allocate device storage and that a short CPKS solve cannot
  publish a partial result or leak live allocations. The response and SCF
  reservations are observed concurrently: aggregate peaks for LDA were
  6,488,544 B (HVP) and 6,494,792 B (dense), and for PBE 6,502,368 B (HVP)
  and 6,508,616 B (dense). These are arena-ledger peaks for the compact
  water fixture, not GPU process memory or the larger water dimer.
- Slurm `2319` interposed the linked CUDA Runtime copy calls between accepted
  SCF and native response return. PBE water-dimer HVP made 836 H2D calls
  (8,293,444 B) and 99 D2H calls (784 B); dense made 835 H2D calls
  (8,293,300 B) and 1,410 D2H calls (9,768 B). This is an API-call census,
  not a driver-level trace. The 144 B direction upload and final HVP/dense
  publication (144 / 2,592 B) are included. The extra D2H is setup and many
  small scalar gates, not downloaded intermediate panels. Symbolization
  attributes 648 of dense's D2H calls to the two checks in the XC mixed
  contraction and 342 to the geometry action. One-electron setup alone
  submitted 800 520-byte H2D copies. These numbers describe the build before
  deferring the mixed XC status to the outer Hessian transaction.
- Slurm `2320` repeated the same census after deferring both geometry and
  mixed-XC audits. H2D was unchanged; D2H fell from 99 to 45 (784 to 568 B)
  for HVP and from 1,410 to 438 (9,768 to 5,880 B) for dense. Output drift
  against the earlier GPU run was at most `4.45e-15` and `9.33e-15` Ha/Bohr²,
  respectively. This removes three scalar reads per left coordinate, not the
  remaining scalar audits between response stages. The same run passed 110
  geometry/XC tests, with eight optional molecular-probe rows skipped.
- Slurm `2312` (RTX 5090) qualified the incumbent linked CUDA path on an
  asymmetric Cartesian STO-3G water dimer for both LDA and PBE. The tests
  compare strict CPU analytic HVPs, raw symmetry and bilinear products, and
  independently reconverged CPU analytic gradients at two displacement sizes.
- Slurm `2313` passed 16 focused grid checks on the new code, including
  propagation of a finite asymmetric input failure through a private panel,
  subsequent valid replay, and non-publication by the standalone checked API.
  Neither run verifies elimination of per-iteration transfers in CPKS.
- Slurm `2314` repeated the two public water dimer cases with the new grid
  composition. Slurm `2315` passed 11 Fock/CUDA endpoint cases after moving
  the Fock census and audit into device execution, before changing GMRES.
- Slurm `1791` ran nine iterations (eight restarts) with only the four-byte setup
  checkpoint. The 32 other executable GMRES gates in `1792` also passed; the
  build-only generation check passed separately in the full source checkout.
- Slurm `1794` ran five independent molecular Fock and CPKS comparisons on the
  graph-controlled build, including Direct with optional resident ERIs disabled.

## Borrowing force-owned center geometry

The force controller and the Hessian already share the canonical quadrature,
but their AO tile layouts are not interchangeable. A force map is certified
only for its own derivative order and cutoff; neither its omissions nor its
reused tile scratch can certify order-three Cartesian AO jets. Likewise,
direction-dependent response features are not the stationary tile's density
values. Keep bounded evaluation when no compatible AO/feature owner exists.

The force's FP64 triangular Becke table does match the mixed geometry owner.
Borrow only its distance and Cartesian slope after successful force publication
for the identical live KS and primary quadrature. Check exact pointers, grid
policy, device and stream, and reborrow against a force publication epoch. A
reset, failure, replaced owner, or changed KS token must not validate an old
lease, even if the allocator reuses its address. Cold second-order calls must
not create a force owner. The compiler specializes the existing norm AD at
the retained unit separation and uses norm homogeneity for the mixed tangent;
the force's first-order ratio entries are not read by the Hessian.

- Slurm `2338` passed all four public LDA/PBE Cartesian water-dimer oracle and
  live-peak tests on graph-controlled build `cc8259c4eedf16d4…` before adding
  the force borrow. It does not qualify the final integration build.
- Slurm `2339` passed ten moving-grid tests including an independently prepared
  FP64 center table, mixed weight differences, output alias rejection and
  revocation. The probe did not use a production force owner. Direct LDA/PBE
  molecular integration additionally exercises the actual force table and
  reset/reborrow. It requires the `lda_rks` and `pbe_rks` force profiles;
  a Hessian-only binary can evaluate the same HVP via bounded recomputation
  but cannot qualify this borrow.
- Slurm `2342` passed all 12 grid and force-to-Hessian integration tests on
  build `ac6c224f867677d4…` with the two profiles linked. The actual compact
  water force replay invalidated the preceding lease at the same KS token.
  Against bounded recomputation, the maximum raw HVP difference was
  `5.55e-17` (LDA) and `2.78e-17` (PBE) Ha/Bohr².
- On the earlier `cc8259c4eedf16d4…` build, the symbolized CUDA Runtime census
  found 37 H2D calls / 8,293,444 B for the HVP and 36 / 8,293,300 B for dense.
  These resolve to setup descriptor uploads and the HVP direction, not response
  panels. D2H was 32 / 492 B (HVP) and 26 / 2,956 B (dense), including 144 B
  and 2,592 B of result respectively. Remaining calls resolve to admission,
  checks, terminal response audits and diagnostics. A driver-level capture is
  still required on the integrated build; Runtime interposition alone cannot
  rule out traffic in a provider that calls the driver directly.
- Slurm `1804` completed the matched LDA dense-water phases on the preceding
  `cc8259c4eedf16d4…` build. Native vs mixed whole-endpoint seconds were
  `1.374 / 1183.012` (cold), `0.354 / 267.573` (warm), and `0.399 / 267.865`
  (moved); mixed cold includes JIT. Raw matrix differences across these phases
  were at most `2.27e-12` Ha/Bohr². PBE and the separate same-node rerun of the
  integrated native binary are needed before making the full benchmark claim.
- Slurm `2343` passed both public scientific and both concurrent-live-memory
  water-dimer gates on build `ac6c224f867677d4…`. Aggregate observed ledger peaks
  (HVP, dense) were `(77,073,749, 77,122,741)` B for LDA and
  `(77,172,053, 77,221,045)` B for PBE. The Runtime census repeated the
  H2D/D2H counts above on the same binary. Nsight Systems 2025.2 captures
  (`2346` HVP, `2347` dense) confirmed identical H2D and D2H call/byte totals
  from the accepted-response boundary. Device-to-device activity was
  134 / 92,354,836 B (HVP) and 2,190 / 1,662,298,024 B (dense); these are
  traffic volumes, not simultaneously live allocations.
- Slurm `2346` passed six final-binary budget, stale-token, unavailable native
  XC, nonconvergence/replay and non-resident CPKS-operator tests. Slurm `2348`
  disabled exact nuclear and induced Coulomb residency together with the
  CPKS operator cache. On compact water, both streamed runs executed 52,374
  induced Direct quartets with zero resident nuclear integral bytes and
  zero cached CPKS columns. Raw HVP differences were below `2.78e-16`, and
  dense differences below `3.34e-15` Ha/Bohr². LDA/PBE have no exchange term;
  this does not qualify exchange-bearing models.

### Final matched endpoint timings

All phases use the `node4` RTX 5090, strict FP64, Cartesian STO-3G and the
same fixed grid policy. Slurm `1805` timed integrated binary
`ac6c224f867677d4…`; mixed HVP baselines are `1803` (LDA) and `1802` (PBE),
and both mixed dense baselines are `1804`. The compact water dense endpoint
publishes nine raw columns; the asymmetric water-dimer HVP publishes one
direction. Each cell is one whole endpoint call, in seconds. Warm reuses the
compilation cache, not a warm SCF density. Mixed cold includes JIT; timings
are not a claim about isolated kernels or other workloads.

| Endpoint | Method | Native cold / warm / moved | Mixed cold / warm / moved |
| --- | --- | ---: | ---: |
| HVP | LDA | 1.570 / 0.331 / 0.326 | 797.174 / 303.444 / 302.865 |
| HVP | PBE | 1.366 / 0.380 / 0.361 | 810.763 / 317.967 / 318.467 |
| Dense | LDA | 1.579 / 0.351 / 0.348 | 1183.012 / 267.573 / 267.865 |
| Dense | PBE | 1.350 / 0.371 / 0.367 | 1189.598 / 271.624 / 272.314 |

All 12 comparisons met the 5% no-regression gate; the max elementwise raw
difference was `3.27e-12 Ha/Bohr²`. Work was invariant across all three phases
for both methods: HVP executed 24 induced Fock actions, 28,224 one-electron
and 516,132 exact Coulomb primitive components, and 18 mixed grid contractions;
dense executed 28, 39,690, 396,576 and 81, respectively. No intermediate
panel downloads or numerical host fallbacks were reported.

The per-phase gate receipts and executable reproduction driver are retained
under `benchmarks/results/semilocal-cartesian-cuda-1905/`. Slurm `1807` repeated
the four public water-dimer cases on the identical binary and recorded the raw
oracle, bilinear, symmetry and live-peak values in that record. No runtime
source was changed for this receipt-only rerun.

The same record holds `1808`'s nine-pair comparison with the prior native
CUDA endpoint, independent of the mixed baseline. All twelve method/endpoint/
phase medians met the 5% gate with unchanged response work. The post-format
rebuild (`1809`) checked numeric and work parity, production force leases and
peak live memory without overwriting any measured binary identity. Slurm
`1811` independently reran the two scientific and two live-ledger tests on
that final library: four passed, with unchanged peak bytes and no skips.

Slurm `2358` closed the transfer census on the final formatted library. CUPTI
and independent Runtime interposition agreed for Direct LDA and PBE after
accepted water-dimer SCF. All uploads were setup before the first graph;
neither endpoint copied host data during a graph. Dense used just two 4 B
status reads between graphs for its 18 directions and did not download any
per-column solution or Krylov receipt. HVP used eight inter-graph status or
diagnostic reads (84 B). This is not a blanket ban on host synchronization:
failure and final-publication status remain observable. The benchmark README
records totals, capture boundaries and the reproducible trace query.

The wider component regression on the pre-format binary recorded 324 passes
and two skips for the unavailable pinned libquadmath oracle. All 21 other
cases initially reused a stale live-KS test executable across the internal
resident-grid layout change. With that probe recompiled from the current
headers and linked to the same verified library, `2355` passed all 21 cases
and 18 complete stationary cases, including Direct/DF, RKS/UKS, streamed
integrals, FP32 proposals with strict physical audits, and failure/replay.
