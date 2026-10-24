# Investigation: #2609 complete cold correspondence with GPU4PySCF

Status: implemented audit; measured repairs and explicit unresolved differences
Date: 2026-10-19

This audit covers the executed all-electron, restricted, FP64, direct
r2SCAN-3c/def2-mTZVPP (1088 AO) and PBE0/def2-SVP (768 AO) E/F paths.
It does not claim a whole-repository equivalence proof. Unsupported angular
momenta, response operators, multi-GPU, unrestricted and signed densities are
recorded where they constrain a safe repair, not misrepresented as timed work.
The main inventory below is the preserved pre-repair checkpoint. Later repair
and ablation results are appended; it must not be read as the final code state.

Native root: `/data/jzzeng/qc-pr2609-accept-20261019`, source identity
`a103ed8b191611276d366d3a676c9f232780629435b5f31f95e96ceb2b38b34d`.
Upstream MD pin: `9af748a71ca8f4881c604332c2f0cc1632e033ac`;
complete upstream endpoints use the released 1.8.1 wheel, not a fully pinned
package. `.artifacts/pr2609/budget/frontend-audit.json` records hashes and AST differences;
raw wheel Python files are retained in `.artifacts/pr2609/budget/upstream-wheel/`.

Labels below: **C** compatibility/scientific contract; **P** work/schedule;
**R** resource/lifetime; **M** measurement/protocol. A difference is not by
itself a bug or a measured cause. Rows explicitly identify unresolved causality.

## Preparation and mathematical representation

| ID | Upstream source/behavior | Native source/behavior | Consequence |
|---|---|---|---|
| J01 | `scf/j_engine.py:_VHFOpt.build`: `SortedGTO.from_mol(decontract=True,diffuse_cutoff=1e200)` | `direct_jk.cpp:MdJHost` retains original shell parents and all primitive products | C/P: both have primitive descriptors; only upstream changes the screening basis to fully decontracted Cartesian shells. Calling both “decontracted” hides this difference. |
| J02 | `gto/mole.py:_optimize_contraction`: normalize shared primitives, move general-contraction coefficients into recontraction matrix | `direct_md_j.cu:md_j_transforms`: original coefficients, public spherical/cartesian AO linear combinations folded into cached E | C: coefficient placement and primitive identity differ. Negative/general contractions preclude deriving primitive q from a contracted q. |
| J03 | `mole.py:_scale_sp_ctr_coeff` applies libcint s/p factors; d+ normalization follows Cartesian/recontraction conventions | native basis normalization and `ao_term_coefficient` are folded into E | C: retain independent AO scaling converter; do not compare raw buffers without public-AO mapping. Existing same-density matrix gates pass. |
| J04 | `contract_Et.cu`: E is reconstructed per retained pair on each forward and backward transform; geometric exponential factored into z | cached sparse E prepared once by `fill_hermite`, exponential per axis; full geometry inventory retained | P/R: native spends extra resident storage to save repeat E construction. Preparation .0535 s and per-J transforms .00064 s show this is not the main gap. |
| J05 | `get_j`: explicit `C D C^T`, Cartesian primitive AO matrix, then Hermite transform; final `C^T J C` | directly public AO D → primitive Hermite D → public AO J | P/R: native avoids a large decontracted AO matrix; upstream has more mappings/temporary matrices, yet is faster overall. |
| J06 | `contract_Et.cu` self-shell half factors, MD self-quartet half, `Rt_to_dm` transpose sum, final `vj *= 2` | off-diagonal shell D folds both orientations, diagonal enumerates components, reciprocal self direction suppressed, no upstream half factors | C: equivalent symmetry bookkeeping, not interchangeable constants. Nonsymmetric/unrestricted tests must remain. |
| J07 | MD LMAX=4 per shell; CPU fallback for higher l; Coulomb range omega in Boys | native MD s/p/d only (pair l≤4), full-range J; normal provider handles unsupported modes; K/range separate | C: tested bases stay in shared domain. Do not expand scope or introduce CPU production fallback. |
| J08 | build caches q/pairs, but `get_j` recreates pair lists/loc, qD, buffers, scaling metadata each call; includes `vj_xyz.get()` host round trip | immutable owner, resident E/loc/geometry, all replay work on its stream; no host metadata polls | R/P: native lifetime design is advantageous, not an explanation for its slow kernels. Cold includes setup on both sides. |

## Screening and actual work inventory

| ID | Upstream source/behavior | Native source/behavior | Consequence |
|---|---|---|---|
| J09 | `jk.py:_cache_q_cond_and_non0pairs` first applies estimated overlap mask at `precision**2` | no corresponding primitive overlap prefilter | P/C: entire primitive domain can disappear before q construction upstream; native contracted-mask policy cannot simply inherit this deletion. |
| J10 | `lib/pbc/overlap.cu:ovlp_mask_estimation_kernel`: FP32 logarithmic Gaussian distance/exponent/coefficient estimate | public contracted AO self-integral bounds in FP64 | C/P: different quantity, precision, and cost. These are not identical Schwarz implementations. |
| J11 | `gvhf-rys/nr_sr_estimator.cu:int2e_qcond_kernel`: FP32 Rys q estimate, exp(30) underflow guard, log output | native independent AO bounds kernel evaluates contracted self integrals; shell min/max derived from both orientations | C/P: native preparation GPU bounds 3.45 s, upstream uses an estimator; copying estimator would change the screening contract. |
| J12 | primitive q sorted descending within `(li,lj)` classes | contracted-parent q repeated for each primitive, sorted within total `li+lj`; active prefix uses parent minimum times global maximum | P: primitive distance decay does not remove individual sources in native. The 79,836 native active and 62,844 upstream retained pair counts are not like-for-like predicates. |
| J13 | `condense(absmax)` on transformed D; log(q)+log(maxabsD), no total-shell-pair error apportionment | `md_j_density_bounds`: max abs public D times component multiplicity (and spin/orientation bound), upward rounding | C/P: native deliberately conservative density envelope; compare products and error model, not only nominal tolerance. |
| J14 | q/logD tile gate against direct_scf_tol | density allowance `min(screening,1e-12)/pair_count`; r2 pair_count=131328 gives ~7.6e-18 | C/P: stronger cumulative omission policy. Threshold sweep upstream 1e-12→1e-22 changes .628→1.100 s, still far below native 3.922 s. Threshold alone is insufficient. |
| J15 | all admitted J uses Hermite precontraction | `md_j_engine_direction`: uniform minimum-AO mask only; partial masks go to `direct_md_jk.cu` | C/P: crucial architectural difference. Native uniform is 1.353 s, residual 2.558 s per fixed-density J. |
| J16 | no AO integral reconstruction in MD inner source loop | residual reconstructs every retained shell AO quartet, then applies the exact per-orientation AO mask and contracts D | P/C: loses the central MD-J contraction advantage for that work. Removing residual without replacing its contributions is wrong. |
| J17 | qD hierarchy is one FP32 logarithmic field (six levels); geometry q separate and sorted | hierarchy stores four FP64 fields q/qD/minimum/density; qD field not sufficient for exact leaf rounding | R/P/C: larger hierarchy and different coarse/leaf decisions. Native hierarchy update is only 19.5 µs per J, not the principal cost. |
| J18 | block-level q screening, subbatch qD screening; generic kernel can evaluate extra within-batch quartets | tile min/density bound plus exact per-source direction gates | P/C: fine-grained native branching has extra load/branch cost but preserves old boundary semantics. |
| J19 | fully decontracted lower triangle reuses primitive shell identity | native diagonal contracted shell retains ordered primitive products, and general contractions can repeat primitive exponents | P/C: additional identity/symmetry reuse needs explicit folding proof. Total native431952 vs 928-shell triangle431056 already differ before pruning. |
| J20 | q/overlap pruning bounds actual traversal before launches | native uniform active prefix bounds traversal, but residual visits the Cartesian/triangular active contracted-pair inventory and compacts128 at a time | P: “bounded resident memory” alone does not bound useful work. A new primitive path must prune its domain before quadratic tile traversal. |

## CUDA inner loop and dispatch

| ID | Upstream source/behavior | Native source/behavior | Consequence |
|---|---|---|---|
| J21 | `md_contract_j.cu:MD_build_j`: low-l unrolled dispatch, generic fallback | generated compact recurrence for all angular classes | P: controlled upstream generic ablation costs only9% r2/11% PBE0. Not the6.23× J explanation. |
| J22 | generic256 threads; shape-dependent source/component lanes, device shared-memory limit, up to32 batches per dimension |128 threads, equal tile widths≤32, static48KiB limit and32KiB target | P: both use quartet/component decomposition after the first repair, but neither tile span nor shared-memory/occupancy policy matches. Untuned. |
| J23 |2D launch covers pair tiles; generic tile spans `threadsx*tilex` by `threadsy*tiley` | persistent≤4096 CTAs, atomic64-bit source cursor, integer-corrected triangular decoding | P: fewer useful sources per native task; atomic/decode/barrier cost and memory reuse differ. Requires layout/work ablation. |
| J24 | cache product centers/exponents only (four doubles/pair); compute centers on load | cache72-byte full primitive descriptors; centers precomputed | R/P: native copies transform/parent/offset metadata unused by part of inner loop. Tradeoff not yet quantified. |
| J25 | generic stages≥4 compute radial states across component lanes via31 registers and barriers | descending generated in-place scalar stages owned by component lane0 per source | P: generated mathematics is shared, schedule is not. Higher-order component lanes wait for serial recurrence. n4:1170 generic ablation: radial 0.6775 s versus generic 0.6817 s; Boys 0.7577 s. Matrices agree within 3.91e-14. These changes alone do not explain the raw-J gap. |
| J26 | positive incomplete-gamma series for small t; threshold m/2+.5; upward recurrence above | shared native alternating highest-order series with80-step bound, order-specific thresholds, downward recurrence; then upward branch | C/P: FP64 both, but arithmetic, convergence and branch costs differ. New independent ablation prepared; do not change shared Boys without its independent oracle. |
| J27 | prefactor folded into Boys seeds once | prefactor multiplied into each output dot (uniform engine); residual owns separate recurrence | P/C: different operation count/order; changing it requires radial/matrix gates. |
| J28 | uint16 summed-state map in global/shared, small map copied to shared; dynamic runtime l | uint8 read-only global map, signed phases in constant memory; compile-time l templates | P: native smaller map but no shared staging. Old constant-memory divergence issue already fixed. |
| J29 | bra values retained across ket batches; shuffle ket sums; final global atomics | same principle after repair, but more tile-level publication/descriptor reloads | P: first repair removed per-source shared-double CAS and idle lanes. Remaining disparity cannot be attributed to that old code. |
| J30 | can batch four D matrices in one recurrence; multi-GPU task iterator | RKS1 or UKS total density J, system batching inside owner, no multi-D MD reuse | C/P: not active for measured RKS; do not credit unused upstream capability. |
| J31 | RKS calls separate MD J and Rys K; no K for r2SCAN | `direct_jk.cpp` resolves J independently; `launch_md_source_jk(...true,false...)` is residual J; K retains generated/canonical provider | C: the residual kernel name does not prove K work is happening. MD repair must not alter K or derivative dispatch. |
| J32 | single J pipeline initializes its own Hermite output | native zeros J in caller and residual entry, refreshes density bounds before residual and again before uniform | P: actual redundant small work, but measured transforms/bounds are tiny. Cleanup is not a solution to a seconds-scale gap. |

## XC, solver, forces, and complete cold endpoint

| ID | Upstream source/behavior | Native source/behavior | Consequence |
|---|---|---|---|
| X01 | `hf.make_rdm1` tags mo_coeff/mo_occ; `nr_rks` chooses `_eval_rho2` | `CudaKsPlan::stage_xc` supplies full D; retained orbital frames exist but no XC factor identity | P/C: occupied-factor rank160 can reduce contraction dimension, but some local AO maps have<160 entries. Need actual shape/work comparison, not universal n/occ speedup. |
| X02 | occupied C^T A via cuBLAS; no-tag fallback D A also BLAS | `xc_contraction_cuda.py` mapped16×16 FP64 scalar-FMA tiled density multiplication | P: measured37.400s/24cold XC density products. Both arithmetic and repeated global D gathering are different. |
| X03 | `numint._tau_dot`, `_scale_ao`, `.dot`/`contract` use GEMM potential assembly | compact meta-GGA diagonal panels, generated triangular tiled products with range-safe doubled-panel fallback | P/C: measured27.216s/24cold diagonal potential products. Fewer formal products does not establish faster execution. |
| X04 | production library calls | native density and potential cuBLAS candidates exist, but `cuda_panel_product.cuh` and `cuda_symmetric_product.cuh` set qualified=false except tests | R/P: do not claim BLAS is absent; qualification and selection are missing. |
| X05 | temporary pool memory acquired by Python helpers | `dft_method.cpp` passes empty `CudaXcPreparationBudget`; potential default provider budget also0 | R: changing qualified alone cannot activate these paths; actual owner reservations and fallback must be supplied/charged. |
| X06 | grid blocks4096; contraction-pattern sorted shells; shell screeningAO_THRESHOLD1e-10 | tile256; public-AO point/derivative sampling cutoff1e-16, immutable sorted local maps | C/P: distinct selection and tile geometry. Native9216tiles/776empty, mean active230.8, max767;2.359Mpoints. Upstream occupancy/cutoff ablation required. |
| X07 | density pass evaluates AO, full-grid XC, potential pass evaluates AO again | one AO pass retained across up to32tile point batches; fused generated native point program | P/R: native saves AO recomputation and controls temporary storage, upstream larger GEMMs. No assumption that fusion wins overall. |
| X08 | CUDA libxc7.0.0 with rho/gradient/tau conversion kernels | native compiler emits r2SCAN expression and physical-domain policy | C/P: same functional name does not imply identical instruction graph or low-density handling. Independent endpoint gates pass; include point cost in trace. |
| X09 | sparse scatter helper adds local matrices | native ordered scatter preserves tile order, finite checks, compensated scalar totals | C/P: different reduction determinism and checks; cannot simply remove safeguards to emulate timings. |
| X10 | allocating buffers/ThreadPoolExecutor/default-stream synchronization each nr_rks | prepared stream/arena, optional graph body, generation authentication | R/P: native avoids Python overhead; measured GPU kernels dominate the slowdown. |
| S01 | wheel `_kernel` uses default MINAO and tagged density; GPU eig/DIIS | native observed `initial_density_used=false`, core seed; native preliminary policy exists separately | C/M: different start and solver trajectories. Raw-J benchmark instead supplies identical MINAO D. |
| S02 | transformed/packed CDIIS residual, history8, cuSOLVER through CuPy, orbital-gradient convergence | retained native DIIS/eigensolver; full physical AO residual max/RMS and density change plus compensated energy gates | C/P: native24Fock vs upstream60 clean/55 instrumented; iterations cannot be normalized away or mixed between populations. |
| S03 | default RKS reuses previous potential with deltaD even when direct_scf=False | native full-density mode in measured protocol; optional incremental path exists | M/C: benchmark explicitly suppresses dm_last/vhf_last while preserving tags. Upstream default incremental advantage is intentionally absent from these results. |
| S04 | warm density restarts SCF, no native retained compensated energy baseline | same-owner native warm D/energy/final-state provenance, one-Fock warm convergence | M/R: warm29.34vs31.04s does not establish per-iteration parity; upstream warm takes5–7JK. |
| F01 | gradient creates its own sorted J/K/Rys derivative path | retained source/spatial SPD force recurrence; MD value selector not reused for derivatives | C/P: J value optimization cannot be assumed to speed force. Post-SCF native21.55s vs upstream gradient16.53s GPU includes different finalization scopes. |
| F02 | grid-response gradient rebuilds/copies/sorts grid; full-density rho,10AOjets, weight derivative batches, nonzero-weight1e-14 mask | stationary controller/token, generated grid response and contraction, retained grid/source bindings | C/P/R: upstream occupied SCF shortcut is not used by full-response gradient. Must inspect both paths separately. |
| F03 | independent CPU D4+gCP in reference driver | native correction owners | M/C: both corrections included in E/F. Reference correction .071s in diagnostic; not a cold bottleneck. |
| R01 | PySCF/CuPy memory pools and dynamic allocations | fixed owner budget/fallback, public-ledger optional exclusion, geometry rebuild invalidates all bindings | R: capacity semantics differ deliberately.128MiB was a reservation hint mistakenly reused as admission cap; already fixed, measured144.96MiB admitted. |
| M01 | complete released package vs pinnedMD+released helpers are separate arms | final source-matched native library | M: retain all hashes and do not call hybrid pinnedMD arm a completely pinned package. |

## Causal evidence and next repairs

Clean complete r2 cold236.820s native vs153.635s upstream (single observation),
with24vs60J builds. Native instrumented cold GPU partitions: J93.765s
(uniform32.471+residual61.278), XC95.101s, solver7.181s, initialization3.594s,
post-SCF21.549s. Separate preparation GPU7.451s. These disjoint regions sum to
the captured kernel total; none should be added again to complete timing.

Already ruled out as sole causes:128MiB refusal (fixed), Python/transfers
(device≈wall J), low-l unrolling (9–11%), nominal J threshold (1e-22 still much
faster upstream), E transforms/hierarchy (submillisecond).

The next implementation must address **screening domain plus complete Hermite
J**, not merely swap the residual's recurrence, and independently qualify
**XC density/potential execution**. New full-primitive screening must have an
explicit error contract; deleting the legacy AO mask or residual under the
existing `reference` name is not a compatible repair. New optional BLAS
admission needs budget ownership, actual shape economics, unavailable/OOM
fallbacks and complete E/F gates. Boys/launch changes should be prioritized
only after the prepared isolated ablations establish their contribution.

## Further source-history check

X04/X05 are deliberate non-promotions, not sufficient evidence of a missing-call bug. The retained mapped-density and mapped-potential crossover notes from 2026-10-05 measured slower complete XC with these particular per-tile BLAS recipes. Those PBE0 data do not establish r2SCAN-3c performance, but rule out promoting them merely because GPU4PySCF uses BLAS.

New n4:1170 fixed-density GPU4 XC medians (5 warm, first separately retained): cutoff1e-10 occupied1.445628/dense1.671369s; cutoff1e-16 occupied1.911823/dense2.435085s. All four arms use identical core-Hamiltonian density. Vxc max difference1.340e-11; energy4.15e-12. Maps576 blocks: active sum114606 at1e-10 and151674 at1e-16. This is not a native endpoint acceleration claim.


## Additional differences found by continued source reading

| ID | Upstream | Native checkpoint | Classification and measured consequence |
|---|---|---|---|
| X11 | `gen_grid.py:Grids.build` defaults `sort_grids=True`; `atomic_group_grids` walks neighboring atoms then groups points by nearest center; coordinates, weights, atomic measures and owners move together | `MolecularGrid::from_cuda` preserves atom/radial/polar/azimuth order | P/R: exact point permutation, distinct from changing quadrature. n4:1176, same D/cutoff1e-16: dense XC unsorted5.953666 vs sorted2.435611s; occupied3.929156 vs1.909951s. Maximum active AO932 vs458. A measured major cause. |
| X12 | `nr_rks` density/potential library phases are independent | native `prepare_point_batches` disables compact batching if either optional provider is enabled | P: density-only provider also changes potential scheduling; fixed-D provider arms measure the whole effect, not pure GEMM speed. |
| X13 | full response gradient constructs/retains its own owner/atomic-weight ordering | resident native stationary binding requires canonical point pointers and `points_per_atom` owner arithmetic | C/R: sorting SCF globally would break force identity. The repair must retain an authenticated canonical force view and separate exact execution permutation. |
| S05 | CDIIS forms only the new history dot row with library contractions; error vector is orthogonal-basis/packed | native default `update_diis_kernel` rebuilds all history dot products with one warp; opt-in incremental/ordered Gram already exists | P/C: current v2 trace spends6.024s in this kernel. Do not conflate different residual metrics or promote an existing reducer without endpoint gates. |
| S06 | optional/default last-D/last-J incremental behavior has no native cumulative chain policy | native `resolve_incremental_direct_jk_policy` binds provider screening, strict precision, bounded age/drift and exact anchoring | C/M: not merely a missing delta call. The matched campaign disables upstream delta reuse and native remains full D. |
| S07 | upstream checks total energy change and orbital-gradient norm; level-shift extra cycle can accept energy OR gradient | native checks compensated energy, density change and physical residual, then retains its authenticated final density/energy for warm restart | C/M: equal tolerances are not identical convergence algorithms. Preserve E/F gates and actual Fock/JK counts. |

## Fixed-density XC evidence and failed alternatives

n4:1174 uses identical public-AO core density in native and upstream, with
explicit AO scaling. Native generated256:3.988047s; density BLAS:3.743939s;
potential BLAS:6.796571s; both:5.991200s. Native1024 generated:5.364324s,
density:4.396650s,potential:8.018814s,both:6.893923s. Candidate V differences
are <=1.51e-14; all energies agree. Thus simply promoting all BLAS or increasing
block size is rejected for this actual domain. Setup and first samples retained.

Upstream same-D sorted XC: cutoff1e-10 occupied1.447156/dense1.677096s;
cutoff1e-16 occupied1.918083/dense2.444948s. Native versus upstream V max error
is5.82e-14 at1e-16,1.34e-11 at1e-10. The native point functional is not causing
an observable scientific mismatch at this input.

The first native exact permutation diagnostic gives256-point XC3.016372s,
point-AO-square work122,486,172,672 versus164,382,040,064 before. At1024,
sorted3.518009s remains slower than sorted256. This includes complete fixed-D
XC publication but excludes sorting/setup; no cold endpoint claim follows yet.

## Primitive repair checkpoint

The separately named primitive recipe supersedes J12/J15/J16/J20 for that
selector, preserves the reference mask recipe, and fixes the shared-memory
source-lane utilization cliff in J22/J23. See the linked primitive decision.
v2 clean r2cold180.891338s remains17.7% slower than upstream153.635193s;
raw J1.561526s remains2.48x upstream. Do not call these performance parity.
The trace n4:1175 has165.517408s total cold GPU kernel time and no residual
candidates. XC density/potential remains dominant. Spatial-grid repair and
complete qualification are recorded separately when available.

[Primitive execution decision](2026-10-19-primitive-complete-md-j.md).


## Qualified spatial repair

The [spatial-order decision](2026-10-19-xc-spatial-grid-order.md) records the
separate canonical force view, bounded optional copy, independent CPU/capture/
resource/sanitizer gates and all18 complete endpoint gates. n4:1178 r2cold is
157.723791s with24Fock, warm26.050681s, moved102.479027s with14Fock,
moved-warm26.020513s. PBE0cold62.549000s has17Fock, warm10.365096s.
The unrepaired source rows remain historical, not claims about this new selector.
58 distinct rows cover the executed paths and relevant excluded contracts;
this remains a scoped source audit rather than a whole-repository equivalence proof.


## Compiler arithmetic correspondence

| ID | Upstream | Native | Boundary |
|---|---|---|---|
| J33 | Locally built pinned MD uses NVCC default implicit FMA contraction; the retained CMake command has no `--fmad=false` | production CUDA uses `--fmad=false`; explicit selected intrinsics/library operations retain their own contracts | C/P: FP64 alone does not mean identical arithmetic or instruction count. Installed wheel build flags are not inferred from local source. An isolated generic MD translation-unit no-FMA ablation keeps screening, layout and helpers unchanged. |


## Final arithmetic/solver controls

n4:1181 upstream generic FMA/default0.683916 vs no-FMA0.884015s;
n4:1182 at1e-22 nominal threshold1.251107 vs1.585252s. Cross matrices agree
within1.78e-14. The thresholds still have different error semantics from native.
A native explicit-FMA candidate passed independent numerics but gave raw J
1.556471s and full cold157.698189s/24Fock, indistinguishable from retained
1.561526/157.723791. It was archived and reverted, not promoted. See the
[rejected FMA decision](../../rejected/2026-10-19-md-j-explicit-fma.md).

Existing native incremental DIIS controls (no new implementation) give
ordered156.680948s and cooperative154.634718s, each24Fock; both cold and one
warm E/F pass, n4:1179. These are separate candidates, not the retained spatial
endpoint's timings or default policy. Fresh GPU4PySCF1.8.1 cold n4:1180 is
153.314540s/60JK and passes E/F, consistent with the prior153.635193s sample.

## Per-Fock follow-up

The [source-norm/XC arithmetic decision](2026-10-19-per-fock-source-norm-xc-fma.md)
confirms that similar cold totals concealed slower native rounds. It records
24 native Focks versus 60 upstream `get_veff` calls in the clean campaign, and
separate actual component profiles rather than dividing E/F by iteration count.

X02's earlier phrase "scalar-FMA" described matrix-product structure too
loosely: the emitted multiply/add expressions were not fused under production
`--fmad=false`. Explicit FP64 XC fusion now reduces same-input complete XC by
40.4%. A tighter precontracted Hermite source certificate reduces J work while
preserving its cumulative omission allowance. Combined native J+XC per round
falls from 4.587006 to 3.257288 s; clean r2 cold falls from 158.655704 to
125.702913 s with the same 24 Focks. Upstream normal-cutoff per round remains
2.173046 s, so the remaining per-round gap is explicitly retained.
