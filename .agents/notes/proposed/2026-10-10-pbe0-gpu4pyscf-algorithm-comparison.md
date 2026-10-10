# Decision: compare PBE0 algorithms before copying GPU4PySCF schedules

Status: proposed; source comparison only, no implementation or new performance claim
Date: 2026-10-10

## Scope and provenance

Compare restricted PBE0 SCF and complete stationary nuclear forces, not another
backend timing contest. Keep direct and density-fitted providers separate.

- GENERATIVEQC inspected source: `99ebd196df529bdd45107af44b86b7879d3bb69f`.
- Observed upstream master: `4b330ff3d7aed41a0f232e90e572c3e9112879e9`.
  Its intervening `7a807d43c` already promotes reciprocal uniform MD-J source
  reuse. This comparison does not merge master or rerun qualification matrices.
- GPU4PySCF inspected upstream source:
  `c1a6e371d1a46afc9932c618108a4bc70e895edf`, not the older local 1.8.1 checkout.
  Primary sources and GitHub contents receipts are retained under
  `.artifacts/algorithm-comparison/gpu4pyscf-c1a6e371d1a46afc9932c618108a4bc70e895edf/`.
  Every decoded file was checked with `git hash-object --no-filters` against the
  contents API's blob SHA. These sources are not a measured installed backend.

No new GPU benchmark, profiler, or numerical qualification is part of this
comparison. Earlier endpoint measurements remain attributed to their own source
snapshots. Existing in-flight qualification is not restarted for this analysis.

## Shared mathematics, different physical schedules

Both restricted PBE0 endpoints combine 25% exact exchange with
`3/4 * PBE exchange + PBE correlation`. AO/occupation conventions must still be
matched explicitly; identical functional names alone do not establish identical
grids, screening, density fitting, stopping metrics, or force coverage.

### Direct Coulomb and exchange

GPU4PySCF RKS `get_veff` calls `get_j` and `get_k` separately. Its J owner uses
McMurchie-Davidson/Hermite density precontraction: transform the AO density into
pair coefficients, contract the screened pair tasks, then transform J back.
Its K owner dispatches Rys-quadrature kernels with shell/density screening and
angular/contraction-dependent work schemes.

Our direct composition also separates the specialized density-precontracted
MD-J owner from compiler-generated exact K. Therefore "switch J to MD" is not
an unimplemented algorithmic opportunity. Master's reciprocal source reuse
further overlaps this direction. K's radial/recurrence and physical task
schedules are a genuine comparison topic, but a whole-engine Rys replacement
does not follow from the source comparison. Compare executed primitive/root/
recurrence work and all relevant angular classes before choosing a replacement.

GPU4PySCF can build from `D - D_last` and reuse the prior potential. Our CUDA KS
owner also has an incremental anchor. The README PBE0 protocol explicitly asks
for full reference Focks; its wrapper clears both incremental inputs because
`direct_scf=False` alone does not disable RKS reuse. Default incremental work
must not be represented as an improvement to a matched full-Fock protocol.

### SCF numerical integration

GPU4PySCF `_nr_rks_task` is a two-pass schedule:

1. Traverse screened AO blocks and populate a whole-device-grid rho panel.
   If the density has occupied-orbital metadata, project the active AO rows
   through occupation-scaled occupied coefficients; otherwise use the local D.
2. Evaluate unpolarized XC on that whole rho panel, then traverse AO blocks
   again and assemble local potential matrices with dense products/scatter.

The inspected source's configured block-size default is 4096 points. Its
unpolarized GGA LibXC input is total rho and sigma; `transform_vxc` converts
the invariant derivatives back to Cartesian feature derivatives.

Our native SCF integration has compiler-owned tiled D/AO and symmetric
potential products, plus optional shared-library providers. Its point batching
retains AO/work panels and preserves each tile's selected AO domain. Ordinary
force tiles prefer 512 points with bounded smaller fallbacks; known fitted
large-grid owners prefer 256. These are different owner policies, not proof
that one universal tile size is best.

Compact contraction batching already exists and is enabled where admitted.
Its current group guard requires every nonempty tile to have 32--128 active
AOs. The historical master cold diagnostic found many real domains outside
that range. The missing opportunity is qualification for larger/ragged local
domains, not a missing GEMM or batching enable switch. Raising 128 alone is
incorrect because the compact launch geometry also encodes that bound.

The schedule tradeoff is explicit: GPU4PySCF's two passes can reevaluate AO
panels, but aggregate point arithmetic into larger domains; our retained panels
save producer work, but small/ragged contractions can fall back to many launches.
Larger point blocks also enlarge the union of active AOs. Merging small tile
maps can increase quadratic local-matrix work even while reducing launch count.

### Occupied projection is not a universally cheaper density algorithm

For B points, m local active AOs and o occupied columns, local D projection
costs approximately `O(B*m*m)` per required jet, whereas occupied projection
costs `O(B*m*o)`. Sparse local m can be smaller than o. The full occupied force
pullback also needs a coefficient backprojection; density-feature savings alone
are not complete-force savings.

Our repository already implements validated orbital sources, tiled occupied
projection and occupied feature reduction. However, the current stationary
task-view contract requires density-product jets and rejects `use_orbitals`.
Simply enabling orbital features does not provide the work panels required by
the current geometry pullback. A real orbital force route would need the
coefficient backprojection and its own producer/resource admission.

Admit such a route only for coefficients/occupations proven to reproduce the
same current D and generation. Preserve D-based fallback for missing, stale,
indefinite, response or otherwise incompatible sources. Do not reconstruct a
different density from convenient final orbitals.

### Complete XC force and Becke response

The inspected GPU4PySCF `get_exc_full_response` first constructs rho from
masked density matrices, not occupied orbitals. It evaluates whole-grid XC,
contracts explicit Becke weight derivatives with the XC energy seed, then
uses higher AO derivatives to construct derivative operator matrices and
contracts them with D. Its ordinary `get_exc` route uses occupied orbitals,
but that is not the same grid-response endpoint.

Our force owner reuses D/AO jet products, pulls XC feature cotangents back to
AO jets, and scatters their coordinate derivatives directly to atoms. This
avoids materializing an additional derivative-potential matrix, but does not
remove the preceding quadratic D/AO projection. Our phased Becke primal/
reverse schedule differentiates the partition/normalization and contracts
energy seeds in bounded owners. GPU4PySCF instead materializes a tile-local
`dweight_dA[atom, Cartesian, point]` from dedicated Becke derivative kernels.
These are contraction-order/storage differences, not different PBE0 physics.

The newly bound restricted point evaluates four common first-derivative
channels and one exchange expression, under an owned equal-spin witness.
GPU4PySCF's unpolarized invariant representation suggests investigating a
rho/sigma lowering, not copying LibXC constants or inserting independent
production formulas. Any new invariant lowering must retain stable low-density/
subnormal behavior and the exact spin/occupation chain factors. Restricted
first derivatives do not qualify spin-antisymmetric response/HVP channels.

Screening contracts also differ. GPU4PySCF's source has an AO screening
threshold of `1e-10`, full-response weight masks at `1e-14`, and a Becke
normalization guard at `1e-14`. These act on different quantities and cannot
be compared numerically to our order-two sampled-AO-jet `1e-16` force cutoff.
Our force maps discover their own derivative domain. Reusing an order-one map
or importing the other engine's thresholds changes the retained work/error
contract and is not an equivalent schedule optimization.

### Density fitting is a separate scientific/provider comparison

GPU4PySCF's optional DF route contracts metric-treated three-center factors
and density/orbital factors, with auxiliary-basis response in the gradient.
Our fitted providers also have occupied-factor machinery. Compare DF against
DF with matched auxiliary basis, metric treatment and full derivative coverage;
do not call direct-to-DF approximation a direct-integral algorithmic speedup.

## Proposed next priorities

1. Extend the existing compact/library XC schedules to actual ragged extents.
   Preserve separate AO maps and final accumulation order; charge descriptors,
   retained AO/work/local matrices and provider workspace to the complete owner.
   Test legal extent-specific segments/groups, not an unconditional larger tile.
2. Investigate a compiler-owned invariant restricted point lowering against the
   existing four-channel bound. Preserve the shared scientific expression and
   all independent point references, including subnormal cases. Keep the current
   bound/general fallback until complete endpoints qualify a replacement.
3. Reduce order-two map-discovery work through derivative-aware conservative
   bounds or authenticated same-geometry reuse. Measure discovery plus the
   resulting active consumer domain; a denser map can erase the producer saving.
4. Treat occupied force backprojection as a cost-admitted alternative, not a
   missing basic feature. Qualify it only where full forward/backward work is
   cheaper than the existing sparse D route.

Master changes trigger dependency inspection, not blanket retesting. The new
MD-J commit affects cold SCF integral work; unrelated CC/tensor commits do not
justify repeating restricted-point/force GPU matrices. Every future real-GPU
qualification still requires finite Slurm execution, complete E+F host return,
unchanged independent gates, semantic work counts and bounded fallback checks.

## Primary-source entry points

GPU4PySCF paths below are relative to the pinned ignored source root above:

- `gpu4pyscf/dft/rks.py`: `get_veff` and incremental J/K dispatch.
- `gpu4pyscf/scf/j_engine.py`: MD density transformation and screened pair tasks.
- `gpu4pyscf/scf/jk.py`: Rys K dispatch, screening and work schemes.
- `gpu4pyscf/dft/numint.py`: `_nr_rks_task`, `eval_xc_eff`, `_block_loop`.
- `gpu4pyscf/grad/rks.py`: `get_exc`, `get_exc_full_response`.
- `gpu4pyscf/hessian/rks.py`: `get_dweight_dA`.
- `gpu4pyscf/dft/gen_grid.py`: `_build_non0ao_idx_cache`.
- `gpu4pyscf/df/df_jk.py`, `gpu4pyscf/df/grad/rks.py`: optional DF and response.

GENERATIVEQC source entry points:

- `src/scf/cuda/direct_jk.cpp`, `src/scf/cuda/direct_md_j.cu`.
- `src/dft/cuda_ks.cpp`, `src/dft/cuda_xc.cpp`, `src/dft/cuda_grid.cu`.
- `python/generativeqc_compiler/dft/xc_point_batch_cuda.py`.
- `python/generativeqc_compiler/dft/xc_density_provider.py`.
- `python/generativeqc_compiler/dft/density_source.py`.
- `python/generativeqc_compiler/method/stationary_cuda.py`.
- `python/generativeqc_compiler/method/stationary_becke_phased.py`.
- `src/dft/xc_point.hpp`, `src/dft/stationary_gradient_cuda.cuh`.
- `benchmarks/compare_gpu4pyscf_batch.py`: full-reference-Fock wrapper.
- `docs/developer/stationary_cuda_scheduling.md`: force map/tile contracts.
- Master `7a807d43c`:
  `.agents/notes/proposed/2026-10-10-master-pbe0-96-cold-roadmap.md`, historical
  work-domain diagnosis; not a current-source timing claim.
