# MD Coulomb engines

The resident direct J/K owner defaults to the complete primitive Hermite
**J-value** engine, diagnosed as `md-j-primitive/retained-k`. Selection is frozen
at plan construction. `GENERATIVEQC_MD_J_ENGINE=primitive` explicitly selects
the same recipe; `incumbent` restores the earlier MD source consumer and
`reference` selects the hierarchy with exact public-AO-mask compatibility.
Their diagnostics are `md-j-hermite/retained-k` and `md-j-reference/retained-k`.
Invalid values are errors. Inspect diagnostics: optional storage refusal retains
a bounded existing provider, so requesting a recipe does not prove admission.

The primitive recipe has the cumulative screening contract described below.
Both hierarchy recipes use the same generated radial/contraction mathematics,
native resource owner and retained K/derivative providers. The
[default-promotion decision](../../.agents/notes/implemented/performance/2026-10-19-md-j-spatial-defaults.md)
records qualification of the ordinary unset selectors.

This engine uses the architecture of official GPU4PySCF at
[`9af748a71ca8f4881c604332c2f0cc1632e033ac`](https://github.com/pyscf/gpu4pyscf/tree/9af748a71ca8f4881c604332c2f0cc1632e033ac).
It is integrated into the ordinary native provider, with generated scalar
mathematics and native lifetime/scheduling ownership. Neither PySCF nor CuPy is a
production dependency. Upstream source hashes and consumers are registered in
`upstream/manifest.json`; the Apache-2.0 license and modification attribution are
in `LICENSES/gpu4pyscf-Apache-2.0.txt` and `THIRD_PARTY_NOTICES.md`.

## Architecture correspondence

All upstream paths below are relative to `gpu4pyscf/` at the pinned revision.
The table describes the `reference` adaptation, which preserves the public-AO
screening contract. The default primitive recipe replaces its exact-mask source
admission with the cumulative bound below. Neither recipe claims binary layout
or screening equivalence with upstream.

| Stage | Pinned GPU4PySCF | Integrated GenerativeQC equivalent and compatibility choice |
| --- | --- | --- |
| Decontraction and sorting | `scf/j_engine.py`: `SortedGTO.from_mol(decontract=True)`, sorted basis and `C D C^T` | `MdJHost` keeps original public-shell identity plus every ordered primitive product. Resident sparse weighted E transforms factor the contraction, normalization and Cartesian/spherical maps without materializing dense C. This reuses the established transform pipeline. |
| Pair layout | `j_engine.py`: `(li,lj)` pair mappings sorted by `q_cond`, packed `pair_loc` | Within total-angular classes 0–4, a deterministic active-prefix, descending-q, primitive-ID permutation drives packed Hermite intervals. `md_j_engine_pair_loc` rewrites only Hermite offsets. Original shell/primitive identities and transform slots stay valid. Total-angular grouping is sufficient after the separate bra/ket E transforms; it is not upstream's raw-shell ordering. |
| Density precontraction | `lib/gvhf-md/contract_Et.cu`: `Et_dot_dm` | Existing `md_j_density` contracts the actual public density into resident primitive Hermite density using cached E. UKS uses alpha+beta; nonsymmetric density retains both AO directions. |
| Hierarchical screening | `j_engine.py` / `md_contract_j.cu`: float32 log q and q+dm, widths 1/2/4/8/16/32 | Six FP64 linear bound levels retain q, qD, a recipe-specific minimum field, and density. Reference nodes maximize leaf Schwarz minima; primitive nodes minimize target source thresholds. The reference recipe preserves the leaf's left-associated upward-rounded `(q_out*q_in)*density_in` expression. The complete primitive recipe uses the precontracted source norm described below at both coarse and leaf decisions. |
| Exact admission | Upstream applies its own shell-pair cutoff contract | `md_j_engine_direction` retains the existing public-AO minimum-product and directional density allowance. Partially masked pairs still use the established exact residual source. Thus upstream log thresholds and half factors are not silently imported. |
| Task/tile scheduling | `j_engine.py` scheme selection; `md_j.cuh` and `md_contract_j.cu` hierarchical task consumers | Checked 64-bit rectangular/triangular inventories and an atomic device cursor feed 128-thread tiles. Adjacent source lanes evaluate independent primitive quartets; larger angular domains reserve power-of-two component lanes using upstream's single-density shape targets. Interleaved radial scratch, register bra accumulation and shuffle/single-writer ket reduction avoid per-source shared floating-point atomics. Angular/workspace-aware tile widths respect a conservative 32-KiB target and 48-KiB hard cap. All 15 canonical angular class pairs, including (4,4), use this backend. |
| Generated recurrence and contraction | `md_indices.cu`, `md_pairdata.c`, `md_contract_j.cu`: compact Cartesian indexing, MD radial recurrence and two directions | Compiler-owned `md_j_engine_cuda.py` emits compact descending in-place R simplexes through order 8 and lane-owned forward/reverse contractions from the existing scalar IR and Boys seeds. One radial source supplies both admitted directions. |
| Multiplicity and reconstruction | `contract_Et.cu` same-decontracted-shell half factor, same-pair factor, transpose sum and final scaling | Existing `md_j_project` applies cached E back to original public AOs. Original-shell density folding and canonical primitive identity handle multiplicity. Upstream half factors cannot be copied independently because a decontracted shell differs from an original contracted shell. |
| Runtime lifecycle | Python/CuPy orchestration and upstream native launch code | Existing native plan owns transforms, layout, density, potential, hierarchy and cursor on one stream. Preparation-only q/minimum downloads are fenced before sorting; sorted uploads are fenced before staging is released. Replay has no host integral work, host polling or per-replay allocation. |
| Production dispatch | Upstream dedicated J entrypoint | Existing direct J/K dispatch selects this engine for admitted J-value work, retains independent K and residual owners, and reconstructs the same public result. Precision and derivative gates stay at the existing dispatch seam. |

## Scientific and resource contracts

The overlap domain is the existing original-shell s/p/d MD inventory, with at
least eight public AOs. A single restricted density or the total of two
unrestricted spin densities is supported. This is not a multi-independent-DM
extension, a high-angular-shell port, or a derivative kernel. Unsupported basis,
fixed-point/mixed-J/census modes and derivative consumers keep their existing
providers. In particular, availability of a derivative-order-one plan does not
mean this J-value kernel computes forces. Complete HF/DFT energy and force tests
exercise integration through the retained derivative providers.

In the `incumbent` and `reference` recipes, the minimum public-AO Schwarz bound
proves that
all public quartets pass; the maximum proves that none pass. Only the uniform
region enters the Hermite source consumer. The existing residual source retains
the exact per-public-AO mask for the partial region. Density omission still uses
`min(screening_tolerance, 1e-12)` apportioned by the **global contracted-pair
census**, downward-rounded allowance and upward-rounded contribution. Forward
and reverse admission remain independent. An exactly zero density has no source
work; a zero screening tolerance disables density omission of nonzero terms.

### Complete primitive recipe

The `primitive` recipe precontracts **all** Coulomb contributions and does not
invoke the partial-mask residual. It computes a conservative Coulomb norm for
each primitive AO-pair expansion from cached normalized E coefficients and
compiler-emitted Hermite norms. Absolute coefficient sums remain valid for
signed/general contractions. The bound belongs to each primitive, rather than
being copied from its contracted parent.

Let `epsilon = min(screening_tolerance, 1e-12)`, `m` be the target's
original-shell-pair primitive multiplicity, and `N` the complete primitive
census. Each target receives an allowance `epsilon/m`, rounded down. Its
source threshold is the larger of two independently certified choices:

- the uniform threshold `(epsilon/(m*N))/q_target`, rounded down;
- a histogram boundary whose complete source-norm tail is at most
  `(epsilon/m)/q_target`, rounded down.

The histogram counts positive finite source norms by their FP64 bit pattern.
Each bin contributes its integer count times its exact upper endpoint, with
upward-rounded multiplication and accumulation. The owner limits the primitive
census to signed 32-bit capacity, so converting counts to FP64 is exact. The
result bounds the sum of **all** sources below each boundary, including sources
that will be removed from traversal prefixes. Zero norms need no budget;
unknown/infinite norms always survive a finite threshold. Overflow in a tail
bound retains work. Choosing the larger threshold selects one complete
certificate and never combines two separately spent allowances.

Source-prefix admission uses the global maximum target norm and multiplicity,
so its threshold is no larger than any individual target threshold. Target
admission compares its threshold with the global maximum source norm. Hence
prefix omissions and leaf omissions belong to each target's same certified
tail. Summing at most `m` target shares bounds the complete absolute screening
error for every public J entry by `epsilon`. A single hierarchy pass builds
prefixes and node bounds; no host readback or second budget-splitting pass is
needed. The [source-tail decision](../../.agents/notes/implemented/performance/2026-10-20-md-j-source-tail-budget.md)
records the proof, measured tradeoffs and rejected alternatives. Other systems and inactive gaps only make the certificate conservative.

After density precontraction, each source gets the outward-rounded triangle
bound `sum_t abs((E D)_t) * norm(Lambda_t)`. This preserves cancellation within
the represented Hermite source and is stored in the existing hierarchy's `qD`
field. It is not capped by the uncontracted primitive norm times the shell
density envelope: precontraction rounding near underflow can make that envelope
smaller than the norm of the stored source. The output primitive norm times
this source bound bounds the directed contribution to any output AO. A tile
stages the source norm and the cached certified threshold. The hierarchy
computes its threshold once per primitive, reduces its minimum across each node, and uses infinity
for inactive/padded targets. A tile is rejected only when neither direction
has a nonzero maximum source norm at or above the opposite minimum target
threshold. The inner loop applies the same leaf threshold. This avoids repeated allowance divisions and bound products for
every directed source. Zero output norms, unknown bounds and zero allowance
have explicit conservative branches. The division may retain additional work
at rounding boundaries.
The complete omitted absolute contribution to each J entry is thus
bounded by `min(screening_tolerance, 1e-12)`; this bounds screening, not floating
point accumulation error. No old per-AO integral mask is applied. Zero tolerance
disables omission of nonzero terms. Nonrepresentable bound arithmetic retains
work, and zero density produces zero source work.

Preparation sorts primitive bounds within angular classes. Every replay computes
the current maximum source norm and five device-only active prefixes. A
primitive remains in the prefix if it can contribute as either an output or a
source. Target admission compares the global maximum source norm with its
actual threshold; source admission uses the global maximum output norm and
maximum target multiplicity. The rectangular/triangular
work inventory is constructed from those frozen prefixes. Source counting,
tail construction and hierarchy updates execute in stream order and are
refreshed on every replay.
Small geometry bounds cannot restore a quadratic traversal over the entire
unpruned primitive inventory. Prefixes are recomputed in captured graphs too;
changing density magnitude never relies on pointer identity. All ordinary E,
density and output mappings remain complete, including primitives excluded from
the current source traversal.

This recipe additionally charges one FP64 bound per primitive and a device
census containing prefix lengths, source-norm bin counts and tail bounds. Its hierarchy and bound storage must fit the same owner's
remaining capacity. An unadmitted optional recipe retains the existing
incumbent provider and its documented screening contract; inspect diagnostics
when requiring a particular recipe. An unset selector requests `primitive`.

### Reference recipe lifetime and resources

The reference recipe's sorted active prefix excludes only primitives whose original pair cannot
participate in any uniform source under the same incumbent geometry test. Full
storage and pair locations remain available to density, reconstruction and the
residual provider. Hierarchy tails are zero-filled. Subset-system replay never
reads uninitialized density for systems outside its window. Same-angular tasks
include the diagonal once and triangular reciprocal half once; cross-angular
tasks use rectangles. Cursor arithmetic is checked without relying on double
precision integer identity above 2^53.

The reference hierarchy shares the direct plan's remaining device capacity,
after retained normal owners are charged. There is no additional fixed MD byte
cap. If extra hierarchy storage does not fit, the admitted incumbent MD pipeline
remains usable. Allocation/preparation failure rolls back the optional MD owner
and retains the normal provider. Private KS shape estimates request a 128-MiB
default reservation; that estimate is not a limit on an explicitly budgeted
owner. Public resource-ledger plans still reserve no optional MD storage and
retain the normal provider. No user budget is bypassed. Plans freeze selection
and immutable geometry; density and hierarchy are refreshed for every replay.
Geometry changes require the existing plan invalidation/reconstruction rules.
The [owner-capacity decision](../../.agents/notes/implemented/performance/2026-10-19-md-j-owner-capacity.md)
records the distinction between the private reservation and actual admission.

`GENERATIVEQC_MD_J_WORK_COUNTS=1` retains existing `MD_J_WORK` counters and adds
`MD_J_ENGINE_WORK` tile/early-rejection counters. Compare roots, directional
contractions, residual work and tile visits along with complete endpoint time.
Memory-bounded traversal alone is not evidence of work or performance improvement.

Source lanes form a two-dimensional bra/ket batch. Each source has one radial
recurrence owner; component lanes read that source's strided compact simplex.
The common Coulomb prefactor is folded into the Boys seed scale before the
linear radial recurrence, sharing it across both admitted directions and
removing one multiplication per output dot. This changes FP64 rounding order;
the unscaled generated entry point remains available for independent comparison.
Compact FP64 radial evaluation opts into the shared Boys owner's reciprocal
reuse: one square root supplies the erf seed and one reciprocal supplies all
upward recurrence orders. Other Boys consumers retain their original arithmetic.
For orders five through eight with argument below six, the compact evaluator
uses the existing positive incomplete-gamma series owner instead of the
alternating series. A bounded series failure falls back to the shared Boys
owner. This does not extend the series region into the established upward
recurrence domain. Both changes retain FP64 storage and accumulation and require
independent high-precision seed/radial and complete energy/force qualification.
Bra partials remain in registers across ket batches, then reuse the radial
workspace for reduction. Ket partials reduce across contiguous bra shuffle
groups before a single writer updates each shared element. Global atomics
publish the tile's outputs. Screened and padded lanes still participate in
every block barrier and shuffle; directional admission never changes their
synchronization obligations. The scalar recurrence remains compiler-owned and
is not an instruction-for-instruction copy of GPU4PySCF. The
[execution-layout decision](../../.agents/notes/implemented/performance/2026-10-19-md-j-quartet-lane-scheduling.md)
records the qualification and reasons for this decomposition.

The component-lane selector also reserves enough shared memory for a complete
bra/ket source-lane rectangle. A radial footprint must not force a tiny tile
that leaves most of its own quartet lanes idle. This utilization invariant
applies to both hierarchy recipes; it does not change mathematical screening.
The [complete primitive decision](../../.agents/notes/implemented/performance/2026-10-19-primitive-complete-md-j.md)
records its scientific bound, schedule rationale and qualification evidence.
On sm_120, a shape-derived launch bound targets the resident blocks that fit
in a 96-KiB shared-memory envelope, capped at eight blocks. It lets the compiler
trade registers against occupancy according to each angular shape. Other
architectures retain the compiler default. The hint does not change tile sizes,
storage admission, or scientific screening. Qualification must check register
spills and complete endpoint performance, not just nominal occupancy.

The [target-budget decision](../../.agents/notes/implemented/performance/2026-10-20-md-j-target-budget.md)
records the actual-target allowance and staged threshold qualification. The
[threshold/residency decision](../../.agents/notes/implemented/performance/2026-10-20-md-j-threshold-residency.md)
records hierarchy reuse and architecture-specific launch-bound qualification.

## Validation and reproducible acceptance

The host gate compiles the actual generated math, public scheduling contracts,
extracted production preparation/kernel bodies and retained E/density/projection
bodies. It checks independent libcint ERIs and full J matrices, signed general
contractions, Cartesian/spherical normalization, nonsymmetric and unrestricted
densities, same-pair factors, threshold neighbors, overflow, padded tails,
compacted prefixes, changed-density/window replay, work conservation and optional
allocation/copy/preparation failure rollback. The native-body scheduling probe
uses host threads and barriers: it is not a CUDA compiler, device racecheck or
performance result. PySCF and the test dependencies must be installed for the
independent science gate; missing/skipped tests do not qualify it.

From a development checkout with a verified compiler cache on PATH:

```sh
compiler_cache="$(command -v sccache || command -v ccache)"
"$compiler_cache" --version
export PYTHONPATH=python:tests/python:.
python benchmarks/md_j_engine_acceptance.py --timeout 1500 \
  --output .artifacts/md-j-host.json
```

A prepared NVIDIA CUDA build must compile the normal native target, including
`direct_md_j.cu` and generated `generated_md_j_engine.cuh`. Use the repository's
[CUDA validation procedure](../maintainer/validation.md) and existing finite
Slurm allocation policy. Set `GENERATIVEQC_LIBRARY` to that build's native
library, `CUDA_PATH` to its toolkit, and record the exact source revision:

```sh
export GENERATIVEQC_LIBRARY=/absolute/path/to/native/libgenerativeqc.so
export CUDA_PATH=/absolute/path/to/cuda
export LD_LIBRARY_PATH="$CUDA_PATH/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export GENERATIVEQC_BENCHMARK_SOURCE="$(git rev-parse HEAD)"
srun --gres=gpu:1 --ntasks=1 --cpus-per-task=4 --time=00:30:00 \
  python benchmarks/md_j_engine_acceptance.py --gpu --timeout 1500 \
  --upstream-root /path/to/clean/pinned/gpu4pyscf \
  --output .artifacts/md-j-gpu.json
```

The optional upstream checkout must be clean at the exact pin and have its own
extension built. The upstream comparison explicitly uses a positive near-unscreened
`direct_scf_tol=1e-30` (upstream takes its logarithm) and native cutoff zero,
recording both. This is a numerical-overlap comparison, not evidence of matched
nonzero screening or exact public-AO mask equivalence. The native suite separately tests that mask,
tight-budget fallback, K preservation and HF/UHF/PBE0 RKS/UKS strict/Fast energy
and force endpoints. JSON retains gate statuses, source hashes, raw logs, JUnit
properties and timing scope. Host-only success leaves CUDA gates `not-run`.

The required `cuda_graph_replay` gate captures the actual prepared resident
enqueue on its owner's stream, instantiates the graph, and replays it against
stable device buffers. It compares ordinary execution and independent AO
references after changed, zero and tail densities. Density upload and result
publication remain outside capture. Diagnostic work-count downloads must be
disabled: those deliberately synchronize with the host and are not part of the
captured production region. A skipped or missing graph test fails CUDA
qualification rather than silently passing it.
The inspected node set permits kernels, memsets and verified same-device copies;
host transfers and callbacks are rejected. Retained-K replay and failed-capture
recovery are covered as well. Compile-only NVIDIA CI builds and links the same
bridge through `generativeqc_md_j_resident_probe`, while host doubles test cleanup
and error behavior without representing device execution.

Run device memory/synchronization qualification separately in the same finite
allocation, retaining exit codes and logs, for example:

```sh
export GENERATIVEQC_TEST_MD_J_ENGINE=1
for tool in memcheck initcheck racecheck synccheck; do
  compute-sanitizer --tool "$tool" --error-exitcode 99 \
    python -m pytest -q tests/python/test_md_j_engine_cuda.py \
    -k 'pair_factors_and_one_density_tail or residual_mask or graph_capture_preserves_retained_k' \
    > ".artifacts/md-j-${tool}.log" 2>&1 || exit "$?"
done
```

Run the complete graph suite separately as part of the CUDA gate. Its
changed-density test deliberately invalidates a capture to check recovery;
Compute Sanitizer reports the expected CUDA API errors as failures. The
sanitizer selection above exercises successful graph capture and replay without
suppressing API-error reporting or weakening the recovery tests.

### Full-size cold, warm and moved measurements

`benchmarks/md_j_engine_benchmark.py` is the bounded measurement entrypoint. It
supplies deterministic 96-atom water clusters in STO-3G (224 spherical AO) and
def2-SVP (768 spherical AO), freezes explicit contracted shells and records the
scientific settings before running. No custom molecular input or driver is
required. Inspect the prescription without loading CUDA:

```sh
python benchmarks/md_j_engine_benchmark.py --scope all --dry-run \
  --output .artifacts/md-j-prescription.json
```

In a prepared, finite allocation, run both scopes with the same source and
library environment as the correctness suite. Performance runs require a
Release build with `GENERATIVEQC_CUDA_FAST_COMPILE=OFF`; the driver verifies the
loaded build's device metadata and source identity. Compile-only CI's fast-build
artifact is not a performance-qualified substitute:

```sh
srun --gres=gpu:1 --ntasks=1 --cpus-per-task=4 --time=03:00:00 \
  python benchmarks/md_j_engine_benchmark.py --scope all \
  --upstream-root /path/to/clean/pinned/gpu4pyscf --require-upstream \
  --cold-repeats 3 --warm-repeats 5 --timeout 1800 --total-timeout 10500 \
  --output .artifacts/md-j-performance.json
```

The explicit time limits are ceilings, not predictions. Exhausting either
limit produces retained failure evidence and does not qualify the case. Use
`--cases water32-sto-3g` or `--cases water32-def2-svp` to measure a case separately;
`--scope j` and `--scope energy-force` likewise isolate the two scopes. The
upstream option applies to J; native incumbent/reference energy-force runs use
identical fixed scientific settings.

- **J:** Fresh subprocesses measure one complete raw-J adapter call, including
  its preparation, upload and publication. Warm samples retain the owner and
  record both complete adapter wall time and separate resident device-event
  time. Test-bridge loading/provenance overhead is identified rather than
  relabelled as production preparation. A separate direct PySCF contraction
  checks numerical results without an in-core four-index ERI tensor. The
  optional pinned GPU4PySCF comparison uses the documented near-unscreened
  threshold representations and includes public-AO phase conversion.
- **SCF energy and forces:** Fresh subprocesses construct a Calculator and
  prepared owner, then execute a complete strict PBE0-RKS energy/analytic-force
  solve. The same owner performs repeated unchanged-geometry solves and one
  deterministic moved-geometry solve. Geometry transfer, invalidation and
  rebuilding are inside the moved timer. Both engines use FP64, the same
  explicit basis, fixed grid, SCF tolerances and screening; converged-density
  reuse is disabled. Synthetic J benchmark densities never enter these solves.
- **Execution evidence:** Separate instrumented endpoint workers retain actual
  native source counters and must reproduce the clean workers' energy/forces.
  Instrumented timings do not enter performance summaries. Every clean phase
  must converge on CUDA and pass energy, force and physical-residual gates.
- **Provenance and evidence:** Receipts retain the source manifest identity
  checked against the loaded native library ABI, source/library hashes, upstream
  pin/binary hash, complete raw repetitions, process logs, matrix/force data,
  input hashes and memory observations. Sampled GPU memory is identified as a
  lower bound on peak, not an exact peak measurement. Keep the JSON and its
  unique adjacent artifact directory together.

Benchmark workers remove inherited experimental/diagnostic `GENERATIVEQC_*`
controls, retain the explicit library and force `GENERATIVEQC_PROFILE=off`.
The driver records the effective controls and removed names; unknown environment
values and remote-cache credentials are not copied into receipts. Timing rejects
`CUDA_LAUNCH_BLOCKING=1`. These rules keep local tuning caches and diagnostic
downloads from silently changing the measured prescription.

The full-size energy/force comparison is incumbent/reference regression parity
at a declared fixed grid. It is not an independent grid-converged production
accuracy ranking. The driver reports cold, warm and moved medians and variability;
any minimum-speedup threshold is declared before measurement. Neither J timing
nor successful receipt collection authorizes default promotion.

Real CUDA compilation/execution, sanitizer receipts, graph-capture replay and
source-matched full-size cold/warm endpoint performance require separate
qualification receipts. The correctness runner
executes graph capture when `--gpu` is selected; full-size measurements use the
separate bounded benchmark driver above. Neither runner authorizes default
promotion. Compare upstream and GenerativeQC with the same basis phase,
contraction, density, screening contract and cold/warm timing boundary, including
preparation and peak memory. No speedup or completed GPU acceptance follows from
host tests or instrumented resident timing. See the
[default-promotion inventory](../../manifests/maintenance/default_promotion_inventory.json)
and [performance engineering policy](../maintainer/performance_engineering.md).
