# Reference-aligned MD Coulomb engine

The resident direct J/K owner has a complete, opt-in MD **J-value** engine selected
at plan construction by `GENERATIVEQC_MD_J_ENGINE=reference`. The default is
`incumbent`; setting that value explicitly restores the existing MD source
consumer. Invalid values are errors. Diagnostics identify the selected owner as
`md-j-reference/retained-k` or `md-j-hermite/retained-k`. Inspect diagnostics:
requesting an optional engine is not proof that its storage was admitted.

This engine uses the architecture of official GPU4PySCF at
[`9af748a71ca8f4881c604332c2f0cc1632e033ac`](https://github.com/pyscf/gpu4pyscf/tree/9af748a71ca8f4881c604332c2f0cc1632e033ac).
It is integrated into the ordinary native provider, with generated scalar
mathematics and native lifetime/scheduling ownership. Neither PySCF nor CuPy is a
production dependency. Upstream source hashes and consumers are registered in
`upstream/manifest.json`; the Apache-2.0 license and modification attribution are
in `LICENSES/gpu4pyscf-Apache-2.0.txt` and `THIRD_PARTY_NOTICES.md`.

## Architecture correspondence

All upstream paths below are relative to `gpu4pyscf/` at the pinned revision.
The adaptation deliberately preserves GenerativeQC's public-AO screening
contract; it does not claim binary layout or screening equivalence with upstream.

| Stage | Pinned GPU4PySCF | Integrated GenerativeQC equivalent and compatibility choice |
| --- | --- | --- |
| Decontraction and sorting | `scf/j_engine.py`: `SortedGTO.from_mol(decontract=True)`, sorted basis and `C D C^T` | `MdJHost` keeps original public-shell identity plus every ordered primitive product. Resident sparse weighted E transforms factor the contraction, normalization and Cartesian/spherical maps without materializing dense C. This reuses the established transform pipeline. |
| Pair layout | `j_engine.py`: `(li,lj)` pair mappings sorted by `q_cond`, packed `pair_loc` | Within total-angular classes 0–4, a deterministic active-prefix, descending-q, primitive-ID permutation drives packed Hermite intervals. `md_j_engine_pair_loc` rewrites only Hermite offsets. Original shell/primitive identities and transform slots stay valid. Total-angular grouping is sufficient after the separate bra/ket E transforms; it is not upstream's raw-shell ordering. |
| Density precontraction | `lib/gvhf-md/contract_Et.cu`: `Et_dot_dm` | Existing `md_j_density` contracts the actual public density into resident primitive Hermite density using cached E. UKS uses alpha+beta; nonsymmetric density retains both AO directions. |
| Hierarchical screening | `j_engine.py` / `md_contract_j.cu`: float32 log q and q+dm, widths 1/2/4/8/16/32 | Six FP64 linear bound levels retain q, qD, maximum leaf minimum, and density. Coarse rejection preserves the leaf's left-associated upward-rounded `(q_out*q_in)*density_in` expression. Precomputed qD is retained as hierarchy metadata but is not substituted at rounded boundaries. |
| Exact admission | Upstream applies its own shell-pair cutoff contract | `md_j_engine_direction` retains the existing public-AO minimum-product and directional density allowance. Partially masked pairs still use the established exact residual source. Thus upstream log thresholds and half factors are not silently imported. |
| Task/tile scheduling | `j_engine.py` scheme selection; `md_j.cuh` and `md_contract_j.cu` hierarchical task consumers | Checked 64-bit rectangular/triangular inventories, atomic device cursor, four-warp tiles, shared density/potential caches and per-warp compact radial scratch. Angular/workspace-aware tile widths respect a conservative 32-KiB target and 48-KiB hard cap. All 15 canonical angular class pairs, including (4,4), use this backend. Shapes are conservative and await device tuning. |
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

For each original shell pair, the minimum public-AO Schwarz bound proves that
all public quartets pass; the maximum proves that none pass. Only the uniform
region enters the Hermite source consumer. The existing residual source retains
the exact per-public-AO mask for the partial region. Density omission still uses
`min(screening_tolerance, 1e-12)` apportioned by the **global contracted-pair
census**, downward-rounded allowance and upward-rounded contribution. Forward
and reverse admission remain independent. An exactly zero density has no source
work; a zero screening tolerance disables density omission of nonzero terms.

The sorted active prefix excludes only primitives whose original pair cannot
participate in any uniform source under the same incumbent geometry test. Full
storage and pair locations remain available to density, reconstruction and the
residual provider. Hierarchy tails are zero-filled. Subset-system replay never
reads uninitialized density for systems outside its window. Same-angular tasks
include the diagonal once and triangular reciprocal half once; cross-angular
tasks use rectangles. Cursor arithmetic is checked without relying on double
precision integer identity above 2^53.

The reference hierarchy shares the existing optional 128-MiB resident cap, after
mandatory owners are charged. If extra hierarchy storage does not fit, the
admitted incumbent MD pipeline remains usable. Allocation/preparation failure
rolls back the optional MD owner and retains the normal provider. No cap is
raised and no user budget is bypassed. Plans freeze selection and immutable
geometry; density and hierarchy are refreshed for every replay. Geometry changes
require the existing plan invalidation/reconstruction rules.

`GENERATIVEQC_MD_J_WORK_COUNTS=1` retains existing `MD_J_WORK` counters and adds
`MD_J_ENGINE_WORK` tile/early-rejection counters. Compare roots, directional
contractions, residual work and tile visits along with complete endpoint time.
Memory-bounded traversal alone is not evidence of work or performance improvement.

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
ccache --version
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

Run device memory/synchronization qualification separately in the same finite
allocation, retaining exit codes and logs, for example:

```sh
export GENERATIVEQC_TEST_MD_J_ENGINE=1
for tool in memcheck initcheck racecheck synccheck; do
  compute-sanitizer --tool "$tool" --error-exitcode 99 \
    python -m pytest -q tests/python/test_md_j_engine_cuda.py \
    -k 'pair_factors_and_one_density_tail or residual_mask' \
    > ".artifacts/md-j-${tool}.log" 2>&1 || exit "$?"
done
```

Real CUDA compilation/execution, sanitizer receipts, graph-capture replay and
source-matched full-size cold/warm endpoint performance are separate outstanding
qualification gates until actual evidence is recorded. The scoped runner exposes
graph capture and complete performance as `not-run`; it cannot authorize default
promotion. Compare upstream and GenerativeQC with the same basis phase,
contraction, density, screening contract and cold/warm timing boundary, including
preparation and peak memory. No speedup or completed GPU acceptance follows from
host tests or instrumented resident timing. See the
[default-promotion inventory](../../manifests/maintenance/default_promotion_inventory.json)
and [performance engineering policy](../maintainer/performance_engineering.md).
