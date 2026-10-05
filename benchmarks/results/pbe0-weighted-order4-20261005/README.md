# Order-four weighted Direct force sources

Complete exact-direct PBE0/def2-SVP energy plus analytic-force qualification on one RTX 5090 per finite Slurm job. The frozen composed baseline includes #1830, #1833, #1847 and indexed force scheduling; this is not an unmodified-master timing claim.

The only candidate treatment is compiler-owned precontracted force roots for pppp, dspp, dsds, dpps and ddss. Both arms preserve FP64, spherical basis, grid, convergence, screening, local AO policies and complete force semantics. GPU4PySCF 1.8.1 rebuilds full-density exact Direct J/K and includes moving-grid response. No DF, incremental reference, preliminary-density oracle or iteration normalization is used.

## Complete endpoint observations

| Atoms | Regime | Control seconds | Candidate seconds | Reference seconds | Native SCF counts (control → candidate) |
| --- | --- | ---: | ---: | ---: | --- |
| 48 | cold | 125.009007 | 120.938740 | 51.947551 | [25] → [24] |
| 48 | warm | 10.780317 | 10.130371 | 6.074627 | [1, 1, 1, 1, 1] → [1, 1, 1, 1, 1] |
| 48 | moved | 55.390709 | 54.336738 | 48.496782 | [12] → [12] |
| 48 | moved-warm | 10.777877 | 10.111430 | 4.014557 | [1, 1, 1, 1, 1] → [1, 1, 1, 1, 1] |
| 96 | cold | 288.648141 | 271.032919 | 78.487122 | [30] → [27] |
| 96 | warm | 32.509183 | 29.012620 | 15.450827 | [1, 1, 1, 1, 1] → [1, 1, 1, 1, 1] |
| 96 | moved | 129.895512 | 126.084906 | 86.536558 | [12] → [12] |
| 96 | moved-warm | 32.554324 | 28.810543 | 10.149312 | [1, 1, 1, 1, 1] → [1, 1, 1, 1, 1] |

Warm medians improve 6.03% at 48 atoms and 10.76% at 96; moved-warm medians improve 6.18% and 11.50%. All eight candidate regime medians are lower. Cold SCF trajectories differ, so their aggregate differences are not attributed to the force-only change. Moved endpoints use 12 actual Fock builds in both arms at both sizes.

The 96-atom stationary derivative warm median falls from about 14.34 s to 10.68 s. Grid geometry remains about 8.8 s. The remaining matched-reference ratios are 1.67×/1.88× warm, and about 2.52×/2.84× moved-warm; #1895 is not closed. Five observed samples and all outliers are retained, not replaced by per-iteration costs.

## Scientific and work gates

All 288 same-geometry native/reference repeat pairings pass unchanged 1e-8 Eh and 1e-7 Eh/Bohr gates. Worst absolute errors are 1.0914e-10 Eh and 3.5214e-11 Eh/Bohr. Full native histories, actual Fock counts, physical residuals, vectors and failed-retry flags remain in the compressed original JSON files.

Each full-range shell task retains the exact shell/AO predicates and canonical orbit weights. The compiler differentiates the external-weight contraction before scatter; PPPP uses two additive roots within the existing 64-component lowering limit. There is no extra resident allocation. Other classes and radial/combined consumers retain their bounded implementations.

Original endpoint ledgers retain 1,179,648/2,359,296 grid points, 4,608/9,216 AO tiles and 1,330,642,944/10,758,389,760 pair productions per geometry. Pair-production counts are semantic work, not FLOPs. Integral root execution and spill traffic were not instrumented in the timed endpoint; prior shell admission counts are upper bounds and are not borrowed as exact work for these densities.

The independent host Hermite displaced-value gate checks 1,761,696 coordinates (worst error 3.36e-10; adapter error 1.67e-16). GPU Slurm5880 through-f values/derivatives and memcheck/initcheck pass with zero sanitizer errors. See the retained build/test receipts; master transplant qualification is reported separately in the code PR.

The validation envelope accepts scientific equivalence. Its generic automatic-provider promotion assay is explicitly unmeasured: these runs are alternating complete arms across sizes, not an interleaved fixed-state assay with isolated whole-build cost and allocator peaks. Observed complete endpoint results above are retained separately. No provider/profile registry is promoted by this evidence.

## Reproduction and exact identities

- Control commit: `9ba032c783addfeede96890c894b7cc9447cde95`; source `1c0b0ea35a9d34f179ab9b48c64685d70e1369a7bfc9455906fc83f8ccb657fc`.
- Candidate commit: `b3acd80f706354276ffb18154e70ae5dd8998383`; source `fdde4fb47a289075bd7d4ebd02779b886d090963924f0db5f077d20f0de884f9`.
- Native library hashes and all native/AOT/driver receipts are retained in `receipts.json` and original endpoint JSON. Job5884 runs 48 control→candidate and 96 candidate→control on the assigned GPU, with a fresh stationary cache per arm and size.
- Reconstruct control source by applying `control-source.patch.gz` to base `9d0f4fc0ddbe9c1dc4539cd4f6ad8a6e019ba990`, then apply `candidate-source.patch.gz` for the candidate. `reconstruction.json` records the independently checked control source identity. Driver text preserves exact historical paths; use equivalent isolated paths when reproducing.
- Input basis: existing `benchmarks/results/pbe0-def2-svp-20261003/def2-svp-ho.json`; geometry, settings and basis identity are embedded in each original protocol.
- Run `PYTHONPATH=python:. python benchmarks/results/pbe0-weighted-order4-20261005/verify.py` from the repository root. It verifies checksums, all pairings, work, source receipts, phase inventories and unnormalized timing summaries without a GPU.

Original ignored artifacts remain under `/data/jzzeng/qc-1895-order4-weighted-forces-20261005/.artifacts/endpoint-5884/`. No external archive or release is needed.

## Master transplant qualification

The code transplant on master base `a32b6cfd63ad07d66f8494591031822e1483e69f` was built and tested at `dfeabdc0d6b0985c89e2a7c88442504c32efa801`, source `a1236412fe2b155036897275049af74e39a888765a73b8f9d32d70b5459b92e0`. Slurm5907 passed native through-f values/derivatives and memcheck/initcheck with zero errors. All 47 focused host tests pass; native targets and six stationary AOT modules build with verified ccache launchers. `shipping-transplant.json` retains exact receipts and the six byte-identical scientific/dispatch source hashes. This is transplant qualification, not another complete endpoint campaign. The shipping branch adds evidence and rationale after the frozen tested code revision.
