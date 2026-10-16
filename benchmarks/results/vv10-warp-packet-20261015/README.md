# Opt-in VV10 rotating-partner qualification

This is historical, source-matched evidence for the explicit
`GENERATIVEQC_VV10_PAIR_SCHEDULE=parallel-packet` experiment. Default selection
does not change. The base is master `5f80ba629`; `summary.json` identifies the
actual patched native source inventory and library, not merely the base commit.

The compressed evidence retains the deployment manifest, executable experiment
drivers, finite n4 Slurm receipts, admission/oracle/sanitizer XML, and complete
3/12-atom WB97M-V off/on populations. Both arms use the same Release library,
FP64, canonical pair roots, original 48/16/32 grid, five warm samples per
geometry and original 1800-second per-point deadline. All native results must
pass the independent reconverged GPU4PySCF energy/force gates. These are small
qualification populations, not completion of the outstanding 48/96-atom work.
Reference cold intervals include the ordinary reference construction; they are
accuracy inputs here, not a claim of native/reference performance parity.

The separate resident-consumer experiment alternates three calls per arm at
10k/80k/320k fixed-grid points. Its complete wall interval includes native plan
and force-owner preparation, geometry/weight and density/gradient transfers,
local scales, admission/compaction, pair traversal, seed packing, output download
and owner retirement. A nested execution interval excludes preparation. Input
generation, library import and result assertions are outside both intervals.
Exact-component reuse is disabled; counters must show one pair sweep and zero
cache hits per call. Report actual nonzero partners, observable positive-zero
rows and scalar pair work, not just nominal point counts. This consumer timing
does not include molecular grid/AO production, SCF or nuclear force assembly.

The original isolated tuning sweep is diagnostic only and is not an endpoint
claim. Neither it nor a small molecular win promotes default selection. Larger
original-protocol populations, changed geometries and repeated interleaved
molecular campaigns remain necessary before promotion. In particular, the
first off/on cold samples can differ in artifact-cache state; do not interpret
their ratio as a causal cold-start speedup.

Run `python benchmarks/results/vv10-warp-packet-20261015/verify.py` to audit the
archive, independent gates, semantic work and all reported medians. This is an
opt-in historical evidence verifier, not a required-CI source fingerprint test.
