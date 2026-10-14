# Proposal: wider VV10 pair tiles for large active domains

Status: experimental, default-off; qualification pending
Date: 2026-10-14

An isolated four-arm RTX 5090 experiment found that eight rows per lane with
row-input retention reduced 320k-point feature/geometry intervals by roughly
9%/11%, but regressed the 10k-point domain by about 40%. The complete 96-atom
WB97M-V population remains the target; isolated timings do not promote it.

The explicit `parallel-wide` selector shares the existing canonical pair roots,
admission and output lease guards. Two mutually exclusive device-count branches
keep four rows below 65,536 active points and eight rows above. It allocates no
new global scratch and performs no host count read. FP64 reduction order may
change, requiring independent oracles, adversarial canaries and sanitizer gates.

The existing default and ordered fallback remain available. A full same-library
off/on population, including changed geometry, is required before promotion.
