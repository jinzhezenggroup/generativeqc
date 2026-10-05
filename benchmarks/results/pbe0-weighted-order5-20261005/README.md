# Order-five weighted Direct force sources

Incremental complete exact-direct PBE0/def2-SVP E+F qualification of dppp, dpds and ddps precontracted derivative roots. Control already includes order four and the composed #1830/#1833/#1847/indexed-policy baseline. These timings do not describe unmodified shipping master.

## Complete endpoint observations

| Atoms | Regime | Order-four seconds | Order-five seconds | GPU4PySCF seconds | Native SCF counts (control → candidate) |
| --- | --- | ---: | ---: | ---: | --- |
| 48 | cold | 124.331621 | 116.433140 | 51.377367 | [25] → [23] |
| 48 | warm | 9.987770 | 9.286528 | 5.998538 | [1, 1, 1, 1, 1] → [1, 1, 1, 1, 1] |
| 48 | moved | 54.055510 | 53.350329 | 47.621540 | [12] → [12] |
| 48 | moved-warm | 10.006721 | 9.200909 | 3.945309 | [1, 1, 1, 1, 1] → [1, 1, 1, 1, 1] |
| 96 | cold | 283.948644 | 244.975738 | 78.143370 | [30] → [25] |
| 96 | warm | 28.913819 | 26.989124 | 13.629942 | [1, 1, 1, 1, 1] → [1, 1, 1, 1, 1] |
| 96 | moved | 133.021434 | 123.166182 | 92.675848 | [13] → [12] |
| 96 | moved-warm | 28.777184 | 26.014840 | 10.086490 | [1, 1, 1, 1, 1] → [1, 1, 1, 1, 1] |

96-atom complete warm improves 6.66%, and moved-warm improves 9.60%. The corresponding 48-atom gains are 7.02% and 8.05%. All warm native samples use one actual SCF iteration. Cold trajectories differ at both sizes; moved differs at 96 atoms. These aggregate differences are retained without assigning them to a force-only treatment or dividing by iterations. All outliers remain in the original samples.

The 96-atom warm derivative component falls from 10.664558 to 7.632423 s. Geometry response remains about 8.80 s. Complete candidate warm/moved-warm are 1.98×/2.58× the matched reference; this does not achieve #1895 parity or the ten-second goal.

## Science and work

All 288 same-geometry native/reference sample pairings pass the unchanged 1e-8 Eh / 1e-7 Eh per Bohr gates. Maximum absolute errors are 1.0414e-10 Eh / 3.1802e-11 Eh per Bohr. Raw JSON retains force vectors, native SCF histories, actual Fock counts and physical residuals. Reference GPU4PySCF rebuilds exact Direct J/K from full current density and includes moving-grid response. Basis, grid, screening, convergence and FP64 are unchanged; no DF/COSX substitution.

WeightedIntegralIR contracts density-derived weights before AD. The 162/108 component classes use additive partitions of at most 64 components. Existing shell/AO predicates, canonical orbit weights and independent J/K outputs are retained, as are bounded f-containing, range-separated and combined-source fallbacks. No extra resident allocation is added. Pair geometry and Boys reuse already existed and are not counted as a new change.

Original ledgers retain 1,179,648/2,359,296 grid points, 4,608/9,216 geometry tiles and 1,330,642,944/10,758,389,760 Becke pair productions at 48/96 atoms. These are semantic counts, not FLOPs. Integral root execution and spill traffic were not instrumented in clean timing; earlier angular replays are not substituted as exact production shares.

The host gates contain 54,864 independent Hermite displaced-value coordinate checks and 746,496 retained AO-adapter comparisons (801,360 total). Their maximum errors are 6.84327e-10 and 1.11022e-16 respectively; the adapter comparisons are not independent oracle checks. Composed Slurm5905 native through-f, memcheck and initcheck passed. Separate shipping qualification and source equivalence are retained in shipping-transplant.json; its endpoint_measured=false deliberately distinguishes code qualification from composed timings.

The numerical evidence is accepted. The generic automatic-provider performance assay is not run: this campaign alternates whole arms across sizes rather than interleaving fixed states with isolated build/allocator peaks. It does not promote a provider/profile registry.

## Reproduction

- Control: `b3acd80f706354276ffb18154e70ae5dd8998383`; candidate: `8f90f0ac33ee6bd8a755fb85d71291e1f66c5884`.
- Composed campaign: finite Slurm5905 on node1, one RTX 5090. 48 atoms runs control then candidate; 96 runs candidate then control. All stages use the assigned CUDA_VISIBLE_DEVICES and a fresh stationary cache per arm/size.
- Apply control-source.patch.gz then candidate-source.patch.gz to reconstruction_base in reconstruction.json. Both reconstructed scientific source identities were checked independently. Build with explicit CXX/CUDA ccache launchers and checkout-root CCACHE_BASEDIR, then run retained drivers under finite Slurm allocations.
- Input basis is benchmarks/results/pbe0-def2-svp-20261003/def2-svp-ho.json; geometry and protocol are embedded in every raw JSON. Drivers retain historical paths and can be adapted to isolated checkouts.
- Run `PYTHONPATH=python:. python benchmarks/results/pbe0-weighted-order5-20261005/verify.py` to verify hashes, all pairings, work counts, actual histories, source receipts and unnormalized phase summaries without a GPU.
- Full ignored artifacts remain at /data/jzzeng/qc-1895-order5-weighted-forces-20261005/.artifacts/endpoint-5905 and gpu-5905. No Release or external archive is used.

## Source-matched next-step profile

Slurm5917 captures the first original-geometry warm call on the exact composed
order-five source/library. Its 26.188054 s NVTX range contains 23.409499 s of
summed GPU kernel durations. Name-based groups are: two-electron force 6.552472 s,
J/K values 4.427064 s, Becke 4.036176 s, AO geometry force 2.667037 s, XC kernels
1.937030 s, AO jets 1.117686 s, one-electron force 1.078898 s, matrix products
0.790117 s, and KS diagnostics 0.482261 s. These device sums are not exclusive
endpoint wall components or angular-class shares. The complete profiled campaign
also passes 72 independent pairings; none of its timings is a clean speedup claim.

Twenty generated streaming classes execute in two disjoint serial regions, first
J then K according to the source dispatch: 2.408810 and 1.974830 s. Native dddd and
other overhead are outside these two generated-class sums. The hot mixed force
kernel uses 255 registers per thread and 256 threads per block in the captured
CUPTI metadata; this does not by itself establish occupancy or spill cost.

The profile supports investigating grouping within admitted force pages and
independent J/K work organization. Becke's atom-log kernel is only 1.500534 s,
so #1950's isolated logarithm scheduling gain cannot close the whole gap. An
ordered-AO experiment qualifies separately on the older baseline; no additive
composition speedup is claimed here. Source/report receipts and the raw local
11 MB Nsight report path/checksum are retained under profile/analysis.json.
