# Complete range-separated DF after stable force response — 2026-10-15

This evidence supersedes the practical-auxiliary **numerical limitation** in
[`../range-separated-df-20261015/`](../range-separated-df-20261015/). The earlier
negative observations, raw data and numerical-only GPU build boundary remain
historical evidence; they are not rewritten into successful measurements.

To respect the repository-wide evidence byte budget, original raw JSON, exact
qualification receipts, harnesses, both source identities, the historical patch
and the original historical README are retained losslessly in `evidence.json.xz`.
`storage.json` authenticates encoded and decoded bytes. Restore all original
relative names into ignored local storage before using the paths below:

```bash
python benchmarks/results/range-separated-df-stable-20261015/verify.py \
  --extract build/validation/range-df-evidence
```

The final implementation patch is recoverable directly from this PR's Git tree
against base `5f80ba629b5f20412215cdc690f8fa458ac3fdf8` and measured source
`e7e7654930e8c41a349920d18aef41bbe9d3ebf9` with
`git diff --binary <base> <measured-source> -- src python tools tests`; its digest and
per-file hashes are retained in the restored final source identity.

## Matched endpoint

Baseline is full-range DF-J/K plus exact Direct LR K (`direct`); candidate also
fits LR K with its own erf metric (`df`). These are distinct Hamiltonians, each
checked against independent libcint energies using the same normalized metric
and relative cutoff. DF-versus-Direct energy agreement alone is not acceptance.

- Water geometry and harness match the historical evidence: WB97M-V/RKS,
  spherical def2-SVP (24 AOs) or def2-TZVP (43 AOs), retained cc-pVDZ-JKFIT,
  FP64, Hcore guess, energy/density tolerances 1e-10/1e-8, screening 1e-14,
  relative metric cutoff 1e-10, one CPU/BLAS thread.
- Version-1 unpruned `24 × 6 × 12` per-atom grid, 5,184 water points; **not**
  the production default grid and not a quadrature-convergence qualification.
- Each selector/endpoint uses a separate fresh process and prepared owner.
  Cold total includes preparation; warm and changed-geometry results are the
  medians of three samples, with raw ranges and SCF work retained in summaries.
  Compiler caches are reused; imports and evidence hashing are outside timing.
- Complete energy-only and energy-plus-force endpoints are separate. Geometry
  alternates the declared H1 displacement and original coordinates, including
  provider rebuilding and SCF. No isolated-kernel or force-only ratio is used.
- CPU is AMD EPYC 7K62, normal Release, scalar native linear algebra. The
  implementation patch, source hashes and reported library hashes bind evidence
  to the measured implementation, not only to an unchanged Git HEAD.

## CPU conclusion

Seconds; baseline → complete DF, speedup in parentheses. The fixed-geometry
energy replay already caches Direct LR values, so removing preparation does
not improve that particular endpoint. Geometry changes and forces benefit.

| Orbitals | Endpoint | Cold including preparation | Warm | Changed geometry |
| --- | --- | ---: | ---: | ---: |
| SVP | Energy | 4.378 → 3.483 (1.26×) | 0.825 → 0.834 (0.99×) | 3.737 → 2.845 (1.31×) |
| SVP | Energy + forces | 6.619 → 4.814 (1.37×) | 3.021 → 2.107 (1.43×) | 5.929 → 4.110 (1.44×) |
| TZVP | Energy | 25.488 → 5.014 (5.08×) | 1.129 → 1.173 (0.96×) | 25.215 → 4.100 (6.15×) |
| TZVP | Energy + forces | 40.407 → 7.168 (5.64×) | 15.729 → 3.258 (4.83×) | 40.044 → 6.197 (6.46×) |

These are descriptive three-repeat, noninterleaved observations, not a passed
five-pair performance-promotion campaign. Warm energy-only medians are slower by
1.0166% (SVP) and 3.8904% (TZVP); the earlier retained campaign also observes
0.9267% and 3.4627% slowdowns. Both campaigns have nonoverlapping within-arm
ranges, so the repeated adverse endpoint must not be dismissed as noise. The
generic comparison gate requires at least five interleaved pairs and improvement
beyond max(2%, twice the summed relative MAD); no campaign-specific tolerated
regression was declared. Numerical qualification of the distinct fitted
Hamiltonian is separate from accepting this replay tradeoff or promoting a
performance default. That acceptance remains unresolved; force gains do not
waive it, and no new optimization or GPU campaign is implied here.

The stable candidate's maximum warm-force replay changes are 7.69e-10 and
5.50e-11 Eh/bohr for SVP and TZVP respectively, replacing the earlier 1e-6
instability. The initial cold force-component oracle error at SVP is 1.44e-9
Eh/bohr (formerly 4.41e-7), without loosening the 3e-7 gate or changing cutoff.
See `oracle-all-cpu.json` for **every** measured cold/warm/geometry energy and
the H1-z plus independent directional force checks, including moved geometry.

The moved SVP geometry exposes an unresolved **reference finite-difference
step**, equally in the old mixed and new fitted models: a 2e-4 Bohr component
step differs by about 9.7e-7 Eh/bohr, while 1e-4 and 5e-5 both agree with the
analytic result near 1e-9. The final all-endpoint oracle therefore uses 1e-4
for that component and retains 2e-4 for the directional displacement, without
changing the force gate. Both the original failing coarse-step receipts
(`oracle-all-cpu-step-2e-4.json`) and the independent reconverged step sweep
(`moved-finite-difference-step-check.json`) are preserved. Do not confuse this
common reference-step issue with the original LR response error, which also
failed at smaller steps before the whitened-response fix.

### Semantic work and limits

CPU force receipts remove the Direct LR quartet pass entirely: canonical AO
domains 45,150 (SVP) and 447,931 (TZVP) become zero. VV10 still evaluates
26,873,856 water grid pairs; this change does not make VV10 itself a fitted
kernel. CUDA quartet counts are not exported and must remain unknown.

The new truncated-metric CUDA adjoint explicitly charges one retained whitened
tensor plus bounded raw/weight panels. Raw source visits are bounded independent
of tile, but density matrix products repeat across response panels. This is not
a large-molecule, constrained-memory, batch, whole-process memory, or asymptotic
constant-work qualification. The historical water-dimer force rejection at the
unchanged 100,000,000 grid-pair admission boundary still applies.

## CUDA candidate observations

Real-device qualification passes 14 selected endpoint/response tests on node2's
RTX PRO 6000 Blackwell through finite Slurm allocations; the truncated-response
memcheck reports zero errors. CPU host execution is not a substitute for those
checks. The extra broad GPU run also hits an unrelated negative-control test's
old message expectation (`numerical failure` versus `invalid argument`); its
original failure receipt is retained separately from the passing qualification.

Absolute candidate energy-plus-force timings use the numerical-qualification
library with **normal Release O3 DF kernels**, but two unrelated legacy Direct
translation units remain fast-compiled. These are observations for that exact
library, **not normally optimized Direct speedups**. Its hash and compilation
boundary are in the final source identity. No comparative GPU ratio is claimed.
The normal-optimization retry completed Direct JK/force objects but its final
legacy Direct fallback ptxas worker was stopped after 38 minutes. The shared
compiler cache and completed objects remain intact for future qualification.

| Orbitals | Cold including preparation | Warm median | Changed-geometry median |
| --- | ---: | ---: | ---: |
| SVP | 2.143 s | 0.0714 s | 1.184 s |
| TZVP | 2.106 s | 0.1108 s | 1.255 s |

Warm GPU replay takes one SCF iteration, versus two on CPU; geometry replay
takes 25. Do not turn these into a CPU-versus-GPU solver speedup. All fourteen
CUDA measured samples have independent component/directional oracle receipts
in `oracle-all-cuda-numerical.json`, including moved geometry. This is still a
controlled small-grid observation, not default-grid or large-system throughput.

## Reproduce

From the checkout root with the documented CPU/CUDA builds and dependencies,
after restoring the evidence into ignored storage, choose a new campaign directory.
The live CPU wrapper rejects an existing file, directory or symlink before any
endpoint, report, oracle or log opens. Report and unchanged all-endpoint oracle
commands run inside that same fresh campaign; the progress journal and logs
retain their failing stage/status and any partial oracle receipts. The live
endpoint runner reserves each output exclusively,
journals initialization/preparation/runtime failures and hashes the calculator's
actually loaded library outside the preparation timer. Calculator creation and
native preparation remain included in the reported preparation total.

The archived scripts are immutable measurement sources, not safe live rerun
entrypoints. In particular, their library hashes describe the configured
`GENERATIVEQC_LIBRARY` file; they did not query a potentially selected CUDA
profile. No profile substitution is established by the retained records. New
runs use `python -m benchmarks.range_separated_df`, including for a separately
scheduled CUDA run; do not infer a new GPU qualification from this runner repair.
The workload, geometry sequence and numerical oracle gates remain unchanged.

```bash
restored=build/validation/range-df-evidence
export RANGE_DF_HARNESS_DIRECTORY="$restored/range-separated-df-20261015"
export RANGE_DF_EVIDENCE_DIRECTORY=build/validation/range-df-rerun
bash benchmarks/run_range_separated_df_cpu.sh
```

GPU qualification and measurements must use finite Slurm allocations, preserve
the assigned device visibility, and use a node other than maintenance node n3.
Never use the earlier fast-compiled Direct numerical library to claim a
normally optimized Direct speedup.
