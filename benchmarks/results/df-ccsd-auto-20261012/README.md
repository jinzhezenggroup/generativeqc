# Native DF-CCSD automatic warmup qualification

## Scope and identities

Frozen master: `2a6de26eea1f83e9fd1e8df01c189fea44058b6b`, fetched October 12,
2026. Timed calls use conventional exact RHF + correlation-only DF CCSD(T),
spherical orbitals, DIIS 6, automatic CCSD Q tiling with limit 32, CCSD energy
tolerance `1e-12` and residual tolerance `1e-10`. (T), Lambda, orbital and
nuclear response retain FP64. Imported orbitals, amplitudes and reference-oracle
work are not part of the timed production endpoint.

CUDA executes only in finite Slurm allocations on n1/n4 RTX 5090 GPUs.
The n3 node under maintenance is not used. Compilations use verified ccache;
compilation and output JSON serialization are outside native endpoint timing.
Every cold sample starts a fresh process. Three-repeat force comparisons reverse
mode order in the second repeat. The retained records contain all repeats, not
just the medians or numerically favorable samples.

The frozen explicit-policy timing binary is master plus
`measured-source.patch.xz`. The final auto-policy binary has a different hash
and uses `final-source.patch.xz` plus the added files in
`final-untracked.patch.xz`. Every run records actual executable/library/input
hashes, Slurm job and assigned visibility, its exact invocation and work fields.
Do not claim that the final rebuilt binary produced the frozen timing matrix.
Binaries are retained only as ignored local/compute-job artifacts, not committed
or published as Releases.

## Threshold selection

The coarse ethane230 energy screen tries `1e-3`, `1e-4`, `1e-5`, `1e-6`, `1e-7`;
the fine screen adds `2e-7`, `3e-7`, `5e-7`. Looser is not automatically faster:
`1e-3` increases complete energy time 49.794 -> 51.294 seconds, and `1e-5`
also loses. The complete three-repeat E+F comparator determines the retained
switch, not FP32 convergence or the number of counted iterations.

| Ethane230, three-repeat median | FP64 | Switch `1e-7` | Switch `5e-7` |
| --- | ---: | ---: | ---: |
| Complete native E+F, s | 249.523569 | 247.303254 | 247.337432 |
| Complete CCSD including replay, s | 25.661352 | 23.326304 | 23.437707 |
| FP32 / FP64 graph evaluations | 0 / 18 | 12 / 10 | 10 / 11 |

Keep the existing default switch `100.0 * 1e-9`, approximately `1e-7`.
It gives 9.0995% lower CCSD time and 0.8898% lower complete E+F time against
this master. `5e-7` saves one total evaluation, but is not faster. Its complete
E+F difference from `1e-7` is small; no large speedup claim rests on that tie.
Warmup is capped at 24 evaluations and never supplies the physical certificate.

Maximum paired ethane230 errors at `1e-7` are `3.183231456205249e-12` hartree
and `4.801936626108727e-12` hartree/bohr. Every repeat passes the original
independent expanded FP64 CCSD replay limit `1e-10` and response gates `1e-8`.

## Default domain and lifetime checks

Uniform mixed warmup loses on small systems. Three paired E+F repeats cover
water24, methane34, ethane58, water92 and ethane144. Water92 CCSD increases
0.583431 -> 0.660647 seconds. Ethane144 CCSD falls 5.905223 -> 5.373957
seconds; complete E+F falls 54.676336 -> 54.116988 seconds (1.0230%).

Native CUDA DF defaults therefore request warmup only with at least 9 occupied
and 128 virtual orbitals. Explicit mixed/FP64 ablations remain available, and
existing dimension/provider/memory refusal retains strict execution. This is a
conservative qualification bracket on RTX 5090, not a universal GPU crossover
model. CPU, conventional CCSD and supplied-Hamiltonian solver defaults do not
change.

The final compiled auto binary is checked separately on water24, ethane144 and
ethane230. A medium-ethane comparison also exercises cold, same-process warm
and moved-geometry E+F endpoints with recycled response state. Each lifecycle
output passes its own physical residual and paired energy/force gates. The
displaced call moves one nucleus by `1e-4` bohr and updates both basis geometries.

Independent H2/water/LiH oracle records use pinned PySCF 2.14.0 energies and
two-step directional finite differences. They cover the retained switch and
explicit `1e-3`/`1e-5` cuts. The final library passes GPU threshold/resource/
nonfinite checks, auto-policy checks and the public energy/force finite-difference
test. Validation logs distinguish genuinely run checks from skipped fixtures.

## Retained files and reproduction

- `summary.json.xz`: recomputed medians, all-repeat errors and evaluation counts.
- `samples.json.xz`: all 58 cold endpoint records, exact commands, source/binary/
  input identities and actual semantic-work counters.
- `oracles.json.xz`: 36 independent energy/directional-force qualification records.
- `lifecycle.json.xz`: all six cold/warm/moved force outputs and artifact identities.
- `validation-logs.json.xz`: actual host, GPU, endpoint and public-test logs.
- Source patches and `water92.input.xz` / `ethane144.input.xz`: scientific source/input
  reconstruction. Other molecular inputs are already committed in the named
  benchmark results directories; their exact paths and hashes are in the samples.
- `lifecycle_gate.py.gz` / `retain_evidence.py.gz`: lifecycle and retention/gate
  scripts. The current `benchmarks/ccsd_mixed_thresholds.py` supports the same
  explicit comparator controls and additionally `--auto` for final-policy checks.
- `manifest.json`: retained-file SHA-256 and byte lengths.

XZ members preserve the original decoded bytes without dropping samples or
changing numerical values. Run `xz --decompress --keep *.xz` in this directory
to restore JSON, molecular inputs and source patches. The frozen retention helper
produces the original gzip/plain layout before this lossless storage compaction;
the manifest identifies the compact stored bytes. Helper scripts remain gzip.

Compile `benchmarks/df_ccsdt_force_endpoint.cpp` against the corresponding
native library, using ccache and the same native CMake configuration. Inside a
finite Slurm allocation, run `benchmarks/ccsd_mixed_thresholds.py` with the
recorded input, `--thresholds 1e-7 5e-7 --forces --repeats 3`. For the final
policy, also pass `--auto`. Preserve Slurm's device visibility and compare every
repeat's physical gate; do not replace complete endpoint timing with kernel
timing. Exported source snapshots use both `--source-commit` and `--source-patch`.

The [decision note](../../../.agents/notes/implemented/performance/2026-10-12-df-ccsd-auto-warmup-threshold.md)
records the rejected blanket default, work/iteration tradeoff and revisit criteria.
