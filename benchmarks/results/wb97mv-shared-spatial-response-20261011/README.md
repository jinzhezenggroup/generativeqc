# Bounded independent-atom scalar force responses

Measured base: master `858fe52f1057bf77f30ad56f3bef9b6ecf00e353`, fetched at the
start of qualification, including #2236 and order-4--6 spatial reuse (#2244).
The candidate automatically shares one scalar L+1 Coulomb preparation across
all N-1 independent atoms for ordinary full/LR orders 7--11. It uses existing
Hermite/radial/contraction algebra, not a separate scientific implementation.
There is no opt-in, molecule/basis/functional/device whitelist, approximation,
precision change, new global tensor, resident lease, or queue traversal.
Order 12, short-range and optional reachable/convolved derivative modes retain
their existing fallbacks; lower orders keep their specialized owners.

## Protocol

- 3/12-atom water proxies, **not OMol25-distribution samples**. Complete
  spherical def2-TZVPD H/O with diffuse and f shells is retained.
- Exact Direct WB97M-V/RKS FP64, grid 48x16x32 per atom, Becke3;
  energy/density/screening 1e-12/1e-10/1e-12; VV10 density cutoff 1e-8;
  4 GiB host/device force allowances.
- Cold means a fresh owner and independently initialized density. Compiler
  and artifact caches are preserved and both source artifacts are primed.
  Imports, CUDA context and Calculator construction precede the clock; prepare,
  SCF, physical force, synchronization and host publication are inside it.
  Teardown and serialization follow. First compiler/artifact use is retained
  separately, never substituted for throughput.
- One finite Slurm job/device runs ABBA, two samples/arm at each size, followed
  by two fresh full-Fock GPU4PySCF 12-atom endpoints. Native never borrows the
  oracle density. The reference reports CUDA LibXC; this is not a stock-default
  DF comparison. All real-GPU work preserves scheduler-assigned visibility.
- Both producers use Release sm_120, fast compile disabled, verified ccache
  CXX/CUDA launchers and checkout-root normalization. The unrelated native CLI
  is disabled in both builds because node1 lacks its JSON SDK dependency.

## Complete results

Clean matched medians in seconds:

| Atoms | Phase | Baseline | Candidate | Change |
| --- | --- | ---: | ---: | ---: |
| 3 | Complete E+F | 11.955799 | 12.061497 | +0.88% |
| 3 | Physical force | 3.386311 | 3.428410 | +1.24% |
| 12 | Complete E+F | 189.466910 | 184.278075 | -2.74% |
| 12 | Physical force | 32.991263 | 27.745079 | -15.90% |
| 12 | SCF/publication | 155.542374 | 155.535303 | approximately unchanged |

Native arms retain identical available semantic work/traffic and iteration/Fock
counts: 15/15 at 3 atoms and 21/21 at 12 atoms. Maximum primary physical force
error is 5.014128e-11. Executed endpoint primitive recurrence counts are not
exposed; null fields remain null, and capacity bounds are not reinterpreted as
executed work.

Four additional small-shape ABBA cycles retain all eight samples/arm:
complete 11.950039 -> 12.078239 s (+1.07%), force 3.374666 -> 3.462618 s (+2.61%).
Preparation accounts for most of the observed difference (median 2.073808 ->
2.166398 s). The default accepts this explicit small-shape startup/preparation
tradeoff for the high-order force improvement; this is **not** a universal
speedup or non-regression claim. No sample or selector hides the unfavorable
small-shape result.

Fresh same-job/device full-Fock GPU4PySCF medians are complete 183.499979 s and
force 11.032725 s. Native complete time is near parity (+0.42%), but native force
is still approximately 2.51x slower. Reference uses 45 iterations / 46 Focks,
native 21 / 21; this is not an equal-per-Fock throughput study.

No complete 96-atom cold E+F or independent 96-atom force acceptance is claimed.
Master advanced during measurement to `98263e06ca13aa88cee504f1f105f5cf3ef6d92a`;
59 additional host/source numerical compatibility cases pass there. All GPU
timing/qualification identities in this receipt still refer to frozen `858fe52f`
plus the retained patch, not that later snapshot.

## Gates and residuals

- 141 targeted/expanded host tests; 515 compiler modules, zero dependency errors;
  351 CUDA ownership files; formatting and native-complexity checks pass.
- 4860 independent atomic-coordinate checks across orders 7--11, three Cartesian
  orientations, signed two-primitive contractions, four atom layouts, full/LR,
  omega 0/0.37/10000 and exponent scales 1e-3/1/1e3. Normalized AD/displaced-value
  maxima are 4.52281e-13 / 3.7247e-8, within predeclared 3e-11 / 3e-7 gates.
  Four-point displacement uses step 1e-3/sqrt(scale), avoiding diffuse-fixture
  cancellation without loosening acceptance.
- Exact per-live-AO-quartet work: four bra preparations, sixteen ket/scalar
  Coulomb preparations and sixteen contractions per independent atom.
  Single-atom quartets perform no derivative preparation. These are exact
  fixture counts, not an endpoint primitive-work census.
- Final-source GPU probes cover all five orders, four atom layouts, full/LR,
  64 sampled Cartesian components/case and the independently checked final atom.
  Memcheck/initcheck/synccheck/racecheck report zero errors/hazards.
- Six Release native suites, separate SR/LR and mixed s/p/d/f gates, eight
  independent Libcint J/K cases and eight independent FCI physical-force cases
  pass. The initial FCI attempt skipped for missing threadpoolctl; retained logs
  document that, and an isolated 3.7.0 overlay replay requires eight passes.

The intrusive residual profile is excluded from throughput. Ordinary LR force
and full f-containing force kernels take 10.274591 / 8.222631 GPU seconds.
The first clean matched integral-derivative region is 26.417960 -> 20.920601 s;
geometry/pair drain stays about 4.41 s. Prioritize bounded cross-component/CTA
reuse without repeating shell enumeration. Preparation identity verification
also deserves its own lifetime-safe optimization: cProfile records 1.214 s in
878 file-hash calls and 1.089 s in stationary artifact resolution.

Compact state is not a compiled-frame claim. Restricted FullSources angular
diagnostic symbols retain 255 registers/thread and 4108 shared bytes/CTA;
reported private stacks grow from 26824--65208 to 33368--65688 bytes/thread.
These static symbols are not endpoint peak allocations; device-function REG=0
means unavailable metadata, not zero use.

## Evidence and reproduction

`evidence.json.xz` contains all primary, supplementary, priming and excluded
initial-cohort physical records, work/traffic, independent gates, source/library
identities, cache statistics, diagnostic profile identity, raw hashes and local
archive receipt. Candidate library SHA-256:
`d1a3e6079f278bf3678c47291c55853a98b1ac28225ac981150d2147d6ce4e54`.
`source.patch.gz` reconstructs the measured dirty source from the exact base.
`cold.py.gz` retains the byte-matched endpoint runner, not a production selector.
Standard Python `lzma`/`gzip` decode these lossless records; compression preserves
the complete receipt and runner bytes while respecting the aggregate evidence cap.

Raw source archives, scripts, compiler commands, logs, Nsight report/SQLite and
sanitizer outputs remain under
`/data/jzzeng/wb97m-shared-response-latest-20261011/` on node1, with an ignored
worktree mirror under `.artifacts/shared-response-latest-20261011/`.
`raw-qualification.tar.xz` is checksum-bound in the compact receipt. The archive
and raw-result inventory exclude receipt, matched-summary, receipt-builder stdout
and raw-retention output, preventing checksum cycles or hashing active stdout.
No Release, release tag or external raw backup is published.

The original candidate-only-primed cohort is invalid for throughput. Baseline-3
first force was 18.7868 s versus 3.37141 s on repetition. Its own job was stopped
after baseline-12 completed; all five complete captures remain retained. The
entire replacement ABBA runs after both artifacts are primed, rather than
dropping only the slow baseline point.

Reproduce each source arm with its explicit library path, `PYTHONPATH` and
`GENERATIVEQC_BENCHMARK_SOURCE` identity. From that checkout, generate an
independent full-Fock reference, then execute the native endpoint:

```bash
mkdir -p .artifacts
gzip -dc benchmarks/results/wb97mv-shared-spatial-response-20261011/cold.py.gz \
  > .artifacts/wb97mv-cold-replay.py
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --time=00:10:00 python \
  .artifacts/wb97mv-cold-replay.py \
  reference --atoms 12 --reference-full-fock --output /tmp/wb97mv-reference.json
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --time=00:10:00 python \
  .artifacts/wb97mv-cold-replay.py \
  native --atoms 12 --reference /tmp/wb97mv-reference.json \
  --output /tmp/wb97mv-native.json
```

Prime both source arms separately and retain those calls. Repeat with fresh
processes in ABBA order on one finite allocation; do not use `--profile` in
throughput arms. Historical input scripts in the archive retain the exact
build/environment and complete-cohort commands.
