# LR retained-domain acceptance adapter

This Linux/ELF **test-only** adapter isolates #1855's packaged `omega=0.3`
long-range stationary derivative schedule without changing the main library,
the full-range derivative schedule, the primary owner's retained prefix, SCF,
density, precision, screening predicates, or physical source coefficients.
It is not part of any installed library or default build.

Setting `GENERATIVEQC_BOUNDED_SCHWARZ_SCHEDULE=0` is **not** this ablation: that
also changes primary preparation and full-range scheduling. Instead preload
`interpose.so` and select `GENERATIVEQC_ACCEPTANCE_LR_DOMAIN=indexed` or
`triangular`. Only the LR launch receives an empty optional domain in the latter
arm. The original primary owner and its charged allocations are unchanged.
Unsupported angular-partition and non-packaged omega routes are not qualified
by this adapter.

## Clean timing versus intrusive work receipts

Without `GENERATIVEQC_ACCEPTANCE_LR_RECEIPTS`, both arms forward to the same
unmodified production launcher. Use alternating arm order, identical explicit
inputs, and complete returned energy/force cold, warm, moved and moved-warm
populations. Count all iterations, retries and failures without normalizing time
by iterations. Each source-matched arm must pass independent references.

`endpoints.py` validates the complete independent reference population before
native execution: one cold/moved row plus five uniquely identified replays for
each exact geometry, with successful finite, shape-correct E/F results. Every
native row is paired with all six same-geometry oracle rows, retaining their
row indices. Native warm replays freeze the corresponding post-cold/post-move
density, matching the declared reference protocol. Failed native item statuses,
SCF work and elapsed time are journaled before the unchanged numerical gate
rejects them; nonfinite failure diagnostics are explicit strings in strict JSON.
Prior runner receipts remain historical evidence under their original runner
identity. Do not relabel them as passing these population or fixed-density gates.

With that variable set to a writable JSONL path, the LR launch instantiates the
**existing** bounded kernel and its **existing** post-screen shell-class
profiler from `direct_bounded_fallback.cu`. The instrumentation uses separate
temporary device scratch, synchronizes and downloads observations. These runs
are numerical/work/resource evidence, **not clean endpoint timings**. Entries
record actual drained cursor claims, launch grid, indexed/triangle products,
and admitted shell/tile/AO/primitive-quartet counts by shell class. Owner entries
check that neither retained allocations nor the borrowed prefix changed. Scratch
bytes are reported separately; no whole-process memory-peak claim follows.
The intrusive kernel specializes its existing radial-operator template to Long;
that is the only admitted observer input. Its compiled register/stack footprint
is not evidence about the original dynamic-range production kernel.

For clean runs the same source can instead be compiled with a C++ compiler
(`g++ -x c++ -shared -fPIC -fvisibility=hidden ... -lcudart -ldl`). That thin
adapter forwards all kernels and fails closed if intrusive receipts are requested.
It avoids needlessly blocking clean qualification on the large diagnostic CUDA
translation unit. Record the clean and intrusive adapter hashes separately.

Compile against the **same frozen source and generated headers** as the main
library. `-fvisibility=hidden` is essential: only the two explicitly exported
interposer functions may override production symbols. `prepare_kernel.cmake`
extracts the exact native template prefix into an ignored diagnostic header,
excluding unrelated public launchers and their many angular/materialized
instantiations. It does not edit or reimplement recurrence or screening science;
a changed extraction boundary fails closed. Retain the source/header hashes.
The original library's symbols must remain dynamically
interposable; missing symbols or an undrained domain fail closed.

An illustrative build (verify the compiler-cache executable first, and compile
objects separately so `ccache` actually caches compilation):

```sh
cmake -DSOURCE_ROOT="$PWD" -DOUTPUT="$PWD/.artifacts/lr-domain-kernel.cuh" \
  -P benchmarks/experiments/issue1855-lr-domain/prepare_kernel.cmake
ccache --version
ccache --show-stats
ccache nvcc -c -O3 -std=c++20 -arch=sm_120 --expt-relaxed-constexpr \
  -Xcompiler=-fPIC,-fvisibility=hidden \
  -Iinclude -Isrc -Ibuild/cuda-release-sm120/generated -I.artifacts \
  benchmarks/experiments/issue1855-lr-domain/interpose.cu \
  -o .artifacts/interpose.o
nvcc --shared .artifacts/interpose.o -Xlinker=--no-as-needed \
  -Lbuild/cuda-release-sm120 -lgenerativeqc -ldl \
  -Xlinker=-rpath -Xlinker="$PWD/build/cuda-release-sm120" \
  -o .artifacts/interpose.so
ccache --show-stats
readelf -d .artifacts/interpose.so
```

Verify that the dynamic section contains `NEEDED libgenerativeqc.so.0`.
`RTLD_NEXT` must find the original implementation even when Python loads the
library with `RTLD_LOCAL`. Without a forced dependency, GNU `--as-needed`
can discard it because the forwarding adapter has no unresolved direct call
to that implementation. With NVCC, **do not** restore `--as-needed` in the
same command: NVCC can move both linker options before the libraries and
silently discard the dependency again. Keep that failed build's receipts,
but do not treat a successful link alone as a qualified adapter.

For the clean C++ object, use `ccache g++ -c -x c++ -O3 -std=c++20 -fPIC
-fvisibility=hidden` with the same source/include directories and the toolkit's
include directory. Link with `g++ -shared`, toolkit `-lcudart`, `-ldl`, the same
library rpath, and `-Wl,--no-as-needed -lgenerativeqc -Wl,--as-needed` in that
order. The thin and CUDA adapters are separate binaries with separate hashes.

Run every native gate, sanitizer, profiler and endpoint through a finite,
compatible `srun`, preserving scheduler visibility. The native executable's
`--lr-domain-only` gate covers sparse/dense indexed claims and exact/one-byte-
short/no-prefix budgets without running unrelated providers. Its
`--range-response-only` gate retains the independent displaced through-f
RKS/UKS range-ERI tests. Acceptance additionally needs the complete endpoint and
intrusive-ledger population; compiling this adapter alone is not a PASS.

## Acceptance completed on 2026-10-10

`acceptance.json` retains the validated source/binary/runner/reference identities,
complete matched-pair timing distributions, allocation receipts, numerical gates
and actual post-screen class work. Production remains frozen at
`15bc697009d191a88120607e0e50a15561d35d61`; its library SHA256 is
`7fc478aa78b15a1ee29fb8659c6121c8feb01f4f61aa917e59207d55af7c37c0`.
The Release build uses sm_120 code generation but reports the portable
`generic_cuda` policy; optional shell/stationary AOT is off. It is not an
installed-production/no-runtime-compilation or mixed-precision qualification.

The qualified runner is `38d67a8caebe33f9dce2b9b6d1e8ac9fe7d4b534`, SHA256
`89b8d53bdf525ec8eb4342ac6a2e40645df4ed8b5c38e86638beb11fa5fe9283`.
The later retained-output-path guard changes only argument validation, outside
native timers; all four actual destinations are ignored scratch directories.
The current strict runner and bounded-schedule host suites pass 25 +3 tests.

Four main/node1 RTX 5090 Slurm allocations, each bounded to 75 minutes, complete
with `srun=0`, controller `COMPLETED` and `ExitCode=0:0`:

- **7102–7104:** three alternating-order pairs, 72 clean complete returned E/F
  calls, each cold plus five fixed post-cold replays and moved plus five fixed
  post-move replays per arm; 432 same-geometry independent gates.
- **7105:** separate 24-call intrusive two-arm population and 144 independent
  gates. Its seconds are never used to claim speedup.

Every endpoint uses exact water-12 / 232 spherical def2-TZVPD AOs, moving original
Becke-3 48×16×32 quadrature, FP64, Direct/no DF and matched VV10 density policy.
The immutable independent reference contains six successful endpoints for each
exact geometry. E/F gates remain **1e-8 Eh / 1e-7 Eh/bohr**. Maximum errors across
all 576 pairings are **5.9117155615240335e-12 Eh** and
**1.099751401056892e-10 Eh/bohr**. Every clean and observed arm records 21 SCF
iterations/Fock builds cold, 14 moved and one per warm replay. No iteration
normalization, failed-run substitution or cross-GPU pooled comparison is used.
This is the water recipe, not OMol validation.

### Complete clean timing

Cells are indexed / triangular median seconds on the **same physical GPU** of
each allocation. The JSON retains every sample and UUID; GPUs differ across rows.

| Pair / Slurm | Cold | Warm (5) | Moved | Moved-warm (5) |
| --- | --- | --- | --- | --- |
| 0 / 7102 | 583.998600 / 583.865209 | 62.396739 / 63.301433 | 402.127066 / 402.393042 | 62.358648 / 63.312096 |
| 1 / 7103 | 586.626814 / 588.464072 | 62.941002 / 63.797725 | 403.950247 / 404.757535 | 62.889153 / 63.849184 |
| 2 / 7104 | 583.567433 / 583.532113 | 62.251430 / 63.418036 | 401.402262 / 402.095322 | 62.246699 / 63.051666 |

Indexed LR shows a small, consistent **1.28–1.84%** warm/moved-warm endpoint
benefit in these three pairs. Cold is effectively flat/mixed (-0.023–0.312%);
moved differs by only 0.066–0.199%. This is not a universal crossover, a large
endpoint-gap closure or evidence about different device/shape/screening domains.
Retain the implementation and bounded triangular fallback; change no default.

### Actual work and ownership

Both observed arms launch 1 LR kernel per endpoint, 1,360 blocks ×256 threads.
Indexed consumes **8,503 products ×16 pages +1,360 terminating claims =137,408**
actual cursor claims; triangular consumes **10,731 +1,360 =12,091**. Fewer indexed
geometric products do **not** mean fewer claims, blocks or surviving integrals.
Both arms have exactly equal post-screen counts, including every one of the 55
class entries, for all six endpoints at each geometry:

| Geometry | Shell quartets | Tiles | AO quartets | Primitive quartets |
| --- | ---: | ---: | ---: | ---: |
| Original | 7,497,817 | 7,911,920 | 335,167,456 | 767,930,390 |
| Moved | 7,497,904 | 7,912,007 | 335,169,706 | 767,939,573 |

Each of the 24 owner checks preserves 19 owner / 28 shared allocations and
10,218,221 / 4,363,861 charged device bytes, with the primary prefix retained
and **zero extra retained index allocations**. Temporary observation scratch is
1,760 bytes, reported separately. These inventories are not whole-process peaks;
the observer specialization supplies no production register/stack evidence.

Native sparse/dense/empty and exact/one-byte-short/no-prefix cases pass, as do
the independent displaced through-f RKS/UKS range gates under both thin and
observer arms. memcheck/initcheck/synccheck report zero errors; racecheck reports
zero hazards/errors/warnings. Slurm 7061 has a completed controller receipt;
7072 retains successful `srun` and pre-exit receipts but its completed controller
entry expired. Only the four final endpoint jobs supply the complete acceptance
population. ELF inspection confirms that each adapter shares exactly the two
intended scientific exports with the main library.

### Reproduce and retain evidence

Compile the two adapters against the frozen source/generated headers using the
object/link procedure above. Use a fresh ignored output directory for each run.
Inside a finite `srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1
--cpus-per-task=8 --mem=64G --time=01:15:00`, preserve assigned visibility and set
`GENERATIVEQC_LIBRARY`, `GENERATIVEQC_PROFILE=off`,
`GENERATIVEQC_BOUNDED_SCHWARZ_SCHEDULE=1`, toolkit/library paths and all three
BLAS/OpenMP thread limits to 8. Preload the thin adapter for each of three jobs:

```sh
python benchmarks/experiments/issue1855-lr-domain/endpoints.py \
  --basis-file "$CAMPAIGN/def2-tzvpd-ho.canonical.json" \
  --reference "$CAMPAIGN/reference.json" \
  --output "$CAMPAIGN/fixed-pair-$REPEAT/clean.json" \
  --repeats 1 --repeat-offset "$REPEAT"
```

Run `REPEAT=0,1,2`; each job contains both arms on one assigned GPU. Preload the
CUDA observer instead for a separate job with `--observe --repeats 1`, offset 0
and a fresh `fixed-observed/observed.json` destination. Retain source/DSO hashes,
runner copies, GPU UUID/visibility, native diagnostics, all records, JSONL work,
exit codes and completed controller receipts. Raw artifacts, the strict
`finalize.py` validator and all unsuccessful pilots remain under ignored
`.artifacts/issue1855-acceptance-20261010/`, mirrored at
`n1:/data/jzzeng/qc-issue-1856-acceptance-20261010/` with the same relative path.

The pre-fix runner populations are **not qualified**, even where their original
runner serialized `PASS`: post-cold/post-move densities were not frozen as the
declared protocol requires. Preserve those original runner identities and
nonqualification receipts. The corrected population restarts from scratch;
no previous timing row is spliced into it. All scientific thresholds stay fixed.
