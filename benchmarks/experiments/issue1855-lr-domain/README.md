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
