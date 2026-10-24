# Experiment: selective canonical ERI residency from master

Status: proposed (diagnostic profiler and offline selector implemented; production partial residency not implemented)
Date: 2026-10-11
Base: master `637466c624c4c4c4dd4eb7c511a2c88675744325`

Historical diagnostic stage: the subsequent
[partial production implementation](../implemented/performance/2026-10-11-partial-canonical-eri-residency.md)
supersedes the implementation boundary below, without promoting these isolated
bucket predictions to complete-endpoint evidence.

## Boundary and diagnosis

PR #2243 is still open at the initial master check. Its screened global-hybrid
`DirectJkSourceLease` is not on this base. Master instead owns phase-local RHF
values in `src/scf/cuda/rhf_resident_values.*`. Its common provider accepts
resident values only for unscreened canonical sources, retains the whole source
or refuses, and replays in the immutable canonical bucket traversal order.

The existing buckets group canonical **AO pairs by angular sum**. They are not
shell-quartet primitive-cost classes. The new diagnostic uses that existing ABI
without generating a second integral evaluator. It must not be presented as a
qualification of screened shell-quartet storage, PBE0 source reuse, or master
complete endpoint acceleration. It leaves all production schedules untouched.

## Diagnostic and selection model

`benchmarks/direct_eri_bucket_probe.cpp` owns one geometry/provider and profiles
each nonempty angular bucket under an explicit optional value allowance. All
values are FP64, full-range and unscreened. Direct and replay use the same signed
symmetric canonical density and the same symmetry scatter. Clean device timings
include output reset, but not census instrumentation, host downloads or fences.
Standalone build timings include the source finite audit and publication fence;
allocation and release are reported separately and charged once in the model.
An untimed observer verifies direct/build/replay candidate counts and recurrence
counts (N/N, N/N, N/0). Direct/replay parity is a diagnostic check, **not an
independent molecular oracle**. Every repeat contributes to its maximum error.

`benchmarks/selective_eri_cache.py` records actual loaded native/probe binary
hashes, input/geometry/basis identity, GPU UUID, driver/runtime versions, raw
timing samples, failed progress and capacity-refused classes. Output goes to an
ignored raw-artifact location, never directly into reviewed results.

For an explicitly supplied future-action scenario, selection models

```text
gain[b] = actions * (median(direct[b]) - median(replay[b]))
          - median(build[b]) - allocation[b] - release[b]
```

The budget is an optional value allowance **after** complete mandatory numeric
admission, source metadata/scratch and live-device reserve. Shared incremental
setup is charged once with `--fixed-setup-seconds`; it is not omitted merely
because the profiler has already prepared an owner. Caller-supplied action counts
are scenarios, not actual iteration predictions. SCF iteration limits are not
eligible substitutes for physical Fock counts.

A deterministic Pareto-frontier search maximizes the additive model over whole
buckets, with a finite frontier bound. Exceeding that bound retains Direct and
reports refusal. Sorting by benefit/byte is not an exact knapsack optimizer.
The additive optimum is also not an endpoint optimum: simultaneous residency
can change memory contention, and per-bucket isolated times cannot price every
interaction. Selection receipts explicitly remain non-production/non-qualified.

## Reproduction

Verify and reuse the compiler cache before compilation. Link the probe against
the same master CUDA library that the metadata source will load; the Python
wrapper rejects different loaded provider/source library paths. Example, with
an existing independently identified master native build and CUDA toolkit:

```bash
mkdir -p .artifacts
ccache --version
ccache g++ -std=c++20 -O2 -fPIC -DGENERATIVEQC_HAS_CUDA=1 \
  -Iinclude -Isrc -I"$CUDA_HOME/include" \
  -c benchmarks/direct_eri_bucket_probe.cpp -o .artifacts/bucket-probe.o
g++ -shared .artifacts/bucket-probe.o "$NATIVE_LIBRARY" \
  -L"$CUDA_HOME/lib64" -lcudart -ldl \
  -Wl,-rpath,"$(dirname "$NATIVE_LIBRARY")" -Wl,-rpath,"$CUDA_HOME/lib64" \
  -o .artifacts/bucket-probe.so
export GENERATIVEQC_LIBRARY="$NATIVE_LIBRARY"
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:10:00 python -m benchmarks.selective_eri_cache profile \
  --input "$BASIS_INPUT" --probe .artifacts/bucket-probe.so \
  --output .artifacts/buckets.json --repeats 5
python -m benchmarks.selective_eri_cache select \
  --input .artifacts/buckets.json --output .artifacts/selection.json \
  --cache-budget-bytes 1073741824 --expected-actions 12
```

Do not override Slurm visibility, schedule the maintenance node, or execute the
probe outside a finite GPU allocation. There is no measured GPU or endpoint
speedup attached to this initial tooling change.

## Initial validation limits

The new probe compiles as a host C++20 object against CUDA 12.9.1 headers with
`-Wall -Wextra -Werror`, through verified ccache 4.5.1. This is not a linked native
product rebuild or a device-execution qualification. Ruff, clang-format and the
focused host gates pass, including exhaustive small knapsack comparisons,
non-profitable/zero-budget/search-bound fallback, per-repeat parity gates,
source/work identity refusals and stdlib-only CLI execution.

The local Slurm inventory reports the only visible main GPU node as drained.
Finite `srun --immediate=5 --time=00:01:00` preflight job 12266 cannot allocate
resources and exits 1. No GPU command executes; no device timing, sanitizer or
complete endpoint result is attributed to this change. Other nodes must not be
assumed available, and maintenance or device-visibility controls must not be
bypassed to obtain samples.

## Remote GPU validation on n1

The subsequent 2026-10-11 validation uses the same frozen master base
`637466c624c4c4c4dd4eb7c511a2c88675744325`, without stacking #2243. The remote
nodes expose separate Slurm controllers: the local drained inventory is not a
cluster-wide availability result. SSH inspection found usable GPU resources on
n1, n2, n4 and n5; the actual validation runs exclusively on n1/node1. Finite
preflight job 7377 confirms RTX 5090 availability. No maintenance-node GPU work
or device-visibility override is used.

A fresh Release native build targets CUDA 12.9.1 / SM120 using GCC 13.3.0 and
verified ccache 4.5.1 for both C++ and CUDA compiler launchers, reusing
`/data/jzzeng/ccache`. The initial remote configuration failure requesting the
unavailable `/usr/bin/g++-11` is retained separately. The native library, bucket
probe and existing oracle probe link successfully, including `-Wl,-z,defs` for
the probes. CPU linear algebra uses the scalar fallback in this build; that is
not a qualified complete-endpoint performance configuration.

Every device execution runs through the n1 controller with
`srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1`, a finite 5–30
minute limit and Slurm's unmodified `CUDA_VISIBLE_DEVICES`. All ten validation
jobs exit zero:

| Job | Diagnostic | Value allowance | Receipt / acceptance |
| --- | --- | --- | --- |
| 7380 | signed synthetic s/p/d/f, five repeats | 1 GiB | 28 measured buckets |
| 7381 | same source, zero-budget admission | 0 | all 28 capacity-refused; empty plans |
| 7382 | s/p/d/f, `--displacement 0.05`, five repeats | 1 GiB | 28 measured; geometry identity changes, basis identity does not |
| 7383 | existing independent resident J/K oracle tests | test-specific | 22 passed, 30 unrelated tests deselected |
| 7384 | s/p/d/f, two repeats under full memcheck | 1 GiB | 28 measured; zero errors and zero leaked bytes/allocations |
| 7385 | 12-atom water / def2-TZVPD, three repeats | 8 GiB | 28 measured buckets |
| 7386 | 24-atom water / def2-TZVPD, three repeats | 8 GiB | 27 measured, one capacity-refused; completed in 23m37s within its 30-minute limit |
| 7387 | 96-atom water / def2-SVP, three repeats | 8 GiB | three measured, 12 capacity-refused, 13 empty |
| 7389 | s/p/d/f low-budget admission, three repeats | 8 KiB | 19 measured, nine capacity-refused |
| 7391 | 24-atom water / def2-TZVPD low-budget admission, three repeats | 1 GiB | 12 measured, 16 capacity-refused |

All nine profiler receipts inventory all 28 buckets. Measured buckets pass the
native `(N,N)/(N,N)/(N,0)` direct/build/replay census and direct/replay parity
gate. The maximum scaled error across these runs is
`2.99759642896418e-14`, below `2e-11`. The 22 independent PySCF/libcint tests
qualify the existing master resident provider's signed J/K actions, including
spherical/Cartesian, restricted/unrestricted, through-f and allocation-refused
cases; they do **not** qualify a new production partial-cache schedule. Memcheck
timings are instrumented diagnostics, not performance evidence.

The complete unscreened canonical value inventories are 4,328,718,848 bytes for
12-atom TZVPD, 68,988,699,648 bytes for 24-atom TZVPD, and 410,625,921,600 bytes
for 96-atom SVP. These are mathematical FP64 inventories, not measured endpoint
peak VRAM or screened-source estimates. With the 12-action scenario, the
24-atom/8-GiB receipt selects 11 buckets totaling 8,558,126,688 bytes, while the
96-atom/8-GiB receipt selects three totaling 7,648,726,848 bytes. The
24-atom/1-GiB run selects four totaling 1,069,541,280 bytes, and the 8-KiB
synthetic run selects 13 totaling 7,872 bytes. An additional offline selection
from the complete 12-atom receipt selects 12 buckets totaling 1,068,437,360
bytes under a 1-GiB allowance.

These are exact selections for the **measured additive model**, not integrated
partial-cache execution. All 27 original 1/4/12-action plans reproduce exactly
from the retained receipts, including receipt SHA256 checks. Action counts are
scenarios rather than observed SCF usage. Shared setup is supplied as zero for
these scenarios because no additional complete-endpoint admission cost has
been measured. No endpoint speedup, Cold/Warm E+F acceptance, force parity or
default-enable conclusion follows from these bucket timings. Production routing
remains unchanged, and plans still report `production_admission=not-qualified`
and `endpoint_performance=not-measured`.

Raw receipts, logs, binary/input hashes, device identities, build failures,
compiler-cache evidence and available terminal Slurm states are retained under
the ignored local directory
`.artifacts/selective-eri-cache/n1-master-637466c6/`, with remote originals at
`/data/jzzeng/selective-eri-cache-master-637466c6-20261011-3e50/` on n1. Slurm
accounting storage is disabled there, so the failed `sacct` lookup is also
retained; scheduler receipts, process exit statuses and per-job outcome files
provide the evidence instead. This remote validation supersedes the initial
local availability and unlinked-build limits above, but not the production or
complete-endpoint qualification limits.

## Next implementation gates

1. Extend the immutable provider layout with per-bucket value offsets and
   explicit uncached sentinels. Cache and Direct must partition the same source
   exactly once; uncached regions must not re-enter a whole-source evaluator.
2. Establish independently referenced compensated J/K parity for partial,
   complete, empty and allocation-refused subsets, including signed densities,
   f shells and multiple geometries. Preserve graph/source retirement ordering.
3. First qualify unscreened RHF on this base. Screened/global-hybrid adoption
   needs the corresponding owner/domain to exist on master; do not silently
   stack #2243 or change screening to create an easier comparison.
4. Measure 96-atom def2-SVP and 12/24-atom def2-TZVPD complete Cold E+F, Warm,
   Moved, Moved-warm and low-memory behavior. Separate unchanged-work timing from
   changed SCF Fock counts and account for final validation/handoff actions.
5. Retain all independent numerical gates, failed runs, binary/device identities
   and explicit bounded fallback before considering automatic/default routing.

XC AO/jet residency and Becke derivative fusion stay independent experiments.
An existing active-AO index lease does not imply that numeric jets persist across
SCF actions, and cold reuse gains do not prove warm phase reuse.
