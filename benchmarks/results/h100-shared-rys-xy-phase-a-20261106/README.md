# Shared Rys x/y phase A: fixture capability and H100 correctness

This stage admits a default-false **compiler fixture capability**, not production
selection. `CudaScheduleIR.fock_shared_rys_xy` shares unweighted x/y Rys3 tables
for derivative-free restricted DPPP/PPPP Fock at the incumbent 192/96-thread
mapping. `KernelSelection` rejects it. No SCF selector, production manifest,
registry or default is enabled. The private UHF path remains the incumbent.

The measured source is `b56c2c935aa63f68049a116e420aa5a5dcb529e8`, tree
`0912d6a8f9ff85bdd5d3e5307978935478363c1b`, based on `b681a1fc3`.
The implementation was initially cherry-picked as `3a4b00fba`. Subsequent
commit `c394f7021` adds a legacy-`ShellClassSpec` admission guard and a rejection
test for general shell signatures; the branch also integrates later main.
These source-admission changes are outside the measured `b56c` device run.
The historical execution is not a device qualification of the amended branch
or its later whole-tree revision.
The [decision note](../../../.agents/notes/implemented/performance/2026-11-06-fixture-only-shared-rys-xy.md)
records ownership, fixed fallbacks and the remaining actual-task gate.

## Completed synthetic device qualification

The root-owned H100 run completed on 2026-11-06 with GPU UUID
`GPU-dbe42bb5-8137-d2c0-10ae-44f06b6d407b`, H100 80GB HBM3, SM90, MIG disabled,
driver 595.58.03. The executable SHA256 is
`c182f0aba2e0a1f06800df307850364ea255d63943bd4840b2f8c9f17df52c46`.
The immutable v2 bundle manifest SHA256 is
`e57a64f960777a965e97df25f2344269702f3f77c298c8d8af8cfcd2019d27ac`.
Exact CUDA 12.9, GCC12, sccache 0.16.0 and sanitizer binaries/support-file
bindings, commands and stage outcomes remain in logical `qualification.json` ([stored bytes](qualification.json.xz)).

One executable contains private/shared variants and six independent Libcint
fixtures: DPPP/PPPP, each Cartesian, coincident and reversed-pair geometry.
Every full Fock matrix entry is checked at unchanged atol=2e-11, rtol=2e-10,
including candidate/private parity. Each launch traverses seven tasks with
four signed primitive quartets per task and full/zero/partial/one/two masks.
The 27-live partial mask screens all six producer component lanes while taking
the shared path. Ordinary execution and captured graphs each run twice, resetting
output and queue head inside each captured body.

Ordinary/graph, memcheck, racecheck and synccheck all pass: 48 matrix reports per
invocation, 192 retained full-precision maximum-error observations in total.
These reports summarize executed per-entry checks; the offline reader does not
recompute matrices. All observed maxima are below the absolute tolerance alone.
Memcheck and synccheck report zero errors; racecheck reports zero hazards,
errors and warnings. These are supported checks on this synthetic fixture,
not universal race-freedom or actual HF96 qualification.

| Persistent RHF kernel | Registers/thread, private → shared | Runtime local bytes/thread | Runtime static shared bytes, private → shared | Extra shared bytes |
| --- | ---: | ---: | ---: | ---: |
| DPPP, 192 threads | 166 → 166 | 96 → 96 | 488 → 1064 | 576 |
| PPPP, 96 threads | 166 → 166 | 72 → 72 | 488 → 920 | 432 |

PTXAS reports zero spill loads/stores for all four reviewed persistent RHF
symbols. Runtime/PTXAS and cuobjdump resource rows remain separately labelled;
their absolute shared-byte reports differ, while each reports the same
candidate-minus-private +576/+432-byte delta. No occupancy or throughput gain
is inferred from these resource reports. Retained compiler warnings are unused
generated constants, counted with the original compiler-stream hash.

## Original failure remains failed

`failed-v1.json` is the unchanged first attempt. Its assumed
`/opt/generativeqc-reuse/cuda-12.9/bin/compute-sanitizer` did not exist, so it
failed before compilation or device execution. V2 binds an existing sanitizer
and support files; code, generated TU, fixture receipt, headers and source patch
remain byte-identical. The original failure and both bundle identities are
retained. V2 success does not turn v1 into a successful numerical run.

## Offline inspection and reproduction

```sh
python benchmarks/results/h100-shared-rys-xy-phase-a-20261106/replay.py
```

The standard-library reader checks storage hashes, measured source/bundle and
fixture bindings, all 192 report identities/margins, supported sanitizer outcomes,
resource deltas and the preserved failure. It does no native import, solver,
reference generation, compilation, GPU work, network access or subprocess.
`cohort.json` binds retained records. The archived `qualifier.py.xz` preserves
the exact execution-driver source; historical paths are not portable defaults.
Full generated TU, executable/object and raw compiler/sanitizer/cache streams
remain in ignored task storage. Material observations and original stream
hashes are retained, rather than copying those build products into Git.

The checked-in fixture generator is the reproducible scientific owner. To
prepare a new fixture in an existing environment, separately from device work:

```sh
PYTHONPATH=python:. uv run --offline --no-env-file --no-project --python 3.12 \
  --with numpy==2.2.6 --with pyscf==2.14.0 \
  python -m tools.generativeqc_validation.shared_rys_xy_fixture \
  --output .artifacts/shared-rys-xy-fixture
```

Any new H100 execution must use a fresh output directory, verified matching
source/input/tool bindings and compiler cache, then the recorded compile/link,
resource, ordinary/graph and three sanitizer commands with their finite limits.
This PR performs none of those runs again. The earlier 143-test arithmetic
suite and H100 qualification are reused as source-bound evidence; integration
on the newer base uses only the focused parser/identity/rejection checks.

Actual HF96 task/density capture and actual-task performance remain separate
phase-B work. This fixture supplies no HF96 speedup, molecular E/F endpoint
qualification or coherent 30-case win. The original full30 goal and all of its
independent energy/force, paired cold and warm-regression gates remain unmet.

Selected JSON receipts are stored as lossless `.json.xz` files. `cohort.json`
retains each original logical filename, decoded length and SHA256, separately
from the stored path/length/hash. The bounded reader decodes them before the
unchanged scientific report audit; original failure bytes remain unchanged.
