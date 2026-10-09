# Shell-class CUDA code generation

## Goal

Move integral differentiation and shell specialization out of GPU execution
and into a deterministic compiler pipeline:

```text
Integral IR -> backend lowering -> CUDA target/schedule IR
            -> CUDA -> correctness/resource/timing gates
```

Generated kernels contain ordinary FP64 arithmetic and no runtime automatic
differentiation objects. The maintained surface is the generator, the
backend-independent oracle, the bounded schedule search space, and the
architecture manifest; generated production CUDA remains a build artifact.

## Current pipeline

The [task-parallel Direct Rys-K](direct_rys_tasks.md) variants use the same
mathematical compiler with lane-local quartet ownership. Prepared target
metadata defaults to nine qualified `sm_120` classes; other classes/profiles
retain the incumbent and explicit experiments remain independently selectable.

`python/generativeqc_compiler/integral/ir.py` is now strictly mathematical, while
`cuda_target.py` and `cuda_schedule.py` own NVIDIA execution policy:

- `IntegralIR` describes two-, three-, or four-shell operators, explicit
  centers, and direct-HF, raw-block, or external-weight consumers. The existing
  four-center force adapter differentiates centers 0, 1, and 2 and restores
  center 3 by exact translation invariance. The new
  [integral contracts](integral_ir.md) distinguish representable requests from
  executable CUDA support.
- `CudaScheduleIR` describes task/component ownership, block size, component tile,
  Coulomb-state placement, pair orientation/storage, and loop unrolling.
  The Fock-only `mixed_pair_products_fp64` option explicitly retains widened
  coefficient products inside a mixed evaluator while its accumulator remains
  FP32. Only the tuned `sm_120` `ddds` Fock schedule enables it; changing pair
  orientation must not silently narrow this class's existing arithmetic.
- `CudaKernelIR` combines the two with a `CudaTargetInfo` and validates target
  limits and component coverage before CUDA is
  emitted.

The backend contracts in `backend.py` cover source emission, compilation,
resource parsing, device probing, benchmark execution, and registry emission.
NVCC process-group handling and finite Slurm execution live in the CUDA adapter,
not in the mathematical IR. Production code imports the generic CUDA emitter
surface; the historical `dppp` pilot is isolated behind a compatibility
specialization module.

The subset/Wick recurrence is shared by ERI values and analytic gradients.
One plan can therefore emit RHF/UHF Fock and force kernels from the same
mathematical definition. The Fock lowering deliberately emits a value-only
Coulomb table, while the force lowering adds the one derivative order needed
for analytic nuclear gradients. The production manifest declares `fock` and `force` consumers per shell class;
mathematical support does not imply that a consumer is production-selected.

The mathematical catalog contains all 55 canonical s/p/d/f quartet classes,
including zero-order `ss` pairs. Current CUDA lowering supports pair orders
zero through six and provides four schedule families:

| Schedule | Mapping | Current status |
| --- | --- | --- |
| `packed_tasks` | one independent small-shell task per lane | emitted and benchmarked |
| `shell_task` | one warp/block cooperates on one shell task | emitted and benchmarked |
| `component_lanes` | one Cartesian component per lane | supported; selected by the target manifest |
| `tiled_components` | a bounded component slice per block | emitted and benchmarked |

Tiled lowering removes the former 1024-component limit. For example, `dddd`
has 1296 Cartesian quartets and defaults to a 64-component tile; `fddd` has
2160. State packing, Coulomb indices, Wick multiplicities, and f-component
axis tables widen automatically for these classes.

### Automation boundary

The mathematical IR represents one-electron and Coulomb operators, nuclear-coordinate
derivatives, exact translation invariants, and consumer-directed RHF/UHF
contractions separately. `KernelConsumer.FORCE` remains a compatibility input
at generator and manifest boundaries; it is normalized to an order-one
`DerivativeSpec` plus a `direct_force -> atomic_force` `ContractionSpec` before
backend lowering. The logical request contains all four derivative centers,
while the operator-declared translation invariant lets lowering evaluate three
and reconstruct the remaining center.

Rys root eligibility in the mathematical IR is derived from the required
value/derivative order rather than shell-name allowlists. Backend and production
promotion constraints remain separate. The `psps` and `ppss` production rows
now lower through the common Rys2 thread-task compiler for force and the common
packed-task compiler for Fock; the former embedded `low_order_force.py` CUDA
source and its CLI/production compatibility branches have been removed. Other
measured specialized fallbacks remain until their common candidates pass the
same correctness, resource, and endpoint gates.

The generic `cuda_emitter` surface now delegates to the backend-named
`cuda_lowering` implementation. `dppp_dispatch` is a narrow compatibility
adapter for the original DPPP specialization and resident-PPPS APIs, so adding
or tuning a generic shell no longer introduces a dependency on a historical
shell-class module.

Scalar thread-task force lowering follows the same structural boundary.  Its
component-scoped subset/Wick helpers are sized from `ShellClassSpec` and route
derivative slots through the attached `IntegralIR`, so f-shell classes such as
`fsss` and `fsps` can be emitted without adding a shell-name branch.  These
fallback candidates remain outside the production manifest until they pass the
independent resource and endpoint gates.

The component-lane fixed-root Fock worker follows the same capability boundary:
it is selected when the schedule covers all components, the decoder supports
the shell angular bounds, and the attached first-derivative IR requests a
three- or four-root Rys program.  Classes such as `dpss` and `ddss` therefore
share the backend path without being added to a promoted-name list.

The current CUDA backend still accepts only first nuclear derivatives of
four-center ERIs and preserves the existing force-vector ABI. Higher derivative
orders need a separate higher-order tensor layout and recovery implementation.
One-electron and density-fitting operators have explicit shell, center, and
bounded-block contracts. [One-electron S/T/V values](one_electron_codegen.md)
now have a separate Hermite DAG lowering and native candidate schedules;
the DF value provider has its own generated Rys lowering. Their derivative
and production-selection boundaries remain independent of the quartet ABI.

For large-AO direct-J/K failures, set `GENERATIVEQC_DIRECT_TILE_VALIDATION=validate`
to run an opt-in device validator immediately after shell-quartet compaction.
It checks partition counts and capacities, pair and shell IDs, Cartesian AO
ranges, and the first decoded AO quartet before any generated or handwritten
consumer runs. The diagnostic stops before Fock evaluation and prints one
first-writer-wins record containing the order, slot, pair/shell IDs,
`direct_nbf`, pair counts, decoded `(i,j,k,l)`, and partition metadata. The
record lives in a separate one-record CUDA allocation, so normal builds and
replays pay no queue or arena overhead when the variable is unset.

## Correctness model

The host oracle evaluates the same factored recurrence independently of CUDA.
Tests cover:

1. symbolic full-integral derivatives versus factored lowering;
2. generated recurrence values and all four center gradients;
3. finite differences, translation, and shell-permutation invariants;
4. every `dpds`/`ddps` component and representative `dddd`, `ffps`, `fddd`,
   `psss`, and `ssss` components;
5. real CUDA 12.9 compilation for joint Fock/force, shared/recomputed Coulomb,
   tiled d/f shells, zero-order pairs, and packed tasks.

Retained handwritten or independent reference implementations, where
they still exist, provide numerical and endpoint oracles; retired kernels
must not be described as active. The historical `psss` force kernel
combined all three weighted Cartesian outputs in one primitive traversal.
Any generated replacement must preserve its numerical quality and pass
the current production-gate contracts. See the
[CUDA ownership ledger](../maintainer/cuda_ownership.md).

## Experimental Rys backend

The backend-independent Rys/TRR/HRR state IR produces topologically ordered
recurrence instructions with the declared Cartesian order and translation
recovery. Task, subgroup, component-lane and tiled schedules are alternative
execution strategies. A correct generator is not automatically production
qualified: the [production shell manifest](../../python/generativeqc_compiler/integral/production_shell_classes.json)
declares the selected recurrence, schedule, consumers and target.

The PPPS/PSPS/PPSS signature grouping work, class-specific Rys variants,
rejected candidates and source-matched numerical/resource/latency evidence
are preserved in the [historical scheduling evidence](../../.agents/notes/implemented/performance/2026-10-09-shell-codegen-schedule-provenance.md).
Old environment toggles, profile numbers and timings must not replace the
current manifest/runtime when determining today's execution policy.

The previous case-study headings remain for stable fragment links.

### DSPS scalar Rys3 promotion

See the [historical scheduling evidence](../../.agents/notes/implemented/performance/2026-10-09-shell-codegen-schedule-provenance.md) for the full source-era case record.

### DPPS uniform component warps

See the [historical scheduling evidence](../../.agents/notes/implemented/performance/2026-10-09-shell-codegen-schedule-provenance.md) for the full source-era case record.

### PPPP uniform component warps

See the [historical scheduling evidence](../../.agents/notes/implemented/performance/2026-10-09-shell-codegen-schedule-provenance.md) for the full source-era case record.

### Batched DSPP and DPSS Rys3 promotion

See the [historical scheduling evidence](../../.agents/notes/implemented/performance/2026-10-09-shell-codegen-schedule-provenance.md) for the full source-era case record.

### Batched DPDP and DPDS Rys4 promotion

See the [historical scheduling evidence](../../.agents/notes/implemented/performance/2026-10-09-shell-codegen-schedule-provenance.md) for the full source-era case record.

### Batched DDPP, DDPS, and DDDS Rys4 promotion

See the [historical scheduling evidence](../../.agents/notes/implemented/performance/2026-10-09-shell-codegen-schedule-provenance.md) for the full source-era case record.

### 768-AO Fock follow-up for issue #52

See the [historical scheduling evidence](../../.agents/notes/implemented/performance/2026-10-09-shell-codegen-schedule-provenance.md) for the full source-era case record.

## Architecture autotuning

`python/generativeqc_compiler/integral/autotune.py` emits every CUDA-supported schedule variant
with unique symbols, compiles the translation units in parallel, links them
into one executable, and runs all variants in one GPU allocation. A candidate
is rejected for:

- CUDA compilation failure;
- spills, excessive registers, stack, or shared memory;
- Fock-value or force disagreement with the independent recompute oracle;
- failure to meet the configured timing threshold.

After the existing per-class candidate bound is applied, the tuner hashes the
exact unsuffixed generated CUDA only for packed candidates that differ solely
in algebra ordering. If two such schedules emit byte-identical CUDA, the
ordering choice is an exact no-op for that compiler revision and target, so only
one representative is sent to NVCC. Production baselines and canonical algebra
resource baselines are protected even when an equivalent peer exists. This is
exact source deduplication, not a profitability model: it cannot remove a
candidate merely because estimated FLOPs, liveness, or source size look worse.
Use `--no-execution-dedup` for exhaustive schedule-ID studies. The report
preserves skipped trial keys, their retained representative, and the
generated-source SHA-256 under `search.execution_deduplicated`.

Passing compiled variants are ranked by measured kernel time. The winner can be written
to a schema-v2, architecture-specific production manifest:

```bash
python -m generativeqc_compiler.integral.autotune \
  --nvcc /group/software/cuda-12.9.1/bin/nvcc \
  --architecture sm_120 \
  --shell-class dpds \
  --partition main --gres gpu:5090:1 \
  --output build/dpds-autotune.json \
  --manifest-output build/production-shells-tuned.json
```

Related hotspot classes can be tuned in one compiler/link/Slurm process by
repeating `--shell-class`, or by supplying a list file. The list-file form
accepts one class per line, comma-separated names, and `#` comments:

```bash
python -m generativeqc_compiler.integral.autotune \
  --nvcc /group/software/cuda-12.9.1/bin/nvcc \
  --architecture sm_120 \
  --shell-class-file benchmarks/issue52-hotspots.txt \
  --consumer fock \
  --allow-experimental-subgroup-winner \
  --partition main --gres gpu:5090:1 \
  --output build/issue52-fock-batch.json \
  --manifest-output build/production-shells-tuned.json \
  --require-all-winners
```

`--require-all-winners` makes manifest promotion all-or-nothing: if any
requested class fails compilation, resource, correctness, or timing gates, the
report is still written but the manifest output is left untouched. This keeps
synthetic batch results reviewable without accidentally installing a partial
production update.

When several classes are expected to benefit from the same mapping, restrict
the batch to that schedule family instead of compiling every legal family for
every class. Fock batches still compile the manifest-declared production
baseline beside the filtered proposals, so the synthetic gate cannot compare
a replacement only against the independent recompute oracle. For example,
this searches the component-lane variants for three Fock classes and
atomically promotes all three winners together:

```bash
python -m generativeqc_compiler.integral.autotune \
  --nvcc /group/software/cuda-12.9.1/bin/nvcc \
  --architecture sm_120 \
  --shell-class dpps --shell-class ddds --shell-class dppp \
  --consumer fock --schedule-kind component_lanes \
  --partition main --gres gpu:5090:1 \
  --output build/fock-component-lanes-batch.json \
  --manifest-output build/production-shells-tuned.json \
  --require-all-winners
```

Repeat `--schedule-kind` to combine a small number of related families. An
omitted filter retains the exhaustive legal search. Manifest output is written
through a temporary sibling and an atomic replacement, so interruption cannot
leave a partially written JSON file.

For Fock batches, the search also includes the exact production subgroup
baselines for ppps, pppp, dpps, dppp, dpdp, ddds, and dddp. This keeps a
candidate's synthetic speedup relative to the worker it would replace; a
candidate that only beats the independent recompute oracle is not sufficient.

Use `--consumer fock` to tune the value-only SCF worker with the same resource,
correctness, and timing gates. A Fock winner upgrades the manifest row to the
joint `fock`/`force` consumer set because both kernels share the canonical task
ABI.

The manifest records every code-shape decision rather than relying on emitter
defaults. The autotune driver queries the allocated device before invoking any
trial and exits on an architecture mismatch. Its artifact records real device
limits, driver/runtime versions, NVCC/PTXAS versions, generator ABI, and the
target-derived resource gates. Each candidate also records generated source
bytes, relocatable object bytes, and compile time. PTXAS resources are
translated into a per-kernel `resource_upper_bound` occupancy estimate; this
is explicitly an auditable upper bound rather than a replacement for a device
occupancy API. The report additionally records link time and the final linked
benchmark executable size, while the batch candidate screener emits the same
source/object/compile and linked-binary provenance.
When a production row declares a separate `fock_schedule` (including the
high-component subgroup baselines omitted from generic schedule discovery),
autotune reads that schedule directly from the manifest instead of maintaining
a second shell-name allowlist.

CMake selects profiles with `GENERATIVEQC_AOT_PROFILE` or `GENERATIVEQC_AOT_PROFILES`.
`auto` resolves only an exact measured or explicitly compatible profile and is
fail-closed when neither exists. Generic CUDA remains available through the
explicit `portable`/`portable_cuda` profile; it is never selected by an `auto`
miss. `GENERATIVEQC_ENABLE_AOT_SHELLS=OFF` omits all generated shell objects.

Optional production code shapes are declared per manifest row through the
`capabilities` list. The current names are `streaming_fock`, `mixed_fock`, and
`local_packed_streaming_fock`; an omitted list keeps those wrappers disabled.
This metadata is profile-scoped and does not promote a candidate by itself.

Large-shell tuning uses a staged compiler pipeline. Equivalent schedules share
one separately compiled correctness oracle per component mapping; tiled oracle
recurrences are noinline so NVVM does not expand the reference into every
candidate. Timing translation units contain only the RHF kernel they execute.
After ranking, the tuner recompiles the fastest passing candidate with all four
RHF/UHF and persistent production wrappers and applies the resource gates again
before writing a manifest. `--compile-timeout` bounds every NVCC invocation and
terminates its entire process group, preventing timed-out `cicc` children from
surviving into later trials.

### Measured schedule evidence on `sm_120`

Historical autotune and promotion measurements for the D/P/F shells are
retained in the [historical scheduling evidence](../../.agents/notes/implemented/performance/2026-10-09-shell-codegen-schedule-provenance.md). Use the current
manifest for active selections; synthetic timing alone is not a
real-molecule endpoint promotion gate.

## Production AOT policy

The separate [arbitrary-weight ERI consumer](weighted_eri.md) now precontracts
Hermite coefficients before differentiation. Direct-HF psss force uses its
force-only generated expression unconditionally inside the existing native
primitive loops and fixed/resident/paged queues. The older component-cloning candidate is recorded as rejected in the
[historical scheduling evidence](../../.agents/notes/implemented/performance/2026-10-09-shell-codegen-schedule-provenance.md); it is not the production force-only lowering.

`production_shell_classes.json` carries explicit tuned and portable profiles.
The [production shell manifest](../../python/generativeqc_compiler/integral/production_shell_classes.json)
is authoritative for target profiles, active shell classes, Fock/force
consumers, recurrence and per-consumer schedules. Do not copy a static
list into prose: it can drift as new kernels are promoted.

Inspect the checked-out manifest dynamically:

```bash
python - <<'PY'
import json
from pathlib import Path

p = Path("python/generativeqc_compiler/integral/production_shell_classes.json")
for target, profile in sorted(json.loads(p.read_text())["architectures"].items()):
    print(target, profile["kind"])
    for consumer in ("fock", "force"):
        selected = [row["shell_class"] for row in profile["kernels"] if consumer in row["consumers"]]
        print(" ", consumer, ":", " ".join(selected) or "(none)")
PY
```

The `ssss` force mathematics is now compiler-owned in production. Runtime still
keeps the standalone AOT `ssss` force row out of the materialized force queue:
the generated force-only weighted expression executes inside the already
qualified bounded/packed scheduler, avoiding a second task stream. The former
handwritten ssss derivative body and its runtime A/B selector have been deleted.

The generated registry records profile identity, target compute capability,
class index, consumer mask, block size, and component tile. Every profile uses
architecture-suffixed C entry points and scoped device/type identifiers. Each
object target is compiled only for its intended SM; one runtime registry picks
the active `KernelSet` once per device. Architecture list order cannot change
the generated sources or dispatch behavior.

Compile-only builds are supported for `sm_80`, `sm_86`, `sm_89`, `sm_90`, and
`sm_120` with the project's CUDA 12.9 toolkit requirement. Only GPU-backed
performance claims are profile-specific; portable builds never apply RTX 5090
resource goldens.

Adding a class to the mathematical catalog does not make it production AOT.
Production promotion requires:

1. oracle and CUDA compile gates;
2. zero spills and architecture resource limits;
3. isolated schedule timing;
4. real molecular energy/force gates;
5. end-to-end improvement after task classification/dispatch overhead.

Sparse, profile-backed AOT remains intentional. Generating all 55 classes is
now possible mathematically, but compiling and dispatching all of them by
default can increase NVCC time, binary size, instruction-cache pressure, and
queue-management cost.

## Task classification and merge discipline

The runtime now buckets all enabled generated classes together:

```text
classify/count once -> 55-entry device prefix -> materialize class slices
                    -> Fock/force registry dispatches (offset, count, head)
```

The classification byte for each active logical quartet survives the prefix
kernel, so materialization does not decode the exact shell class again. Counts,
offsets, write cursors, and persistent-worker heads stay on the device; no host
readback or synchronization is introduced. Adding another generated class no
longer adds another full active-tile scan.

Historical class-promotion endpoint results and old checkout-integration
diaries are in the [historical scheduling evidence](../../.agents/notes/implemented/performance/2026-10-09-shell-codegen-schedule-provenance.md). Current runtime
dispatch and manifest identity are authoritative.

## Commands

Inspect deterministic source:

```bash
python tools/generate_shell_kernels.py \
  --shell-class dpds --lowering fused --consumer fock
python tools/generate_shell_kernels.py \
  --shell-class dppp --lowering fused --format stats
python tools/generate_shell_kernels.py \
  --shell-class fddd --lowering fused --format stats
cmake --build build --target generativeqc_codegen_pilot
```

Generate the structural report for all 55 catalog classes:

```bash
python tools/report_codegen_capabilities.py \
  --architecture sm_120 \
  --output build/codegen-capabilities.json
```

The report lists target-legal schedule families, recurrence failures, force
derivative-order boundaries, and the production manifest state for every class.
Its top-level `backend` object records the CUDA target, generator ABI, and the
emitter used for validation; `recurrence_supported` and
`force_derivative_supported` provide catalog-wide counts. Each production row
also carries a `status` and `promotion_gate`, so a structurally compilable but
unselected class cannot be mistaken for a production-ready kernel.
The current CUDA force ABI is explicitly reported as order-one-only; a future
Hessian implementation must add a result contract before enabling order two.
To screen a bounded set of classes
that is automatically discovered from the consumer-specific manifest gap:

```bash
python -m generativeqc_compiler.integral.batch_benchmark \
  --discover --consumer force --limit 12 \
  --partition main --gres gpu:5090:1
```

Discovery only chooses synthetic candidates. A real molecular endpoint remains
required before adding a capability or promoting a class in the manifest.
The checked-in `docs/codegen_capabilities.json` is regenerated with the same
command for the current `sm_120` manifest.

Candidate batch screening is intentionally bounded: pass an explicit
`--shell-class` list, a real-profile `--profile` plus `--limit`, or
`--discover` to rank the consumer-specific manifest gap by static work.
Omitting all three is rejected so the tool cannot accidentally compile every
uncovered class in one batch. Profile rows are ranked by measured `primitive_work`,
falling back to `primitive_quartets` for older artifacts; duplicate canonical
class rows are aggregated before applying `--limit`. The same screener accepts
`--consumer fock` to time coefficient-only candidates through the shared
device task slices; force remains the default and the endpoint gate is still
independent of synthetic screening. Production exclusion is consumer-specific:
classes already promoted for force remain eligible for Fock screening until
their value-only path is promoted, and the explicit `--shell-class` and
profile-ranked paths apply the same rule.

The previous 768-AO Slurm and cross-engine commands, including
machine-specific CUDA and environment paths, are in the
[historical scheduling evidence](../../.agents/notes/implemented/performance/2026-10-09-shell-codegen-schedule-provenance.md). Use the active benchmark scripts and
allocated target environment for new comparisons.

Run Python gates:

```bash
python -m pytest tests/python/test_codegen_high_l.py tests/python/test_codegen_production.py tests/python/test_codegen_cuda_compile.py -q
python -m ruff check python/generativeqc_compiler/integral tests/python/test_codegen_high_l.py tests/python/test_codegen_production.py tests/python/test_codegen_cuda_compile.py
```

Run the explicit CUDA gate:

```bash
GENERATIVEQC_NVCC="$(command -v nvcc)" \
GENERATIVEQC_CUDA_ARCH=sm_120 \
python -m pytest tests/python/test_codegen.py -q -s
```

## Remaining work

The former per-shell candidate queue is frozen in the
[historical scheduling evidence](../../.agents/notes/implemented/performance/2026-10-09-shell-codegen-schedule-provenance.md). Discover current gaps from
the production manifest and workload profile. New class promotions
require independent scientific comparison, resource/spill limits and
complete molecular Fock/force endpoint gates. Higher-order nuclear
derivatives require their own tensor result ABI and Hessian qualification.
Use the [maintainer roadmap](../maintainer/roadmap.md) and current issues
for priorities rather than copying the old issue-specific queue.
