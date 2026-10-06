"""Bounded runtime-indexed CUDA schedules for pure TensorIR reduction regions.

Only pointwise linear producers, independent reductions, and linear output
composition are admitted. Equations and axis maps come from TensorIR; the
schedule changes traversal and FP64 summation order, not the graph. A caller
must retain its original lowering and independently qualify the tree order.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from .ir import Node
    from .program import Program
    from .types import Index, TensorSpec


@dataclass(frozen=True)
class IndexedReductionSchedule:
    """Fixed FP64 trees with a bounded, reusable scalar-partial frontier.

    Each scalar uses at most ``maximum_partials`` CTAs followed by one ordered
    tree. Vector outputs use one CTA per output, sharing the traversal of their
    compatible reduction frontiers without materializing pointwise producers.
    No floating-point atomics or device-dependent launch policy are used.
    """

    threads: int = 256
    maximum_partials: int = 256
    scalar_tile_elements: int = 4096

    def __post_init__(self) -> None:
        if type(self.threads) is not int or not 32 <= self.threads <= 1024:
            raise ValueError("reduction threads must be a whole number of CUDA warps")
        if self.threads % 32:
            raise ValueError("reduction threads must be a whole number of CUDA warps")
        for name in ("maximum_partials", "scalar_tile_elements"):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= 65535:
                raise ValueError(f"{name} must be a positive bounded integer")


@dataclass(frozen=True)
class IndexedReductionPlan:
    """A compiler-visible region with distinct output roots and shared seeds."""

    program: Program
    schedule: IndexedReductionSchedule
    roots: tuple[Node, ...]
    frontiers: tuple[tuple[Node, ...], ...]
    inputs: tuple[Node, ...]

    def to_payload(self) -> dict[str, object]:
        nodes = self.program.live_nodes
        return {
            "schema": "generativeqc.indexed-cuda-reduction.v1",
            "program": self.program.logical_hash,
            "schedule": asdict(self.schedule),
            "roots": [nodes.index(root) for root in self.roots],
            "frontiers": [
                [nodes.index(node) for node in frontier] for frontier in self.frontiers
            ],
            "arithmetic": "fp64-rn-fixed-warp-tree-no-floating-atomics",
            "producer_storage": "streamed",
            "partial_storage": "runtime-tile-count-capped-by-schedule",
        }

    @property
    def identity(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_payload(), sort_keys=True).encode()
        ).hexdigest()


def plan_indexed_reduction(
    program: Program, schedule: IndexedReductionSchedule | None = None
) -> IndexedReductionPlan:
    """Prove a small linear reduction region; reject unsupported composition.

    Nested reductions, broadcast/index maps, nonlinear producers, precision
    changes, and differently shaped output frontiers require other lowerings.
    Aliased outputs keep a single root and therefore a single evaluation.
    """

    nodes = program.live_nodes
    if any(node.spec.dtype != "float64" for node in nodes):
        raise ValueError("indexed reduction requires an FP64 region")
    roots = tuple(dict.fromkeys(program.outputs.values()))
    frontiers = []

    def producer(node: Node) -> None:
        if node.op == "input":
            return
        if node.op != "add":
            raise ValueError("indexed reduction requires pointwise linear producers")
        for child in node.inputs:
            if child.spec.shape != node.spec.shape:
                raise ValueError("indexed reduction producer shape mismatch")
            producer(child)

    def output(node: Node, frontier: list[Node], root: Node) -> None:
        if node.spec.shape != root.spec.shape:
            raise ValueError("indexed reduction requires compatible output frontiers")
        if node.op == "reduce":
            if not node.attrs["axes"]:
                raise ValueError("indexed reduction requires a nonempty reduction")
            producer(node.inputs[0])
            if node not in frontier:
                frontier.append(node)
            return
        if node.op != "add":
            raise ValueError("indexed reduction output must compose reductions")
        for child in node.inputs:
            output(child, frontier, root)

    for root in roots:
        frontier: list[Node] = []
        output(root, frontier, root)
        if not root.spec.indices and root.op != "reduce":
            raise ValueError("scalar partials require a single reduction root")
        frontiers.append(tuple(frontier))
    return IndexedReductionPlan(
        program,
        schedule or IndexedReductionSchedule(),
        roots,
        tuple(frontiers),
        tuple(node for node in nodes if node.op == "input"),
    )


def emit_indexed_reduction_cuda(
    plan: IndexedReductionPlan,
    prefix: str,
    *,
    dimension: Callable[[Index], str],
    parameters: tuple[str, ...],
    input_bindings: Mapping[str, str],
    state_type: str,
    output_type: str,
) -> tuple[str, str, str]:
    """Emit kernels/runner, bounded arena sizing, and exact launch-count code.

    The caller binds positive symbolic dimensions and stream-owned storage;
    input storage must not overlap the output/partial arena. Partial
    scratch is reused after each scalar drain on that same stream; publication
    remains the caller's responsibility. Counts describe actual launches,
    not the number of logical TensorIR nodes.
    """

    schedule = plan.schedule
    nodes = plan.program.live_nodes
    input_numbers = {node: number for number, node in enumerate(plan.inputs)}
    root_numbers = {node: number for number, node in enumerate(plan.roots)}
    warp_count = schedule.threads // 32
    dimensions = ",".join(f"std::size_t {name}" for name in parameters)
    argument_names = ",".join(parameters)

    def size(spec: TensorSpec) -> str:
        return "*".join(f"({dimension(index)})" for index in spec.indices) or "1"

    def checked_size(spec: TensorSpec) -> str:
        return (
            "checked_product({"
            + ",".join(dimension(index) for index in spec.indices)
            + "})"
            if spec.indices
            else "1"
        )

    partial_capacity = "0"
    for root, frontier in zip(plan.roots, plan.frontiers, strict=True):
        if not root.spec.indices:
            extent = checked_size(frontier[0].inputs[0].spec)
            blocks = f"{prefix}_partial_blocks({extent})"
            partial_capacity = (
                f"std::max<std::size_t>({partial_capacity},({blocks}>1?{blocks}:0))"
            )

    def linear(node: Node, leaves: Mapping[Node, str]) -> str:
        if node in leaves:
            return leaves[node]
        terms = []
        for child, coefficient in zip(
            node.inputs, node.attrs["coefficients"], strict=True
        ):
            numerator, denominator = coefficient
            factor = f"({numerator}.0/{denominator}.0)"
            terms.append(f"__dmul_rn({factor},{linear(child, leaves)})")
        expression = terms[0]
        for term in terms[1:]:
            expression = f"__dadd_rn({expression},{term})"
        return f"generativeqc_tensor::finite({expression},error,{nodes.index(node)})"

    def contribution(node: Node) -> tuple[str, str]:
        source = node.inputs[0]
        axes = tuple(sorted(node.attrs["axes"]))
        retained = tuple(
            axis for axis in range(len(source.spec.indices)) if axis not in axes
        )
        coordinates = {}
        for selected, flat in (
            (axes, "reduction_index"),
            (retained, "output_index"),
        ):
            for position, axis in enumerate(selected):
                stride = "*".join(
                    f"({dimension(source.spec.indices[following])})"
                    for following in selected[position + 1 :]
                )
                quotient = f"({flat}/({stride}))" if stride else flat
                coordinates[axis] = (
                    f"({quotient}%({dimension(source.spec.indices[axis])}))"
                )
        address = "0"
        for axis, index in enumerate(source.spec.indices):
            address = f"({address}*({dimension(index)})+{coordinates[axis]})"
        leaves = {
            seed: f"input_{number}[{address}]" for seed, number in input_numbers.items()
        }
        extent = "*".join(f"({dimension(source.spec.indices[axis])})" for axis in axes)
        return linear(source, leaves), extent

    input_arguments = [
        f"const double* input_{number}" for number in range(len(plan.inputs))
    ]
    signature = ",".join([*input_arguments, "double* output", dimensions, "int* error"])
    bound_inputs = ",".join(input_bindings[node.attrs["name"]] for node in plan.inputs)
    kernels = []
    allocations = [f"  double* partials=arena; std::size_t cursor={partial_capacity};"]
    launches = []
    launch_counts = []
    materialized = []
    reads = []
    writes = []
    summands = []
    required = partial_capacity
    for group, (root, frontier) in enumerate(
        zip(plan.roots, plan.frontiers, strict=True)
    ):
        scalar = not root.spec.indices
        output_count = size(root.spec)
        allocations += [
            f"  double* output_{group}=arena+cursor;",
            f"  cursor=checked_add(cursor,{checked_size(root.spec)});",
        ]
        required = f"checked_add({required},{checked_size(root.spec)})"
        writes.append(checked_size(root.spec))
        for reduction in frontier:
            summands.append(checked_size(reduction.inputs[0].spec))

            def input_count(node: Node) -> int:
                return (
                    1
                    if node.op == "input"
                    else sum(input_count(child) for child in node.inputs)
                )

            reads.extend(
                [checked_size(reduction.inputs[0].spec)]
                * input_count(reduction.inputs[0])
            )
        lines = [
            f"__global__ void {prefix}_group_{group}({signature}){{",
            f"  __shared__ double shared[{len(frontier)}][{warp_count}];",
            "  const unsigned lane=threadIdx.x&31,warp=threadIdx.x>>5;",
        ]
        if scalar:
            lines.append("  {")
        else:
            lines.append(
                f"  for(std::size_t output_index=blockIdx.x;output_index<{output_count};output_index+=gridDim.x){{"
            )
        expressions = [contribution(node) for node in frontier]
        for reduction, (expression, extent) in enumerate(expressions):
            first = (
                "std::size_t(blockIdx.x)*blockDim.x+threadIdx.x"
                if scalar
                else "threadIdx.x"
            )
            stride = "std::size_t(blockDim.x)*gridDim.x" if scalar else "blockDim.x"
            lines += [
                f"    double value_{reduction}=0.0;",
                f"    for(std::size_t reduction_index={first};reduction_index<{extent};reduction_index+={stride})",
                f"      value_{reduction}=__dadd_rn(value_{reduction},{expression});",
                "    for(unsigned offset=16;offset;offset>>=1)",
                f"      value_{reduction}=__dadd_rn(value_{reduction},__shfl_down_sync(0xffffffffu,value_{reduction},offset));",
                f"    if(lane==0) shared[{reduction}][warp]=value_{reduction};",
            ]
        lines += ["    __syncthreads();", "    if(warp==0){"]
        for reduction in range(len(frontier)):
            lines += [
                f"      value_{reduction}=lane<{warp_count}?shared[{reduction}][lane]:0.0;",
                "      for(unsigned offset=16;offset;offset>>=1)",
                f"        value_{reduction}=__dadd_rn(value_{reduction},__shfl_down_sync(0xffffffffu,value_{reduction},offset));",
                f"      if(lane==0) value_{reduction}=generativeqc_tensor::finite(value_{reduction},error,{nodes.index(frontier[reduction])});",
            ]
        result = linear(
            root,
            {node: f"value_{reduction}" for reduction, node in enumerate(frontier)},
        )
        address = "blockIdx.x" if scalar else "output_index"
        lines += [
            f"      if(lane==0) output[{address}]={result};",
            "    }",
            "    __syncthreads();",
            "  }",
            "}",
        ]
        kernels.append("\n".join(lines))
        if scalar:
            extent = expressions[0][1]
            launches += [
                f"  const auto blocks_{group}={prefix}_partial_blocks({extent});",
                f"  {prefix}_group_{group}<<<blocks_{group},{schedule.threads},0,state.stream>>>({bound_inputs},blocks_{group}>1?partials:output_{group},{argument_names},state.error);",
                "  generativeqc_tensor::cuda_check(cudaGetLastError());",
                f"  if(blocks_{group}>1){{",
                f"    {prefix}_drain<<<1,{schedule.threads},0,state.stream>>>(partials,blocks_{group},output_{group},state.error);",
                "    generativeqc_tensor::cuda_check(cudaGetLastError());",
                "  }",
            ]
            launch_counts.append(f"1+({prefix}_partial_blocks({extent})>1)")
            temporary = f"({prefix}_partial_blocks({extent})>1?{prefix}_partial_blocks({extent}):0)"
            materialized.append(temporary)
            reads.append(temporary)
            writes.append(temporary)
            summands.append(temporary)
        else:
            launches += [
                f"  {prefix}_group_{group}<<<generativeqc_tensor::blocks({output_count},1),{schedule.threads},0,state.stream>>>({bound_inputs},output_{group},{argument_names},state.error);",
                "  generativeqc_tensor::cuda_check(cudaGetLastError());",
            ]
            launch_counts.append("1")
    drain = f"""__global__ void {prefix}_drain(const double* partials,std::size_t count,double* output,int* error){{
  __shared__ double shared[{warp_count}];
  double value=0.0;
  for(std::size_t index=threadIdx.x;index<count;index+=blockDim.x)
    value=__dadd_rn(value,partials[index]);
  for(unsigned offset=16;offset;offset>>=1)
    value=__dadd_rn(value,__shfl_down_sync(0xffffffffu,value,offset));
  const unsigned lane=threadIdx.x&31,warp=threadIdx.x>>5;
  if(lane==0) shared[warp]=value;
  __syncthreads();
  if(warp==0){{
    value=lane<{warp_count}?shared[lane]:0.0;
    for(unsigned offset=16;offset;offset>>=1)
      value=__dadd_rn(value,__shfl_down_sync(0xffffffffu,value,offset));
    if(lane==0) output[0]=generativeqc_tensor::finite(value,error,0);
  }}
}}"""
    helper = f"""inline unsigned {prefix}_partial_blocks(std::size_t count){{
  const auto tiles=count/{schedule.scalar_tile_elements}+(count%{schedule.scalar_tile_elements}!=0);
  return static_cast<unsigned>(std::min<std::size_t>({schedule.maximum_partials},tiles));
}}
inline std::size_t {prefix}_arena_elements({dimensions}){{return {required};}}
inline std::size_t {prefix}_kernel_count({dimensions}){{return {"+".join(launch_counts)};}}
inline constexpr const char* {prefix}_schedule_identity="{plan.identity}";"""
    for name, terms in (
        ("materialized_elements", materialized),
        ("value_reads", reads),
        ("value_writes", writes),
        ("reduction_summands", summands),
    ):
        count = "0"
        for term in terms:
            count = f"checked_add({count},{term})"
        helper += (
            f"\ninline std::size_t {prefix}_{name}({dimensions}){{return {count};}}"
        )
    returned = ",".join(
        f"output_{root_numbers[root]}" for root in plan.program.outputs.values()
    )
    runner = [
        f"static {output_type} run_{prefix}({state_type}& state){{",
        *(f"  const auto {name}=state.{name};" for name in parameters),
        "  auto* arena=state.response_arena;",
        *allocations,
        *launches,
        f"  return {{{returned}}};",
        "}",
    ]
    return "\n".join([*kernels, drain, *runner]), helper, plan.identity
