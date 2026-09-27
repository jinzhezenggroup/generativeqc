"""Method-name-free stationary execution graph.

This layer describes cross-primitive dataflow only.  It does not allocate,
compile, select a backend, or change scientific arithmetic.  Runtime lifetime,
residency and schedule decisions are deliberately separate follow-up passes.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass

from vibeqc_compiler.common.provenance import canonical_hash

from .stationary_gradient import StationaryGradientPlan

STATIONARY_EXECUTION_GRAPH_VERSION = "stationary-execution-graph-v1"

_INTEGRAL_SOURCES = (
    "one_electron",
    "coulomb",
    "exact_exchange",
    "exchange_short_range",
    "exchange_long_range",
    "overlap_pulay",
    "nuclear",
)
_SEMILOCAL_SOURCES = ("xc_ao", "xc_grid", "xc_weight")
_NONLOCAL_SOURCES = ("nonlocal_ao", "nonlocal_grid", "nonlocal_weight")


@dataclass(frozen=True, slots=True)
class StationaryExecutionNode:
    """One semantic execution node independent of a named functional."""

    name: str
    kind: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]

    def semantic_payload(self) -> typing.Any:
        return {
            "name": self.name,
            "kind": self.kind,
            "inputs": self.inputs,
            "outputs": self.outputs,
        }


@dataclass(frozen=True, slots=True)
class StationaryExecutionValue:
    """Produced value plus the consumers relevant to later lifetime planning."""

    name: str
    producer: str
    consumers: tuple[str, ...]
    memory_space: str
    borrowable: bool
    recomputable: bool
    host_visible: bool

    def semantic_payload(self) -> typing.Any:
        return {
            "name": self.name,
            "producer": self.producer,
            "consumers": self.consumers,
            "memory_space": self.memory_space,
            "borrowable": self.borrowable,
            "recomputable": self.recomputable,
            "host_visible": self.host_visible,
        }


@dataclass(frozen=True, slots=True)
class StationaryExecutionGraph:
    """Cross-primitive stationary dataflow derived from StationaryGradientPlan."""

    plan_identity: str
    source_names: tuple[str, ...]
    grid_features: tuple[str, ...]
    nodes: tuple[StationaryExecutionNode, ...]
    values: tuple[StationaryExecutionValue, ...]
    version: str = STATIONARY_EXECUTION_GRAPH_VERSION

    def __post_init__(self) -> None:
        if len({node.name for node in self.nodes}) != len(self.nodes):
            raise ValueError("stationary execution graph contains duplicate node names")
        if len({value.name for value in self.values}) != len(self.values):
            raise ValueError("stationary execution graph contains duplicate values")
        produced = {value.name for value in self.values}
        expected = {f"source:{name}" for name in self.source_names}
        if not expected.issubset(produced):
            raise ValueError(
                "stationary execution graph does not cover every gradient source"
            )

    def semantic_payload(self) -> typing.Any:
        return {
            "version": self.version,
            "plan_identity": self.plan_identity,
            "source_names": self.source_names,
            "grid_features": self.grid_features,
            "nodes": [node.semantic_payload() for node in self.nodes],
            "values": [value.semantic_payload() for value in self.values],
        }

    @property
    def identity(self) -> str:
        return canonical_hash(self.semantic_payload())

    def node(self, name: str) -> StationaryExecutionNode:
        for node in self.nodes:
            if node.name == name:
                return node
        raise KeyError(name)

    def value(self, name: str) -> StationaryExecutionValue:
        for value in self.values:
            if value.name == name:
                return value
        raise KeyError(name)


def _grid_features(plan: StationaryGradientPlan) -> tuple[str, ...]:
    """Map scientific density ingredients onto reusable CUDA-grid feature owners."""

    ingredients = set(plan.method.requirements["ingredients"])
    features: list[str] = []
    if "rho" in ingredients:
        features.append("rho")
    if "sigma" in ingredients:
        # sigma is formed from the Cartesian density gradient; retaining that
        # primitive also serves nonlocal and geometry-response consumers.
        features.append("gradient")
    if "tau" in ingredients:
        features.append("tau")
    return tuple(features)


def _source_node(source: str) -> StationaryExecutionNode:
    if source == "one_electron":
        inputs = ("final_density",)
    elif source == "overlap_pulay":
        inputs = ("weighted_density",)
    elif source == "nuclear":
        inputs = ()
    else:
        inputs = ("final_density",)
    return StationaryExecutionNode(
        name=f"integral:{source}",
        kind="integral-derivative-source",
        inputs=inputs,
        outputs=(f"source:{source}",),
    )


def compile_stationary_execution_graph(
    plan: StationaryGradientPlan,
) -> StationaryExecutionGraph:
    """Compile one stationary scientific plan into cross-primitive dataflow.

    The result intentionally contains no method identifier, backend policy,
    allocation size or kernel choice.  Two aliases with the same MethodIR
    semantics therefore share this graph identity.
    """

    if not isinstance(plan, StationaryGradientPlan):
        raise TypeError("stationary execution graph requires StationaryGradientPlan")

    features = _grid_features(plan)
    nodes: list[StationaryExecutionNode] = [
        StationaryExecutionNode(
            "final_state",
            "stationary-state",
            (),
            ("final_density", "weighted_density"),
        )
    ]
    if features:
        nodes.append(
            StationaryExecutionNode(
                "grid_features",
                "density-feature-producer",
                ("final_density",),
                tuple(f"grid_feature:{name}" for name in features),
            )
        )

    for source in plan.source_names:
        if source in _INTEGRAL_SOURCES:
            nodes.append(_source_node(source))

    semilocal = tuple(
        source for source in plan.source_names if source in _SEMILOCAL_SOURCES
    )
    if semilocal:
        nodes.append(
            StationaryExecutionNode(
                "semilocal_geometry",
                "semilocal-geometry-consumer",
                tuple(f"grid_feature:{name}" for name in features),
                tuple(f"source:{name}" for name in semilocal),
            )
        )

    nonlocal_sources = tuple(
        source for source in plan.source_names if source in _NONLOCAL_SOURCES
    )
    if nonlocal_sources:
        nonlocal_inputs = tuple(
            f"grid_feature:{name}" for name in features if name in ("rho", "gradient")
        )
        nodes.extend(
            (
                StationaryExecutionNode(
                    "nonlocal_pairs",
                    "nonlocal-pair-producer",
                    nonlocal_inputs,
                    ("nonlocal_seeds",),
                ),
                StationaryExecutionNode(
                    "nonlocal_geometry",
                    "nonlocal-geometry-consumer",
                    (*nonlocal_inputs, "nonlocal_seeds"),
                    tuple(f"source:{name}" for name in nonlocal_sources),
                ),
            )
        )

    nodes.append(
        StationaryExecutionNode(
            "gradient_reduction",
            "stationary-source-reduction",
            tuple(f"source:{name}" for name in plan.source_names),
            ("gradient",),
        )
    )

    producer: dict[str, str] = {}
    consumers: dict[str, list[str]] = {}
    for node in nodes:
        for output in node.outputs:
            if output in producer:
                raise ValueError(
                    f"stationary execution value {output!r} has two producers"
                )
            producer[output] = node.name
        for value in node.inputs:
            consumers.setdefault(value, []).append(node.name)

    values = []
    for name, owner in producer.items():
        host_visible = name == "gradient"
        values.append(
            StationaryExecutionValue(
                name=name,
                producer=owner,
                consumers=tuple(consumers.get(name, ())),
                memory_space="device",
                borrowable=not host_visible,
                recomputable=name.startswith("grid_feature:")
                or name == "nonlocal_seeds",
                host_visible=host_visible,
            )
        )

    return StationaryExecutionGraph(
        plan_identity=plan.identity,
        source_names=plan.source_names,
        grid_features=features,
        nodes=tuple(nodes),
        values=tuple(values),
    )
