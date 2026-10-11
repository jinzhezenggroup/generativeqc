"""User transforms that compile canonical TensorIR derivative DAGs.

Only argument binding, seed validation and output packaging live here. The
compiler owns every derivative rule, including scientific index domains and
symmetry adjoints. No reference JVP/VJP evaluator is used by native transforms.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Mapping
from typing import Any

import numpy as np
from generativeqc_compiler.tensor import Program, linearize, transpose_program

from . import _array_api_eager as _eager
from . import _array_api_symbolic as _symbolic


def _compiled(function: Any, backend: str | None, options: dict[str, Any]) -> Any:
    from .array_api import CompiledFunction, compile

    if isinstance(function, CompiledFunction):
        configuration = {**function._options, **options}
        specs = configuration.pop("specs", function._specs)
        differentiable = configuration.pop(
            "differentiable", tuple(function._differentiable)
        )
        return CompiledFunction(
            function._function,
            backend=backend or function._backend,
            differentiable=differentiable,
            specs=specs,
            **configuration,
        )
    return compile(function, backend=backend or "cpu", **options)


def _arguments(function: Any, argnums: Any) -> tuple[tuple[str, ...], bool]:
    parameters = tuple(
        name
        for name, parameter in function._signature.parameters.items()
        if parameter.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    )
    single = type(argnums) is int
    positions = (argnums,) if single else argnums
    if not isinstance(positions, tuple) or not positions:
        raise TypeError("argnums must be an integer or nonempty tuple of integers")
    names = []
    for position in positions:
        if type(position) is not int or not -len(parameters) <= position < len(
            parameters
        ):
            raise ValueError("argnums must index positional function parameters")
        names.append(parameters[position])
    if len(set(names)) != len(names):
        raise ValueError("argnums must select distinct arguments")
    return tuple(names), single


def _seed(value: Any, spec: Any, name: str) -> np.ndarray:
    array = _symbolic.asarray(value)
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a concrete array")
    if array.shape != spec.shape or array.dtype.name != spec.dtype:
        raise ValueError(f"{name} must match primal shape and dtype")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _combine(primal: Program, derivative: Program) -> Program:
    if set(primal.outputs) & set(derivative.outputs):
        raise ValueError("primal and derivative output names collide")
    return Program(
        {**primal.outputs, **derivative.outputs},
        tuple(dict.fromkeys((*primal.nodes, *derivative.nodes))),
        provenance={
            **dict(derivative.provenance),
            "public_array_transform": "primal-and-derivative",
        },
    )


def _outputs(program: Program, values: dict[str, Any]) -> Any:
    if tuple(program.outputs) == ("output",):
        return _eager.wrap(values["output"])
    return {name: _eager.wrap(values[name]) for name in program.outputs}


class _GradientFunction:
    def __init__(
        self,
        function: Any,
        argnums: Any,
        backend: str | None,
        options: dict[str, Any],
        *,
        with_value: bool,
    ) -> None:
        base = _compiled(function, backend, options)
        self.names, self.single = _arguments(base, argnums)
        self.function = base._variant(self.names)
        self.with_value = with_value
        self.programs: dict[tuple[Any, ...], Any] = {}
        functools.update_wrapper(self, base._function, updated=())

    def _prepare(self, *args: Any, **kwargs: Any) -> Any:
        primal, feeds = self.function._prepare(*args, **kwargs)
        if tuple(primal.outputs) != ("output",) or primal.outputs["output"].spec.shape:
            raise ValueError(
                "grad/value_and_grad require one scalar array output; use vjp"
            )
        # Equation hashes omit disconnected arguments, but their gradients still
        # have the declared shape/dtype/scientific type for this specialization.
        input_specs = {
            node.attrs["name"]: node.spec for node in primal.nodes if node.op == "input"
        }
        key = (primal.logical_hash, *(input_specs[name] for name in self.names))
        cached = self.programs.get(key)
        if cached is None:
            reverse = transpose_program(primal, ("output",), inputs=self.names)
            program = (
                _combine(primal, reverse.program)
                if self.with_value
                else reverse.program
            )
            mapping = {
                original: generated
                for generated, original in reverse.output_map.items()
            }
            cached = program, mapping, reverse.input_map
            self.programs[key] = cached
        program, mapping, seeds = cached
        for generated, original in seeds.items():
            feeds[generated] = np.ones((), dtype=primal.outputs[original].spec.dtype)
        return program, feeds, mapping

    def lower(self, *args: Any, **kwargs: Any) -> Program:
        """Inspect the actual derivative/combined graph that will be compiled."""
        return self._prepare(*args, **kwargs)[0]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        program, feeds, mapping = self._prepare(*args, **kwargs)
        outputs = self.function._execution.execute(program, feeds)
        gradients = tuple(_eager.wrap(outputs[mapping[name]]) for name in self.names)
        result = gradients[0] if self.single else gradients
        return (_eager.wrap(outputs["output"]), result) if self.with_value else result

    def compilation_info(self) -> dict[str, object]:
        """Report the execution backend and cached native derivative endpoints."""
        return self.function.compilation_info()

    def close(self) -> None:
        """Release compiled derivative executors and reject subsequent execution."""
        self.function.close()


def _jvp(
    function: Any,
    primals: Any,
    tangents: Any,
    backend: str | None,
    options: dict[str, Any],
) -> tuple[Any, Any]:
    if (
        not isinstance(primals, tuple)
        or not isinstance(tangents, tuple)
        or len(primals) != len(tangents)
    ):
        raise TypeError("primals and tangents must be matching tuples")
    base = _compiled(function, backend, options)
    names, _ = _arguments(base, tuple(range(len(primals))))
    base = base._variant(names)
    primal, feeds = base._prepare(*primals)
    forward = linearize(primal, names)
    inputs = {node.attrs["name"]: node for node in primal.nodes if node.op == "input"}
    directions = dict(zip(names, tangents, strict=True))
    for generated, name in forward.input_map.items():
        feeds[generated] = _seed(directions[name], inputs[name].spec, f"tangent {name}")
    combined = _combine(primal, forward.program)
    try:
        values = base._execution.execute(combined, feeds)
        derivative = {
            original: values[generated]
            for generated, original in forward.output_map.items()
        }
        return _outputs(primal, values), _outputs(primal, derivative)
    finally:
        base.close()


class _Pullback:
    def __init__(
        self,
        function: Any,
        primal: Program,
        feeds: dict[str, Any],
        names: tuple[str, ...],
    ) -> None:
        self.function, self.primal, self.feeds, self.names = (
            function,
            primal,
            feeds,
            names,
        )
        self.reverse = transpose_program(primal, tuple(primal.outputs), inputs=names)

    def __call__(self, cotangents: Any) -> tuple[Any, ...]:
        if tuple(self.primal.outputs) == ("output",):
            cotangents = {"output": cotangents}
        if not isinstance(cotangents, Mapping) or set(cotangents) != set(
            self.primal.outputs
        ):
            raise ValueError("cotangents must match all function outputs")
        feeds = dict(self.feeds)
        for generated, original in self.reverse.input_map.items():
            feeds[generated] = _seed(
                cotangents[original],
                self.primal.outputs[original].spec,
                f"cotangent {original}",
            )
        values = self.function._execution.execute(self.reverse.program, feeds)
        mapping = {
            original: generated
            for generated, original in self.reverse.output_map.items()
        }
        return tuple(_eager.wrap(values[mapping[name]]) for name in self.names)

    def close(self) -> None:
        """Release the snapshot's primal and derivative native execution owners."""
        self.function.close()
        self.feeds.clear()


def _vjp(
    function: Any,
    primals: tuple[Any, ...],
    backend: str | None,
    options: dict[str, Any],
) -> tuple[Any, _Pullback]:
    base = _compiled(function, backend, options)
    names, _ = _arguments(base, tuple(range(len(primals))))
    base = base._variant(names)
    primal, feeds = base._prepare(*primals)
    # A pullback belongs to this forward call, not to subsequently mutated inputs.
    feeds = {name: np.array(value, copy=True) for name, value in feeds.items()}
    try:
        pullback = _Pullback(base, primal, feeds, names)
        values = base._execution.execute(primal, feeds)
        return _outputs(primal, values), pullback
    except BaseException:
        base.close()
        raise
