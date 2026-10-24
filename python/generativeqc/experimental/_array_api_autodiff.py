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


def _arguments(
    function: Any, argnums: Any, *, positional_count: int | None = None
) -> tuple[tuple[str, ...], bool]:
    parameters = tuple(
        name
        for name, parameter in function._signature.parameters.items()
        if parameter.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    )
    if positional_count is not None:
        parameters = parameters[:positional_count]
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
    if any(
        derivative.provenance.get(key)
        for key in (
            "external_mask_where_jvp",
            "external_mask_where_vjp",
            "comparison_where_jvp",
            "comparison_where_vjp",
        )
    ):
        # Fixed-mask derivatives freeze real feeds as non-differentiable execution
        # data. Rebuild the primal against those same canonical inputs before
        # combining; restoring their AD flags would reopen higher-order AD.
        from generativeqc_compiler.tensor.ad_program import _rebuild

        inputs = {
            node.attrs["name"]: node for node in derivative.nodes if node.op == "input"
        }
        primal = _rebuild(
            primal,
            {
                node: inputs[node.attrs["name"]]
                for node in primal.nodes
                if node.op == "input"
            },
        )
    if set(primal.outputs) & set(derivative.outputs):
        raise ValueError("primal and derivative output names collide")
    return Program(
        {**primal.outputs, **derivative.outputs},
        tuple(dict.fromkeys((*primal.nodes, *derivative.nodes))),
        provenance={
            **dict(primal.provenance),
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
        allow_external_mask_where: bool = False,
        allow_comparison_where: bool = False,
    ) -> None:
        if type(allow_external_mask_where) is not bool:
            raise TypeError("allow_external_mask_where must be Boolean")
        if type(allow_comparison_where) is not bool:
            raise TypeError("allow_comparison_where must be Boolean")
        base = _compiled(function, backend, options)
        if (
            allow_external_mask_where or allow_comparison_where
        ) and base._backend not in ("reference", "cpu"):
            base.close()
            raise ValueError("external-mask gradients require reference or CPU backend")
        self.allow_external_mask_where = allow_external_mask_where
        self.allow_comparison_where = allow_comparison_where
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
            reverse = transpose_program(
                primal,
                ("output",),
                inputs=self.names,
                allow_external_mask_where=self.allow_external_mask_where,
                allow_comparison_where=self.allow_comparison_where,
            )
            program = (
                _combine(primal, reverse.program)
                if self.with_value
                else Program(
                    reverse.program.outputs,
                    reverse.program.nodes,
                    provenance={
                        **dict(primal.provenance),
                        **dict(reverse.program.provenance),
                    },
                )
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


def _gradient_jvp(
    function: _GradientFunction,
    primals: tuple[Any, ...],
    tangents: tuple[Any, ...],
    backend: str | None,
    options: dict[str, Any],
) -> tuple[Any, Any]:
    """Forward-over-reverse with second-order molecular jets captured up front.

    Input/source snapshots are bound only after both TensorIR transforms. A
    fresh compiled variant prevents a cached first-order source graph from
    being mistaken for a quadratic jet on a later Hessian evaluation.
    """
    from ._array_api_molecular import _SOURCE_ORDER

    if function.allow_external_mask_where or function.allow_comparison_where:
        raise ValueError("second-order mask gradients are unsupported")
    base = _compiled(function.function, backend, options)
    try:
        names, _ = _arguments(base, tuple(range(len(primals))))
        base = base._variant(names)
        token = _SOURCE_ORDER.set(2)
        try:
            primal, feeds = base._prepare(*primals)
        finally:
            _SOURCE_ORDER.reset(token)
        if tuple(primal.outputs) != ("output",) or primal.outputs["output"].spec.shape:
            raise ValueError("gradient JVP requires a scalar objective")
        reverse = transpose_program(primal, ("output",), inputs=function.names)
        for seed, original in reverse.input_map.items():
            feeds[seed] = np.ones((), dtype=primal.outputs[original].spec.dtype)
        derivative = (
            _combine(primal, reverse.program)
            if function.with_value
            else reverse.program
        )
        forward = linearize(derivative, names)
        specs = {
            node.attrs["name"]: node.spec for node in primal.nodes if node.op == "input"
        }
        directions = dict(zip(names, tangents, strict=True))
        for seed, name in forward.input_map.items():
            feeds[seed] = _seed(directions[name], specs[name], f"tangent {name}")
        values = base._execution.execute(_combine(derivative, forward.program), feeds)
        reverse_names = {
            original: generated for generated, original in reverse.output_map.items()
        }
        forward_names = {
            original: generated for generated, original in forward.output_map.items()
        }
        gradients = tuple(
            _eager.wrap(values[reverse_names[name]]) for name in function.names
        )
        products = tuple(
            _eager.wrap(values[forward_names[reverse_names[name]]])
            for name in function.names
        )
        first = gradients[0] if function.single else gradients
        second = products[0] if function.single else products
        if function.with_value:
            return (
                (_eager.wrap(values["output"]), first),
                (_eager.wrap(values[forward_names["output"]]), second),
            )
        return first, second
    finally:
        base.close()


def _jvp(
    function: Any,
    primals: Any,
    tangents: Any,
    backend: str | None,
    options: dict[str, Any],
    *,
    allow_external_mask_where: bool = False,
    allow_comparison_where: bool = False,
) -> tuple[Any, Any]:
    if type(allow_external_mask_where) is not bool:
        raise TypeError("allow_external_mask_where must be Boolean")
    if type(allow_comparison_where) is not bool:
        raise TypeError("allow_comparison_where must be Boolean")
    if (
        not isinstance(primals, tuple)
        or not isinstance(tangents, tuple)
        or len(primals) != len(tangents)
    ):
        raise TypeError("primals and tangents must be matching tuples")
    if isinstance(function, _GradientFunction):
        if allow_external_mask_where or allow_comparison_where:
            raise ValueError("second-order mask gradients are unsupported")
        return _gradient_jvp(function, primals, tangents, backend, options)
    positions = tuple(
        position
        for position, tangent in enumerate(tangents)
        if not (allow_external_mask_where or allow_comparison_where)
        or tangent is not None
    )
    if (allow_external_mask_where or allow_comparison_where) and not positions:
        raise ValueError("external-mask JVP requires at least one real tangent")
    base = _compiled(function, backend, options)
    try:
        if (
            allow_external_mask_where or allow_comparison_where
        ) and base._backend not in ("reference", "cpu"):
            raise ValueError("external-mask JVP requires reference or CPU backend")
        names, _ = _arguments(base, positions)
        base = base._variant(names)
        primal, feeds = base._prepare(*primals)
        forward = linearize(
            primal,
            names,
            allow_external_mask_where=allow_external_mask_where,
            allow_comparison_where=allow_comparison_where,
        )
        inputs = {
            node.attrs["name"]: node for node in primal.nodes if node.op == "input"
        }
        directions = dict(
            zip(names, (tangents[position] for position in positions), strict=True)
        )
        for generated, name in forward.input_map.items():
            feeds[generated] = _seed(
                directions[name], inputs[name].spec, f"tangent {name}"
            )
        combined = _combine(primal, forward.program)
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
        *,
        allow_external_mask_where: bool = False,
        allow_comparison_where: bool = False,
    ) -> None:
        self.function, self.primal, self.feeds, self.names = (
            function,
            primal,
            feeds,
            names,
        )
        self.reverse = transpose_program(
            primal,
            tuple(primal.outputs),
            inputs=names,
            allow_external_mask_where=allow_external_mask_where,
            allow_comparison_where=allow_comparison_where,
        )

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
    *,
    argnums: int | tuple[int, ...] | None = None,
    allow_external_mask_where: bool = False,
    allow_comparison_where: bool = False,
) -> tuple[Any, _Pullback]:
    if type(allow_external_mask_where) is not bool:
        raise TypeError("allow_external_mask_where must be Boolean")
    if type(allow_comparison_where) is not bool:
        raise TypeError("allow_comparison_where must be Boolean")
    base = _compiled(function, backend, options)
    try:
        if (
            allow_external_mask_where or allow_comparison_where
        ) and base._backend not in ("reference", "cpu"):
            raise ValueError("external-mask VJP requires reference or CPU backend")
        names, _ = _arguments(
            base,
            tuple(range(len(primals))) if argnums is None else argnums,
            positional_count=len(primals),
        )
        base = base._variant(names)
        primal, feeds = base._prepare(*primals)
        # Snapshot every primal, including selectors and non-differentiated data.
        feeds = {name: np.array(value, copy=True) for name, value in feeds.items()}
        pullback = _Pullback(
            base,
            primal,
            feeds,
            names,
            allow_external_mask_where=allow_external_mask_where,
            allow_comparison_where=allow_comparison_where,
        )
        values = base._execution.execute(primal, feeds)
        return _outputs(primal, values), pullback
    except BaseException:
        base.close()
        raise
