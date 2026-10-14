"""Fixed external-mask generated JVP: independent math and eager domains."""

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from generativeqc.experimental import array_api as xp
from generativeqc_compiler.tensor import Program, dot_test, execute, jvp, optimize, vjp
from generativeqc_compiler.tensor.ad_program import linearize, transpose_program


def capture(function: Any, feeds: dict, names: tuple = ("x", "y")) -> Program:
    return xp.compile(function, backend="reference", differentiable=names).lower(
        *feeds.values()
    )


def generated(program: Program, names: tuple = ("x", "y"), **options: Any) -> Any:
    return linearize(program, names, allow_external_mask_where=True, **options)


def run(program: Program, feeds: dict, backend: str, cache: Path) -> dict:
    if backend == "reference":
        return execute(program, feeds).outputs
    from generativeqc_compiler.common.cpp_adapter import CppCompilerAdapter
    from generativeqc_compiler.tensor.cpu import NativeTensorProgram

    owner = NativeTensorProgram(
        program,
        compiler=CppCompilerAdapter(Path(os.environ.get("CXX", "c++"))),
        cache=cache,
    )
    return owner.execute(feeds)


@pytest.mark.parametrize("backend", ["reference", "native"])
@pytest.mark.parametrize("shape", [(), (5,), (0, 3)])
def test_analytic_runtime_and_fixed_mask_fd(
    tmp_path: Path, backend: str, shape: tuple
) -> None:
    x = np.full(shape, 0.7)
    y = np.full(shape, 1.3)
    mask = np.arange(x.size).reshape(shape) % 2 == 0
    feeds = {"c": mask, "x": x, "y": y}
    directions = {"x": np.full(shape, 0.2), "y": np.full(shape, -0.3)}
    primal = capture(lambda c, x, y: {"out": xp.where(c, xp.exp(x), xp.log(y))}, feeds)
    with pytest.raises(ValueError, match="Boolean"):
        linearize(primal, directions)
    derivative = generated(primal)
    all_feeds = {**feeds, **{f"d_{k}": v for k, v in directions.items()}}
    # Separate scalar-per-index analytical oracle, independent of IR rules.
    oracle = np.asarray(
        [
            np.exp(float(a)) * float(da) if c else float(db) / float(b)
            for c, a, b, da, db in zip(
                mask.flat,
                x.flat,
                y.flat,
                directions["x"].flat,
                directions["y"].flat,
                strict=True,
            )
        ]
    ).reshape(shape)
    step = 1e-5
    plus = np.where(
        mask, np.exp(x + step * directions["x"]), np.log(y + step * directions["y"])
    )
    minus = np.where(
        mask, np.exp(x - step * directions["x"]), np.log(y - step * directions["y"])
    )
    reference = jvp(primal, feeds, directions).output_tangents["out"]
    for graph in (
        derivative.program,
        optimize(derivative.program),
        Program.loads(derivative.program.dumps()),
    ):
        actual = run(graph, all_feeds, backend, tmp_path)["d_out"]
        np.testing.assert_allclose(actual, oracle, rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(actual, reference, rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(
            actual, (plus - minus) / (2 * step), rtol=2e-7, atol=2e-9
        )
        for value in all_feeds.values():
            assert not np.shares_memory(actual, value)
    changed = {**all_feeds, "c": ~mask}
    expected = np.where(~mask, np.exp(x) * directions["x"], directions["y"] / y)
    np.testing.assert_allclose(
        run(derivative.program, changed, backend, tmp_path)["d_out"], expected
    )
    assert generated(primal).derivative_hash == derivative.derivative_hash
    with pytest.raises(ValueError, match="Boolean"):
        transpose_program(primal, ["out"])


@pytest.mark.parametrize("backend", ["reference", "native"])
@pytest.mark.parametrize(
    "bad", ["sqrt", "log", "exp", "divide", "overflow", "singular", "other_output"]
)
def test_unselected_primal_and_derivative_domains_survive_subset(
    tmp_path: Path, backend: str, bad: str
) -> None:
    feeds = {
        "c": np.ones(3, dtype=bool),
        "x": np.ones(3),
        "y": np.zeros(3) if bad == "singular" else np.ones(3),
    }

    def expression(c: Any, x: Any, y: Any) -> dict:
        branch = {
            "sqrt": lambda: xp.sqrt(-y),
            "log": lambda: xp.log(-y),
            "exp": lambda: xp.exp(y * 1e4),
            "divide": lambda: y / (y - y),
            "overflow": lambda: y * 1e308 * 2,
            "singular": lambda: xp.sqrt(y),
            "other_output": lambda: y,
        }[bad]()
        return {
            "chosen": xp.where(c, x, branch),
            "other": xp.log(-y) if bad == "other_output" else y,
        }

    primal = capture(expression, feeds)
    derivative = generated(primal, ("x",), outputs=("chosen",))
    with pytest.raises(ValueError):
        jvp(primal, feeds, {"x": np.ones(3)}, outputs=("chosen",))
    for graph in (
        derivative.program,
        optimize(derivative.program),
        Program.loads(optimize(derivative.program).dumps()),
    ):
        with pytest.raises(ValueError):
            run(graph, {**feeds, "d_x": np.ones(3)}, backend, tmp_path)


@pytest.mark.parametrize("backend", ["reference", "native"])
def test_broadcast_bool_views_repeated_gather_and_output_selection(
    tmp_path: Path, backend: str
) -> None:
    feeds = {
        "c": np.asarray([[True], [False]]),
        "x": np.asarray([[0.7, 0.9, 1.2]]),
        "y": np.asarray([[1.3], [1.5]]),
    }
    direction = np.asarray([[0.2, -0.3, 0.4]])

    def expression(c: Any, x: Any, y: Any) -> dict:
        mask = xp.take(xp.broadcast_to(c, (2, 3)), (2, 0, 2), axis=1)
        value = xp.take(x, (2, 0, 2), axis=1)
        selected = xp.where(mask, value * value, xp.log(y))
        return {"chosen": xp.sum(selected, axis=1), "alias": selected, "other": x}

    primal = capture(expression, feeds)
    derivative = generated(primal, ("x",), outputs=("chosen", "alias"))
    expected = np.where(
        feeds["c"], 2 * feeds["x"][:, [2, 0, 2]] * direction[:, [2, 0, 2]], 0
    )
    for graph in (
        derivative.program,
        optimize(derivative.program),
        Program.loads(derivative.program.dumps()),
    ):
        actual = run(graph, {**feeds, "d_x": direction}, backend, tmp_path)
        assert set(actual) == {"d_chosen", "d_alias"}
        np.testing.assert_allclose(actual["d_alias"], expected)
        np.testing.assert_allclose(actual["d_chosen"], expected.sum(axis=1))
        reference = jvp(primal, feeds, {"x": direction}, outputs=("chosen", "alias"))
        np.testing.assert_allclose(
            actual["d_alias"], reference.output_tangents["alias"]
        )
        assert not np.shares_memory(actual["d_alias"], actual["d_chosen"])
        if backend == "native":
            with pytest.raises(ValueError):
                actual["d_alias"].flags.writeable = True


@pytest.mark.parametrize("backend", ["reference", "native"])
@pytest.mark.parametrize("empty", [False, True])
def test_guards_zero_before_reduce_and_preserve_empty_demand(
    tmp_path: Path, backend: str, empty: bool
) -> None:
    feeds = {
        "c": np.asarray([True, False, True]),
        "x": np.full(3, 1e308),
        "y": np.full(3, 1e308),
    }
    primal = capture(
        lambda c, x, y: {
            "out": xp.where(c, x, y)[:0] if empty else xp.where(c, x, y),
            "other": y,
        },
        feeds,
    )
    derivative = generated(primal, ("x",), outputs=("out",))
    for graph in (derivative.program, optimize(derivative.program)):
        result = run(graph, {**feeds, "d_x": np.full(3, 1e308)}, backend, tmp_path)[
            "d_out"
        ]
        np.testing.assert_array_equal(
            result, np.empty(0) if empty else [1e308, 0, 1e308]
        )
        with pytest.raises(ValueError):
            run(
                graph,
                {**feeds, "y": np.full(3, np.inf), "d_x": np.ones(3)},
                backend,
                tmp_path,
            )


def test_explicit_domain_and_legacy_controls() -> None:
    feeds = {"c": np.asarray([True, False]), "x": np.ones(2), "y": np.ones(2)}
    primal = capture(lambda c, x, y: {"out": xp.where(c, x, y)}, feeds)
    for option in (1, None, "true"):
        with pytest.raises(TypeError, match="Boolean"):
            linearize(primal, ("x",), allow_external_mask_where=option)
    for dtype in (np.float32,):
        lower = capture(
            lambda c, x, y: {"out": xp.where(c, x, y)},
            {**feeds, "x": feeds["x"].astype(dtype)},
        )
        with pytest.raises(ValueError, match="float64"):
            generated(lower)
    comparison = capture(lambda c, x, y: {"out": xp.where(y > 0, x, y)}, feeds)
    with pytest.raises(ValueError, match="external Boolean"):
        generated(comparison, ("x",))
    with pytest.raises(ValueError, match="tangent input"):
        generated(primal, ("c",))
    derivative = generated(primal)
    with pytest.raises(ValueError, match="higher-order"):
        generated(derivative.program, ("x",))
    with pytest.raises(ValueError, match="precision"):
        generated(
            Program(primal.outputs, primal.definitions, {"precision_execution": {}})
        )
    ordinary = capture(lambda c, x, y: {"out": x * x + y}, feeds)
    for options in (
        {},
        {"allow_external_mask_where": False},
        {"allow_external_mask_where": True},
    ):
        result = linearize(ordinary, ("x",), **options)
        assert result.program.dumps() == linearize(ordinary, ("x",)).program.dumps()
        assert result.generation_version == 3

    def constant_expression(fixed: bool) -> Any:
        return lambda c, x, y: {"out": xp.where(fixed, x, y)}

    for constant_mask in (True, False):
        constant = capture(constant_expression(constant_mask), feeds)
        candidate = generated(constant)
        assert candidate.derivative_hash != derivative.derivative_hash
        if constant_mask:
            true_hash = candidate.derivative_hash
        else:
            assert true_hash != candidate.derivative_hash


def test_generated_names_and_serialized_backend_boundaries() -> None:
    from generativeqc_compiler.tensor.optimize import prepare_for_backend

    feeds = {"c": np.asarray([True]), "x": np.ones(1), "d_x": np.ones(1)}
    primal = capture(lambda c, x, d_x: {"out": xp.where(c, x, d_x)}, feeds, ("x",))
    with pytest.raises(ValueError, match="collides"):
        generated(primal, ("x",))
    primal = capture(lambda c, x, d_x: {"out": xp.where(c, x, d_x)}, feeds, ("d_x",))
    graph = Program.loads(generated(primal, ("d_x",)).program.dumps())
    for backend in ("cuda", "scalar", "portable"):
        with pytest.raises(ValueError):
            prepare_for_backend(graph, backend)


@pytest.mark.parametrize("replay", ["reconstruct", "serialize", "optimize"])
def test_first_order_boundary_survives_provenance_removal(replay: str) -> None:
    feeds = {"c": np.asarray([True]), "x": np.asarray([2.0]), "y": np.asarray([3.0])}
    primal = capture(lambda c, x, y: {"out": xp.where(c, x * y, x + y)}, feeds)
    first = generated(primal, ("x",))
    graph = Program(first.program.outputs)
    if replay == "serialize":
        graph = Program.loads(graph.dumps())
    elif replay == "optimize":
        graph = optimize(graph)
    assert all(not node.spec.differentiable for node in graph.live_nodes)
    np.testing.assert_array_equal(
        execute(graph, {**feeds, "d_x": np.ones(1)}).outputs["d_out"], [3.0]
    )
    for names in (("y",), ("x",), None):
        with pytest.raises(ValueError, match="tangent input|differentiable input"):
            generated(graph, names)
    with pytest.raises(ValueError, match="not declared differentiable"):
        jvp(graph, {**feeds, "d_x": np.ones(1)}, {"y": np.ones(1)})
    for names in (None, ("y",)):
        with pytest.raises(ValueError):
            vjp(
                graph, {**feeds, "d_x": np.ones(1)}, {"d_out": np.ones(1)}, inputs=names
            )
        with pytest.raises(ValueError):
            transpose_program(graph, ("d_out",), inputs=names)
    with pytest.raises(ValueError):
        dot_test(
            graph,
            {**feeds, "d_x": np.ones(1)},
            {"y": np.ones(1)},
            {"d_out": np.ones(1)},
        )


@pytest.mark.parametrize("backend", ["reference", "native"])
def test_nested_selection_transpose_reshape_slice(tmp_path: Path, backend: str) -> None:
    feeds = {
        "c": np.asarray([[True, False, True], [False, True, False]]),
        "x": np.arange(4.0) + 1,
        "y": np.arange(4.0) + 2,
    }

    def expression(c: Any, x: Any, y: Any) -> dict:
        mask = xp.reshape(xp.permute_dims(c, (1, 0)), (6,))[:4]
        return {"out": xp.where(mask, x * x, xp.where(True, xp.log(y), x))}

    primal = capture(expression, feeds)
    graph = generated(primal).program
    seeds = {"d_x": np.full(4, 0.2), "d_y": np.full(4, -0.3)}
    mask = feeds["c"].T.reshape(6)[:4]
    expected = np.where(mask, 2 * feeds["x"] * seeds["d_x"], seeds["d_y"] / feeds["y"])
    for candidate in (graph, optimize(graph), Program.loads(graph.dumps())):
        np.testing.assert_allclose(
            run(candidate, {**feeds, **seeds}, backend, tmp_path)["d_out"], expected
        )


@pytest.mark.parametrize("backend", ["reference", "native"])
def test_unreturned_derivative_overflow_remains_live(
    tmp_path: Path, backend: str
) -> None:
    feeds = {"c": np.asarray([True]), "x": np.ones(1), "y": np.full(1, 1e300)}
    primal = capture(lambda c, x, y: {"out": xp.where(c, x, y), "other": x * y}, feeds)
    direction = np.full(1, 1e100)
    with pytest.raises(ValueError):
        jvp(primal, feeds, {"x": direction}, outputs=("out",))
    graph = generated(primal, ("x",), outputs=("out",)).program
    for candidate in (graph, optimize(graph), Program.loads(graph.dumps())):
        with pytest.raises(ValueError):
            run(candidate, {**feeds, "d_x": direction}, backend, tmp_path)


def test_scientific_annotation_and_generation_budget_reject() -> None:
    from generativeqc_compiler.tensor import Index, IndexSpace, TensorSpec, input_tensor

    feeds = {"c": np.asarray([True, False]), "x": np.ones(2), "y": np.ones(2)}
    primal = capture(lambda c, x, y: {"out": xp.where(c, x, y)}, feeds)
    annotated = input_tensor(
        "z",
        TensorSpec(
            (Index("i", IndexSpace("occ", "occupied", 2)),),
            role="parameter",
            differentiable=True,
        ),
    )
    with pytest.raises(ValueError, match="generic"):
        generated(Program({**primal.outputs, "scientific": annotated}))
    for budget in (-1, True):
        with pytest.raises(ValueError, match="max_elements"):
            generated(primal, max_elements=budget)


@pytest.mark.parametrize("backend", ["reference", "native"])
def test_unreturned_where_domain_with_nonselection_output(
    tmp_path: Path, backend: str
) -> None:
    feeds = {"c": np.asarray([True]), "x": np.ones(1), "y": np.zeros(1)}
    primal = capture(
        lambda c, x, y: {"pure": x, "other": xp.where(c, x, xp.sqrt(y))}, feeds
    )
    # The legacy default still prunes an unrelated selection output.
    execute(linearize(primal, ("x",), outputs=("pure",)).program, {"d_x": np.ones(1)})
    graph = generated(primal, ("x",), outputs=("pure",)).program
    with pytest.raises(ValueError):
        jvp(primal, feeds, {"x": np.ones(1)}, outputs=("pure",))
    for candidate in (graph, optimize(graph), Program.loads(graph.dumps())):
        with pytest.raises(ValueError):
            run(candidate, {**feeds, "d_x": np.ones(1)}, backend, tmp_path)


def test_native_typed_abi_budget_failure_recovery_and_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, record_property: Any
) -> None:
    import ctypes as ct

    from generativeqc.experimental._array_api_execution import _Executor
    from generativeqc_compiler.common.compiler_cache import run_cached_compiler
    from generativeqc_compiler.common.compiler_process import CompileResult
    from generativeqc_compiler.common.cpp_adapter import CppCompilerAdapter
    from generativeqc_compiler.tensor.cpu import describe_cpu_compilation

    feeds = {"c": np.asarray([True, False, True]), "x": np.ones(3), "y": np.ones(3)}
    primal = capture(lambda c, x, y: {"out": xp.where(c, xp.exp(x), xp.sqrt(y))}, feeds)
    graph = generated(primal, ("x",)).program
    all_feeds = {**feeds, "d_x": np.ones(3)}
    _, _, resources, _ = describe_cpu_compilation(graph)
    for options in (
        {"max_bytes": resources["required_bytes"] - 1},
        {"max_work": resources["scalar_work"] - 1},
        {"max_nodes": 1},
    ):
        with pytest.raises(ValueError, match="budget"):
            describe_cpu_compilation(graph, **options)
    owner = _Executor("cpu", {"cache": tmp_path})

    def fail(
        compiler: CppCompilerAdapter, source: Path, output: Path, **options: Any
    ) -> CompileResult:
        # Execute the real cached driver with a deliberately invalid option;
        # restoring the adapter retries the original generated equation.
        return run_cached_compiler(
            [
                str(compiler.cxx),
                "-gqc-invalid-qualification-option",
                str(source),
                "-o",
                str(output),
            ],
            compiler.compile_timeout,
            label="C++ generated JVP failure injection",
        )

    with monkeypatch.context() as patch:
        patch.setattr(CppCompilerAdapter, "compile_shared", fail)
        with pytest.raises(RuntimeError, match="invalid-qualification-option"):
            owner.execute(graph, all_feeds)
    assert not owner.executables and not list(tmp_path.rglob("artifact.json"))
    actual = owner.execute(graph, all_feeds)["d_out"]
    np.testing.assert_allclose(actual, [np.e, 0, np.e])
    native = next(iter(owner.executables.values()))
    # Retain actual execution provenance in CI's JUnit artifact. These are
    # observations, never expected source/body fingerprints or cache-hit claims.
    _, cpp_source, _, _ = describe_cpu_compilation(graph)
    record_property("generated_jvp_logical_hash", graph.logical_hash)
    record_property("cpu_execution_identity", native.identity)
    record_property(
        "cpu_source_sha256", hashlib.sha256(cpp_source.encode()).hexdigest()
    )
    record_property(
        "cpu_library_sha256",
        hashlib.sha256(native.artifact.library.read_bytes()).hexdigest(),
    )
    assert native.resources["typed_bytes"] == 1
    assert 75 <= native.resources["input_count"] <= 80
    for bad in (
        {"c": feeds["c"].astype(np.uint8)},
        {"d_x": np.ones(3, dtype=np.float32)},
    ):
        with pytest.raises(ValueError, match="invalid"):
            native.execute({**all_feeds, **bad})
    packed = np.zeros(native.resources["input_count"], dtype=np.uint8)
    offsets, cursor = {}, 0
    for node in native.inputs:
        value = all_feeds[node.attrs["name"]]
        cursor = (cursor + value.itemsize - 1) // value.itemsize * value.itemsize
        offsets[node.attrs["name"]] = cursor
        packed[cursor : cursor + value.nbytes] = value.view(np.uint8)
        cursor += value.nbytes
    assert cursor == packed.size
    output = np.full(native.resources["output_count"], 77, dtype=np.uint8)
    original = packed.copy()
    pointer = lambda value: value.ctypes.data_as(ct.POINTER(ct.c_ubyte))
    for poison in ("bool", "singular", "nan", "budget", "size", "alias", "alignment"):
        packed[:] = original
        if poison == "bool":
            packed[offsets["c"]] = 2
        elif poison in ("singular", "nan"):
            packed[offsets["y"] : offsets["y"] + 8] = np.asarray(
                [0.0 if poison == "singular" else np.nan]
            ).view(np.uint8)
        target = pointer(packed) if poison == "alias" else pointer(output)
        source = (
            ct.cast(packed.ctypes.data + 1, ct.POINTER(ct.c_ubyte))
            if poison == "alignment"
            else pointer(packed)
        )
        before = packed.copy()
        code = native.call(
            source,
            packed.size - (poison == "size"),
            target,
            output.size,
            native.max_bytes if poison != "budget" else resources["required_bytes"] - 1,
        )
        assert code != 0 and np.all(output == 77), poison
        np.testing.assert_array_equal(packed, before)
    packed[:] = original
    assert (
        native.call(
            pointer(packed), packed.size, pointer(output), output.size, native.max_bytes
        )
        == 0
    )
    np.testing.assert_allclose(output.view(np.float64), [np.e, 0, np.e])
    saved = actual.copy()
    owner.close()
    assert not owner.executables
    np.testing.assert_array_equal(actual, saved)
    with pytest.raises(RuntimeError, match="closed"):
        owner.execute(graph, all_feeds)
