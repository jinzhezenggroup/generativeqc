"""Cheap control-flow guard for the heavy-ECP geometry restoration gate."""

import ast
import typing
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize("ignore_coordinates", [False, True])
def test_heavy_replay_detects_ignored_geometry(ignore_coordinates: bool) -> None:
    """Exercise the real test body without repeating a heavy SCF calculation."""
    path = Path(__file__).with_name("test_ecp_heavy.py")
    tree = ast.parse(path.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "test_heavy_resource_boundary_replay"
    )
    function.decorator_list = []
    coordinates = np.array([[0.13, -0.21, 0.17], [0.43, 0.19, 4.4]])
    calls = []

    class Batch:
        def __enter__(self) -> typing.Self:
            return self

        def __exit__(self, *args: object) -> bool:
            return False

        def execute(
            self, coordinates: typing.Any = None, strict: bool = False
        ) -> SimpleNamespace:
            assert strict
            delta = 0.0 if coordinates is None else float(coordinates[0][1, 2] - 4.4)
            calls.append(delta)
            response = 0.0 if ignore_coordinates else delta
            item = SimpleNamespace(
                converged=True,
                energy=-1.0 + response,
                forces=np.array([[0.0, 0.0, response], [0.0, 0.0, -response]]),
            )
            return SimpleNamespace(items=[item])

    class Calculator:
        def __init__(self, **kwargs: typing.Any) -> None:
            pass

        def estimate_resources(
            self, *args: typing.Any, **kwargs: typing.Any
        ) -> SimpleNamespace:
            return SimpleNamespace(
                require_feasible=lambda: None, peak_bytes={"host": 1024}
            )

        def prepare_batch(self, *args: typing.Any, **kwargs: typing.Any) -> Batch:
            return Batch()

    namespace = {
        "__name__": __name__,
        "typing": typing,
        "np": np,
        "Calculator": Calculator,
        "ResourceBudget": lambda **kwargs: kwargs,
        "require_device": lambda device: None,
        "scf_options": lambda symbol: {},
        "heavy_fixture": lambda symbol, spin: (
            [],
            None,
            SimpleNamespace(charge=0, atom_coords=lambda: coordinates.copy()),
        ),
    }
    code = compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec")
    exec(code, namespace)  # noqa: S102 - execute only the checked-in test function
    replay = namespace[function.name]
    if ignore_coordinates:
        with pytest.raises(AssertionError):
            replay("Rb", 0, "cpu")
    else:
        replay("Rb", 0, "cpu")
        assert calls == pytest.approx([0.0, 0.02, 0.0])
