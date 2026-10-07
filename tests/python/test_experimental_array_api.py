"""Experimental public Array API facade over the canonical TensorIR frontend."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from fractions import Fraction

from generativeqc.experimental import API_VERSION as EXPERIMENTAL_API_VERSION
from generativeqc.experimental import array_api as xp
from generativeqc_compiler.array_api import namespace as compiler_xp
from generativeqc_compiler.array_api import trace as compiler_trace


def _vector_spec() -> xp.TensorSpec:
    ao = xp.IndexSpace("ao", "ao", 3)
    return xp.TensorSpec((xp.Index("p", ao),), role="input")


def test_experimental_package_keeps_array_surface_lazy() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import sys
                import generativeqc.experimental as experimental

                assert experimental.API_VERSION == 1
                assert experimental.__all__ == ["API_VERSION", "array_api"]
                assert "array_api" in dir(experimental)
                assert "generativeqc.experimental.array_api" not in sys.modules
                assert not any(
                    name.startswith("generativeqc_compiler.array_api")
                    for name in sys.modules
                )
                """
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


def test_public_preview_reuses_canonical_tensorir_identity() -> None:
    spec = _vector_spec()
    public = xp.trace(
        lambda x: {"out": xp.sum(Fraction(1, 2) * x * x)},
        {"x": spec},
    )
    internal = compiler_trace(
        lambda x: {"out": compiler_xp.sum(Fraction(1, 2) * x * x)},
        {"x": spec},
    )

    assert isinstance(public, xp.Program)
    assert public.logical_hash == internal.logical_hash


def test_public_preview_declares_experimental_nonconformance() -> None:
    assert EXPERIMENTAL_API_VERSION == 1
    assert xp.API_VERSION == 1
    report = xp.capabilities()
    assert report["public_api_version"] == 1
    assert report["surface"] == "array-api-shaped-experimental-public-preview"
    assert report["stability"] == "experimental"
    assert report["import_path"] == "generativeqc.experimental.array_api"
    assert report["array_api_version"] is None
    assert report["array_namespace_protocol"] is False

    value = xp.input_array("x", _vector_spec())
    assert isinstance(value, xp.VibeArray)
    assert not hasattr(value, "__array_namespace__")


def test_public_preview_exports_declared_symbolic_operations() -> None:
    for name in (
        "add",
        "broadcast_to",
        "divide",
        "einsum",
        "exp",
        "log",
        "matmul",
        "multiply",
        "negative",
        "permute_dims",
        "pow",
        "reshape",
        "slice",
        "sqrt",
        "subtract",
        "sum",
        "take",
    ):
        assert callable(getattr(xp, name))
