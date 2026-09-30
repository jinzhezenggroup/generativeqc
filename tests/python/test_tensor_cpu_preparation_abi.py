"""CPU packing and publication must use the exact graph passed to emission."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from generativeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    add,
    cpu,
    einsum,
    input_tensor,
)
from generativeqc_compiler.tensor.optimize import prepare_for_backend


def _inputs() -> tuple:
    spaces = {
        name: IndexSpace(name, "ao", size)
        for name, size in (("i", 2), ("j", 3), ("k", 10), ("l", 20))
    }
    return tuple(
        input_tensor(
            name,
            TensorSpec(tuple(Index(axis, spaces[axis]) for axis in axes), role="input"),
        )
        for name, axes in (("a", "ij"), ("b", "kl"), ("c", "jk"))
    )


def _input_names(program: Program) -> tuple[str, ...]:
    return tuple(
        node.attrs["name"] for node in program.live_nodes if node.op == "input"
    )


@pytest.mark.parametrize("seed", [2, 7])
def test_native_cpu_reassociation_packs_lowered_input_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seed: int
) -> None:
    a, b, c = _inputs()
    original = Program({"out": einsum("ij,kl,jk->il", a, b, c)})
    preparations = []

    def prepare(program: Program, backend: str) -> Program:
        prepared = prepare_for_backend(program, backend)
        preparations.append(prepared)
        return prepared

    monkeypatch.setattr(cpu, "prepare_for_backend", prepare)
    native = cpu.NativeTensorProgram(
        original, compiler=CppCompilerAdapter(Path("c++")), cache=tmp_path
    )
    assert len(preparations) == 1
    assert native.program is preparations[0]
    # Require a real binary-tree rewrite. Canonical content-hash ordering may
    # preserve the input order even though A*C is the cheaper first contraction.
    contractions = [node for node in native.program.live_nodes if node.op == "einsum"]
    assert len(contractions) == 2
    assert all(len(node.inputs) == 2 for node in contractions)
    assert native.program.logical_hash != original.logical_hash
    assert tuple(node.attrs["name"] for node in native.inputs) == _input_names(
        native.program
    )
    assert native.resources["input_count"] == sum(
        node.spec.size for node in native.inputs
    )

    random = np.random.default_rng(seed)
    feeds = {
        "a": random.normal(size=(2, 3)),
        "b": random.normal(size=(10, 20)),
        "c": random.normal(size=(3, 10)),
    }
    expected = np.einsum(
        "ij,kl,jk->il", feeds["a"], feeds["b"], feeds["c"], optimize=False
    )
    np.testing.assert_allclose(
        native.execute(feeds)["out"], expected, atol=1e-12, rtol=1e-12
    )


def test_native_cpu_projected_graph_owns_input_count_and_output_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    a, _, _ = _inputs()
    b = input_tensor("other", a.spec)
    original = Program({"sum": add(a, b), "kept": a})
    preparations = []

    def prepare(program: Program, backend: str) -> Program:
        # Use the real shared output-demand pass to exercise a smaller ABI.
        prepared = prepare_for_backend(program, backend, requested_outputs=("kept",))
        preparations.append(prepared)
        return prepared

    monkeypatch.setattr(cpu, "prepare_for_backend", prepare)
    native = cpu.NativeTensorProgram(
        original, compiler=CppCompilerAdapter(Path("c++")), cache=tmp_path
    )
    assert len(preparations) == 1
    assert native.program is preparations[0]
    assert _input_names(native.program) == ("a",)
    assert native.resources["input_count"] == a.spec.size
    values = np.arange(6, dtype=np.float64).reshape(2, 3)
    result = native.execute({"a": values})
    assert set(result) == {"kept"}
    np.testing.assert_array_equal(result["kept"], values)


def test_native_cpu_admission_still_precedes_compiler_discovery(tmp_path: Path) -> None:
    a, b, c = _inputs()
    original = Program({"out": einsum("ij,kl,jk->il", a, b, c)})
    discovered = []

    def compiler() -> CppCompilerAdapter:
        discovered.append(True)
        return CppCompilerAdapter(Path("c++"))

    with pytest.raises(ValueError, match="byte/work budget"):
        cpu.NativeTensorProgram(
            original, compiler=compiler, cache=tmp_path, max_bytes=1
        )
    assert discovered == []
