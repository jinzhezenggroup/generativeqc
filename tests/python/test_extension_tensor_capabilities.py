"""Public TensorIR compile-capability contract for #558."""

from __future__ import annotations

import shutil
import subprocess

import generativeqc_compiler.tensor.cpu as tensor_cpu
import pytest
from generativeqc.extensions import tensor
from generativeqc_compiler.common import cpp_adapter
from generativeqc_compiler.common.provenance import canonical_hash
from generativeqc_compiler.tensor.optimize import prepare_for_backend


def _program() -> tensor.Program:
    space = tensor.IndexSpace("ao", "ao", 2)
    index = tensor.Index("i", space)
    node = tensor.input_tensor("x", tensor.TensorSpec((index,), role="input"))
    return tensor.Program({"value": node})


def test_compile_capabilities_reports_lowering_without_toolchain_activation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("capability inspection must not activate a toolchain")

    class ForbiddenCompilerAdapter:
        def __init__(self, *_: object, **__: object) -> None:
            forbidden()

    monkeypatch.setattr(shutil, "which", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(cpp_adapter, "CppCompilerAdapter", ForbiddenCompilerAdapter)
    monkeypatch.setattr(tensor_cpu, "CppCompilerAdapter", ForbiddenCompilerAdapter)

    program = _program()
    report = tensor.compile_capabilities(
        program,
        max_bytes=4096,
        max_work=4096,
        max_nodes=32,
    )
    source, _ = tensor_cpu.emit_cpu(
        program,
        max_bytes=4096,
        max_work=4096,
        max_nodes=32,
        symbol="tensor_cpu",
    )

    assert report["extension_api_version"] == tensor.API_VERSION
    assert report["kind"] == "tensor-compile-capabilities"
    assert report["logical_hash"] == program.logical_hash
    assert report["target"] == "cpu"
    assert report["mode"] == "jit"
    assert report["identity"] == canonical_hash(
        {
            "schema": "generativeqc.tensor.cpu-compilation.v2",
            "logical_hash": prepare_for_backend(program, "cpu").logical_hash,
            "source": source,
        }
    )
    assert report["represented"] is True
    assert report["compilable"] is True
    assert report["lowering_validated"] is True
    assert report["validated"] is False
    assert report["production_promoted"] is False
    assert report["toolchain_checked"] is False
    assert report["reason"] is None
    assert report["resources"]["required_bytes"] > 0
    assert report["resources"]["scalar_work"] > 0


def test_compile_capability_identity_is_deterministic_and_semantic() -> None:
    program = _program()
    first = tensor.compile_capabilities(program)
    replay = tensor.compile_capabilities(program)
    doubled = tensor.Program(
        {"value": tensor.multiply(program.outputs["value"], program.outputs["value"])}
    )
    changed = tensor.compile_capabilities(doubled)

    assert first["identity"] == replay["identity"]
    assert first["identity"] != changed["identity"]



def test_cpu_jit_identity_reuses_equivalent_custom_programs() -> None:
    original = _program()
    custom_a = tensor.Program(
        original.outputs, provenance={"user_label": "first", "origin": {"run": 1}}
    )
    custom_b = tensor.Program(
        original.outputs, provenance={"user_label": "second", "origin": {"run": 2}}
    )

    assert custom_a.logical_hash == custom_b.logical_hash
    assert custom_a.to_payload() != custom_b.to_payload()
    source_a, _ = tensor_cpu.emit_cpu(custom_a)
    source_b, _ = tensor_cpu.emit_cpu(custom_b)
    assert source_a == source_b
    assert tensor.compile_capabilities(custom_a)["identity"] == (
        tensor.compile_capabilities(custom_b)["identity"]
    )


def test_cpu_jit_identity_preserves_diagnostic_pass_selection() -> None:
    original = _program()
    diagnostic = tensor.optimize(original, stop_after="dead_nodes")

    # A pass bisection must keep its own artifact identity even when the
    # current small program happens not to be changed by either pass set.
    assert diagnostic.logical_hash == original.logical_hash
    assert tensor.compile_capabilities(original)["identity"] != (
        tensor.compile_capabilities(diagnostic)["identity"]
    )


def test_cpu_jit_identity_keeps_precision_evidence_isolated() -> None:
    original = _program()
    first = tensor.Program(
        original.outputs, provenance={"precision_source_equation": "source-A"}
    )
    second = tensor.Program(
        original.outputs, provenance={"precision_source_equation": "source-B"}
    )
    assert first.logical_hash == second.logical_hash
    assert tensor.compile_capabilities(first)["identity"] != (
        tensor.compile_capabilities(second)["identity"]
    )


@pytest.mark.parametrize(
    ("target", "mode", "message"),
    [
        ("cpu", "aot", "mode='jit'"),
        ("cuda", "jit", "target='cpu'"),
    ],
)
def test_compile_capabilities_reports_unsupported_target_or_mode(
    target: str, mode: str, message: str
) -> None:
    report = tensor.compile_capabilities(_program(), target=target, mode=mode)

    assert report["represented"] is True
    assert report["compilable"] is False
    assert report["lowering_validated"] is False
    assert report["validated"] is False
    assert report["production_promoted"] is False
    assert report["toolchain_checked"] is False
    assert report["identity"] is None
    assert report["resources"] is None
    assert message in report["reason"]


def test_compile_capabilities_reports_unsupported_ir_without_promotion() -> None:
    program = _program()
    unsupported = tensor.Program({"value": tensor.exp(program.outputs["value"])})

    report = tensor.compile_capabilities(unsupported)

    assert report["represented"] is True
    assert report["compilable"] is False
    assert report["lowering_validated"] is False
    assert report["validated"] is False
    assert report["production_promoted"] is False
    assert report["identity"] is None
    assert report["resources"] is None
    assert "unsupported CPU primitive" in report["reason"]


def test_compile_capabilities_applies_requested_resource_bounds() -> None:
    report = tensor.compile_capabilities(_program(), max_bytes=0)

    assert report["represented"] is True
    assert report["compilable"] is False
    assert report["identity"] is None
    assert report["resources"] is None
    assert "budget" in report["reason"]


def test_compile_capabilities_rejects_non_program() -> None:
    with pytest.raises(TypeError, match="capability query requires Program"):
        tensor.compile_capabilities(object())  # type: ignore[arg-type]
