"""Real provider admission, source identity and production history codegen graph."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from generativeqc_compiler.common.provenance import canonical_hash
from generativeqc_compiler.method.gfn2_history_lowering import (
    emit_gfn2_history_artifacts,
    gfn2_history_gram_bindings,
)
from generativeqc_compiler.tensor.broyden_cpu_lowering import emit_broyden_cpu_artifacts
from generativeqc_compiler.tensor.ordered_history import (
    ordered_history_candidate,
    ordered_history_program,
    require_history_candidate,
    retained_history_schedule,
)
from generativeqc_compiler.tensor.ordered_history_gram import (
    emit_history_gram,
    emit_history_window,
)

ROOT = Path(__file__).resolve().parents[2]


def test_candidates_bind_actual_scalar_graphs_layout_and_execution() -> None:
    p = ordered_history_program()
    for name, backend, ld in (("compact-cpu", "cpu", 2), ("capacity-cuda", "cuda", 4)):
        candidate = ordered_history_candidate(p, name)
        request, execution = candidate.request, candidate.execution
        assert execution is not None
        assert request.scientific_identity == p.identity
        assert execution.algorithm == name and execution.layouts == request.operands
        assert candidate.providers[0].name == "generated." + backend
        assert (
            candidate.workspace_bytes
            == candidate.provider_bytes
            == execution.temporary_bytes
            == 0
        )
        assert request.constraints.determinism == execution.determinism == "exact-order"
        assert execution.capture_safe == (backend == "cuda")
        assert request.precisions[0].schedule.is_strict_fp64
        assert (
            dict(request.effects)["arithmetic"]
            == "fp64-native-expression-retained-contraction-policy"
        )
        assert dict(request.semantics)["stage_graphs"] == canonical_hash(
            [(n, g.logical_hash) for n, g in p.stages]
        )
        assert next(op for op in request.operands if op.operand == "beta").strides == (
            ld,
            1,
        )
        assert all(
            op.access == "read"
            for op in request.operands
            if op.operand in ("df_history", "u_history", "weights")
        )
        assert next(op for op in request.operands if op.operand == "beta").modes == (
            1,
            3,
        )
        assert require_history_candidate(p, candidate).name == name


@pytest.mark.parametrize(
    "field,value",
    (
        ("vector_size", 0),
        ("history_count", 0),
        ("capacity", 0),
        ("history_count", 5),
        ("vector_size", True),
        ("capacity", -1),
    ),
)
def test_candidate_admission_rejects_invalid_physical_domains(
    field: str, value: int
) -> None:
    with pytest.raises(ValueError, match="positive"):
        ordered_history_candidate(
            ordered_history_program(), "compact-cpu", **{field: value}
        )


@pytest.mark.parametrize(
    "mutation",
    ("status", "algorithm", "strides", "publication", "provider", "precision"),
)
def test_forged_selected_candidate_cannot_keep_an_unrelated_body(mutation: str) -> None:
    p = ordered_history_program()
    candidate = ordered_history_candidate(p, "capacity-cuda")
    execution = candidate.execution
    assert execution is not None
    if mutation == "status":
        candidate = replace(candidate, status="unsupported", reason="unqualified")
    elif mutation == "algorithm":
        candidate = replace(
            candidate, execution=replace(execution, algorithm="compact-cpu")
        )
    elif mutation == "strides":
        layouts = tuple(
            replace(op, strides=(2, 1)) if op.operand == "beta" else op
            for op in execution.layouts
        )
        candidate = replace(
            candidate,
            request=replace(candidate.request, operands=layouts),
            execution=replace(execution, layouts=layouts),
        )
    elif mutation == "publication":
        candidate = replace(
            candidate,
            request=replace(
                candidate.request, effects=(("publication", "persistent-early"),)
            ),
        )
    elif mutation == "provider":
        candidate = replace(
            candidate, providers=(replace(candidate.providers[0], name="cublas"),)
        )
    else:
        candidate = replace(candidate, numerical_mode="forced-unfused")
    with pytest.raises(ValueError, match="admitted"):
        emit_gfn2_history_artifacts("cuda", candidate=candidate)


@pytest.mark.parametrize("backend", ("cpu", "cuda"))
def test_native_generation_uses_the_candidate_selected_by_the_provider_boundary(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
) -> None:
    from generativeqc_compiler.tensor import ordered_history_artifacts as shared

    bad = replace(
        ordered_history_candidate(ordered_history_program(), "compact-cpu"),
        status="unsupported",
        reason="not admitted",
    )
    monkeypatch.setattr(shared, "ordered_history_candidate", lambda *_: bad)
    with pytest.raises(ValueError, match="admitted"):
        if backend == "cpu":
            emit_broyden_cpu_artifacts()
        else:
            emit_gfn2_history_artifacts("cuda")


def test_candidate_backend_and_extent_binding_cannot_be_silently_reinterpreted() -> (
    None
):
    p = ordered_history_program()
    with pytest.raises(ValueError, match="backend"):
        emit_broyden_cpu_artifacts(
            candidate=ordered_history_candidate(p, "capacity-cuda")
        )
    with pytest.raises(ValueError, match="backend"):
        emit_gfn2_history_artifacts(
            "cuda", candidate=ordered_history_candidate(p, "compact-cpu")
        )
    for backend, name in (("cpu", "compact-cpu"), ("cuda", "capacity-cuda")):
        emit = (
            emit_broyden_cpu_artifacts
            if backend == "cpu"
            else lambda **kwargs: emit_gfn2_history_artifacts("cuda", **kwargs)
        )
        canonical = emit()
        actual = emit(
            candidate=ordered_history_candidate(
                p, name, vector_size=137, history_count=65, capacity=65
            ),
        )
        assert all(
            actual[key] == body
            for key, body in canonical.items()
            if key.endswith(".inc")
        )
        identity = next(key for key in canonical if key.endswith("_identity.json"))
        assert actual[identity] != canonical[identity]


@pytest.mark.parametrize(
    "field",
    (
        "delta_f",
        "new_u",
        "residual",
        "df_history",
        "u_history",
        "weights",
        "new_weight",
        "capacity",
        "coefficients",
        "beta",
        "omega_zero",
    ),
)
def test_full_gram_physical_bindings_are_not_raw_code(field: str) -> None:
    p = ordered_history_program()
    bad = replace(
        gfn2_history_gram_bindings("cuda"), **{field: "a[(side_effect(), i)]"}
    )
    schedule = retained_history_schedule(p, "capacity-cuda")
    for emit in (emit_history_gram, emit_history_window):
        with pytest.raises(ValueError, match="declared"):
            emit(p, schedule, bad)


@pytest.mark.parametrize(
    "field",
    (
        "coefficient_dot_failure",
        "coefficient_product_failure",
        "overlap_failure",
        "matrix_failure",
        "weight_failure",
    ),
)
def test_full_gram_cannot_omit_a_finite_failure(field: str) -> None:
    p = ordered_history_program()
    bad = replace(gfn2_history_gram_bindings("cuda"), **{field: ()})
    with pytest.raises(ValueError, match="failure effect"):
        emit_history_gram(p, retained_history_schedule(p, "capacity-cuda"), bad)


def test_cli_emits_consumed_fragments_with_complete_source_identity(
    tmp_path: Path,
) -> None:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/generate_ordered_history_native.py"),
            "--output-directory",
            str(tmp_path),
        ],
        check=True,
        timeout=30,
    )
    assert len(list(tmp_path.iterdir())) == 10
    for prefix in ("generated_broyden_cpu", "generated_gfn2_history_cuda"):
        identity = json.loads((tmp_path / f"{prefix}_identity.json").read_text())
        digest = identity.pop("source_identity")
        assert digest == canonical_hash(identity)
        assert identity["candidate"]["status"] == "ready"
        assert (
            identity["candidate"]["request"]["scientific_identity"]
            == identity["scientific_identity"]
        )
        assert len(identity["sources"]) == 4
        for name, expected in identity["sources"].items():
            assert (
                hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == expected
            )


def test_production_build_registers_every_fragment_and_direct_compiler_dependency() -> (
    None
):
    source = (ROOT / "cmake/GenerativeQCGeneratedSources.cmake").read_text()
    assert "foreach(_phase helpers window gram correction)" in source
    assert "generated_broyden_cpu_identity.json" in source
    assert "generated_gfn2_history_cuda_identity.json" in source
    cpu = source.split("NAME generativeqc_broyden_cpu_codegen", 1)[1].split(
        "if(TARGET generativeqc_gfn2_cuda)", 1
    )[0]
    assert "--backend cpu" in cpu and "method/" not in cpu
    assert "tensor/broyden_cpu_lowering.py" in cpu
    assert (
        "add_dependencies(generativeqc_gfn2_cuda generativeqc_ordered_history_codegen)"
        in source
    )
    for name in (
        "ordered_history.py",
        "ordered_history_emit.py",
        "ordered_history_artifacts.py",
        "broyden_cpu_lowering.py",
        "ordered_history_gram.py",
        "scalar_cpp.py",
        "gfn2_history_lowering.py",
        "lowering_provider.py",
        "lowering_contract.py",
    ):
        assert name in source


def test_generator_is_in_build_identity_and_records_real_import_dependencies(
    tmp_path: Path,
) -> None:
    inventory = json.loads((ROOT / "cmake/GenerativeQCSourceIdentity.json").read_text())
    assert "tools/generate_ordered_history_native.py" in inventory["files"]
    target = tmp_path / "generated_broyden_cpu_helpers.inc"
    depfile = tmp_path / "history.d"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/run_codegen.py"),
            "--depfile",
            str(depfile),
            "--target",
            str(target),
            "--source-root",
            str(ROOT),
            str(ROOT / "tools/generate_ordered_history_native.py"),
            "--output-directory",
            str(tmp_path),
            "--backend",
            "cpu",
        ],
        check=True,
        timeout=30,
    )
    assert target.is_file()
    dependencies = depfile.read_text()
    for name in (
        "tensor/broyden_cpu_lowering.py",
        "tensor/ordered_history_artifacts.py",
        "tensor/ordered_history.py",
        "tensor/ordered_history_emit.py",
        "tensor/ordered_history_gram.py",
        "tensor/scalar_cpp.py",
        "tensor/ir.py",
        "tensor/program.py",
        "common/lowering_provider.py",
        "common/lowering_contract.py",
    ):
        assert name in dependencies
    assert "generativeqc_compiler/method/" not in dependencies


def test_cpu_fragments_keep_preextraction_mathematics_and_candidate_identity() -> None:
    baseline = ROOT / "tests/native/fixtures/johnson_prechange_3b97c234"
    artifacts = emit_broyden_cpu_artifacts()
    for name, body in artifacts.items():
        if name.endswith(".inc"):
            old_name = name.replace(
                "generated_broyden_cpu", "generated_gfn2_history_cpu"
            )
            assert body == (baseline / old_name).read_text()
    old = json.loads(
        (baseline / "generated_gfn2_history_cpu_identity.json").read_text()
    )
    current = json.loads(artifacts["generated_broyden_cpu_identity.json"])
    assert current["scientific_identity"] == old["scientific_identity"]
    assert current["candidate"] == old["candidate"]
    assert current["sources"] == {
        name.replace("generated_gfn2_history_cpu", "generated_broyden_cpu"): digest
        for name, digest in old["sources"].items()
    }


def test_cuda_artifact_names_bodies_and_diagnostic_identity() -> None:
    # Source identity includes every emitted filename and fragment digest.
    artifacts = emit_gfn2_history_artifacts("cuda")
    identity = json.loads(artifacts["generated_gfn2_history_cuda_identity.json"])
    digest = identity.pop("source_identity")
    assert digest == "8592b95f849192ab17f614af64178651b92b25e5d1b4ca5d2e583378ff594784"
    assert canonical_hash(identity) == digest
    for name, expected in identity["sources"].items():
        assert hashlib.sha256(artifacts[name].encode()).hexdigest() == expected


def test_cpu_generation_imports_no_method_or_runtime_modules(tmp_path: Path) -> None:
    script = """
import importlib.abc
import runpy
import sys

class RejectMethodRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "generativeqc_compiler.method" or fullname.split(".")[0] in {
            "generativeqc", "pyscf", "torch", "cupy"
        }:
            raise AssertionError("CPU generator imported " + fullname)

sys.meta_path.insert(0, RejectMethodRuntime())
sys.argv = [sys.argv[1], "--backend", "cpu", "--output-directory", sys.argv[2]]
runpy.run_path(sys.argv[0], run_name="__main__")
"""
    subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(ROOT / "tools/generate_ordered_history_native.py"),
            str(tmp_path),
        ],
        check=True,
        timeout=30,
    )
    assert {path.name for path in tmp_path.iterdir()} == set(
        emit_broyden_cpu_artifacts()
    )


@pytest.mark.parametrize("with_cuda", (False, True))
def test_production_codegen_targets_separate_cpu_method_dependencies(
    tmp_path: Path,
    with_cuda: bool,
) -> None:
    cmake = shutil.which("cmake")
    cache = shutil.which(os.environ.get("CCACHE", "ccache"))
    generator = "Ninja" if shutil.which("ninja") else "Unix Makefiles"
    if (
        cmake is None
        or cache is None
        or (generator == "Unix Makefiles" and shutil.which("make") is None)
    ):
        pytest.skip("CMake, a build tool and a verified compiler cache are required")
    subprocess.run([cache, "--version"], check=True, capture_output=True, timeout=10)
    for name in ("tools", "python", "src", "include", "manifests", "upstream", "data"):
        (tmp_path / name).symlink_to(ROOT / name, target_is_directory=True)
    (tmp_path / "dummy.cpp").write_text("int fixture;\n")
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.24)\n"
        "project(HistoryCodegenDependencies LANGUAGES CXX)\n"
        f'include("{ROOT / "cmake/GenerativeQCGenerated.cmake"}")\n'
        f'include("{ROOT / "cmake/GenerativeQCGeneratedSources.cmake"}")\n'
        f'set(Python3_EXECUTABLE "{sys.executable}")\n'
        # Imported modules resolve symlinks; their identity root must match.
        f'set(PROJECT_SOURCE_DIR "{ROOT}")\n'
        "add_library(generativeqc STATIC dummy.cpp)\n"
        + (
            "add_library(generativeqc_gfn2_cuda STATIC dummy.cpp)\n"
            if with_cuda
            else ""
        )
        + "generativeqc_register_host_generated_sources(generativeqc)\n"
    )
    build = tmp_path / "build"
    subprocess.run(
        [
            cmake,
            "-S",
            str(tmp_path),
            "-B",
            str(build),
            "-G",
            generator,
            "-DCMAKE_CXX_COMPILER_LAUNCHER=" + cache,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    targets = ["generativeqc_broyden_cpu_codegen"]
    if with_cuda:
        targets.append("generativeqc_ordered_history_codegen")
    subprocess.run(
        [cmake, "--build", str(build), "--target", *targets],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    generated = build / "generated"
    expected = emit_broyden_cpu_artifacts()
    if with_cuda:
        expected.update(emit_gfn2_history_artifacts("cuda"))
    assert {path.name for path in generated.iterdir() if path.suffix != ".d"} == set(
        expected
    )
    for name, body in expected.items():
        assert (generated / name).read_text() == body
    dependencies = (generated / "generated_broyden_cpu_helpers.inc.d").read_text()
    assert "tensor/broyden_cpu_lowering.py" in dependencies
    assert "tensor/ordered_history_artifacts.py" in dependencies
    assert "generativeqc_compiler/method/" not in dependencies
    if with_cuda:
        dependencies = (
            generated / "generated_gfn2_history_cuda_helpers.inc.d"
        ).read_text()
        assert "method/gfn2_history_lowering.py" in dependencies
        assert "tensor/broyden_cpu_lowering.py" not in dependencies
