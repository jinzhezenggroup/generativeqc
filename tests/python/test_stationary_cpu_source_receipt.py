"""Same-build source reuse retains generation and fail-closed audit contracts."""

from __future__ import annotations

import json
import os
import runpy
import sys
from typing import TYPE_CHECKING, Any

import pytest

from tools import audit_derivative_aot_registry as audit
from tools import generate_stationary_cpu_derivative_aot as generator
from tools import stationary_cpu_source_receipt as receipt

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


@pytest.fixture
def source_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "source"
    for relative in (
        "python/generativeqc_compiler/__init__.py",
        "python/generativeqc_compiler/integral/expr.py",
        "python/generativeqc_compiler/integral/nested/leaf.json",
        "tools/generate_stationary_cpu_derivative_aot.py",
        "tools/stationary_cpu_source_receipt.py",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("input-a\n", encoding="utf-8")
    monkeypatch.setattr(receipt, "ROOT", root)
    return root


@pytest.fixture
def generated(
    source_tree: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, tuple[str, ...]]:
    directory = tmp_path / "generated"
    sources = tuple(f"// emitted shard {i}: λ\n" for i in range(receipt.CPU_AOT_SHARDS))
    monkeypatch.setattr(
        generator, "derivative_cpu_aot_sources", lambda: tuple(((), s) for s in sources)
    )
    monkeypatch.setattr(sys, "argv", ["generate", "--output-directory", str(directory)])
    generator.main()
    return directory, sources


def test_exact_generated_sources_replay_and_preserve_output_times(
    generated: tuple[Path, tuple[str, ...]],
) -> None:
    directory, sources = generated
    paths = sorted(directory.glob("*.cpp"))
    times = {path: path.stat().st_mtime_ns for path in paths}
    assert receipt.load(directory) == sources
    generator.main()
    assert {path: path.stat().st_mtime_ns for path in paths} == times
    assert receipt.load(directory) == sources


def test_existing_crlf_outputs_are_normalized_before_receipt_publication(
    generated: tuple[Path, tuple[str, ...]],
) -> None:
    directory, sources = generated
    for path in directory.glob("*.cpp"):
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    generator.main()
    for shard, source in enumerate(sources):
        assert (
            directory / f"generativeqc_stationary_cpu_derivative_{shard}.cpp"
        ).read_bytes() == source.encode("utf-8")
    assert receipt.load(directory) == sources
    times = {path: path.stat().st_mtime_ns for path in directory.glob("*.cpp")}
    generator.main()
    assert {path: path.stat().st_mtime_ns for path in directory.glob("*.cpp")} == times


@pytest.mark.parametrize("missing", ["none", "directory", "receipt"])
def test_missing_handoff_retains_generation(
    generated: tuple[Path, tuple[str, ...]], missing: str
) -> None:
    directory, _ = generated
    if missing == "receipt":
        (directory / receipt.RECEIPT).unlink()
    assert (
        receipt.load(
            None
            if missing == "none"
            else directory / "absent"
            if missing == "directory"
            else directory
        )
        is None
    )


@pytest.mark.parametrize("change", ["rewrite", "add", "delete", "generator", "json"])
def test_current_dependency_changes_invalidate_even_same_size_and_mtime(
    generated: tuple[Path, tuple[str, ...]], source_tree: Path, change: str
) -> None:
    directory, _ = generated
    relative = {
        "generator": "tools/generate_stationary_cpu_derivative_aot.py",
        "json": "python/generativeqc_compiler/integral/nested/leaf.json",
    }.get(change, "python/generativeqc_compiler/integral/expr.py")
    path = source_tree / relative
    if change == "add":
        path.with_name("added.py").write_text("new dependency\n")
    elif change == "delete":
        path.unlink()
    else:
        stamp = path.stat()
        path.write_text("input-b\n")
        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        assert path.stat().st_size == stamp.st_size
        assert path.stat().st_mtime_ns == stamp.st_mtime_ns
    assert receipt.load(directory) is None


@pytest.mark.parametrize("change", ["missing", "tamper", "truncate", "swap", "symlink"])
def test_current_receipt_rejects_changed_generated_outputs(
    generated: tuple[Path, tuple[str, ...]], change: str
) -> None:
    directory, _ = generated
    path = directory / "generativeqc_stationary_cpu_derivative_0.cpp"
    if change == "missing":
        path.unlink()
    elif change == "tamper":
        stamp = path.stat()
        path.write_text(path.read_text().replace("shard", "wrong"))
        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    elif change == "truncate":
        path.write_bytes(path.read_bytes()[:-1])
    elif change == "swap":
        other = directory / "generativeqc_stationary_cpu_derivative_1.cpp"
        left, right = path.read_bytes(), other.read_bytes()
        path.write_bytes(right)
        other.write_bytes(left)
    else:
        target = directory / "target"
        path.rename(target)
        path.symlink_to(target)
    with pytest.raises(ValueError, match="integrity"):
        receipt.load(directory)


@pytest.mark.parametrize(
    "change",
    ["schema", "json", "order", "count", "hash", "size", "identity", "symlink"],
)
def test_corrupt_receipt_never_supplies_unverified_source(
    generated: tuple[Path, tuple[str, ...]], change: str
) -> None:
    directory, _ = generated
    path = directory / receipt.RECEIPT
    metadata = json.loads(path.read_text())
    if change == "schema":
        metadata["schema"] = "unsupported"
    elif change == "order":
        metadata["sources"].reverse()
    elif change == "count":
        metadata["sources"].pop()
    elif change == "hash":
        metadata["sources"][0]["sha256"] = "0" * 64
    elif change == "size":
        metadata["sources"][0]["bytes"] = True
    elif change == "identity":
        metadata["identity"] = None
    if change == "symlink":
        target = directory / "receipt-target"
        path.rename(target)
        path.symlink_to(target)
    else:
        path.write_text("{" if change == "json" else json.dumps(metadata))
    with pytest.raises(ValueError, match="integrity"):
        receipt.load(directory)


def test_stale_identity_retains_generation(
    generated: tuple[Path, tuple[str, ...]],
) -> None:
    directory, _ = generated
    path = directory / receipt.RECEIPT
    metadata = json.loads(path.read_text())
    metadata["identity"]["python"]["version"][0] += 1
    path.write_text(json.dumps(metadata))
    assert receipt.load(directory) is None


@pytest.mark.parametrize("failure", ["lowering", "writing", "input_drift"])
def test_interrupted_generation_never_publishes_a_new_receipt(
    source_tree: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    directory = tmp_path / "partial"
    sources = tuple(((), f"// unit {i}\n") for i in range(receipt.CPU_AOT_SHARDS))

    def lower() -> tuple[tuple[tuple[()], str], ...]:
        if failure == "lowering":
            raise RuntimeError("injected lowering failure")
        if failure == "input_drift":
            (source_tree / "python/generativeqc_compiler/integral/expr.py").write_text(
                "new input\n"
            )
        return sources

    writer = generator._write_if_changed

    def write(path: Path, source: str) -> None:
        if failure == "writing" and path.name.endswith("_1.cpp"):
            raise RuntimeError("injected output failure")
        writer(path, source)

    monkeypatch.setattr(generator, "derivative_cpu_aot_sources", lower)
    monkeypatch.setattr(generator, "_write_if_changed", write)
    monkeypatch.setattr(sys, "argv", ["generate", "--output-directory", str(directory)])
    with pytest.raises((RuntimeError, ValueError)):
        generator.main()
    assert not (directory / receipt.RECEIPT).exists()
    assert receipt.load(directory) is None


def _light_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Callable[..., dict[str, object]]:
    manifest = tmp_path / "radials.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "generativeqc.derivative-aot.radials.v1",
                "entries": [{"backend": "cpu", "family": "short_range", "omega": 0.3}],
            }
        )
    )
    monkeypatch.setattr(audit, "_range_programs", lambda radials: ([], ("range",)))
    return lambda **kwargs: audit.audit(radial_manifest=manifest, **kwargs)


def test_audit_cached_and_uncached_outputs_are_identical(
    generated: tuple[Path, tuple[str, ...]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory, sources = generated
    run = _light_audit(tmp_path, monkeypatch)
    monkeypatch.setattr(
        audit, "derivative_cpu_aot_sources", lambda: tuple(((), s) for s in sources)
    )
    baseline = run()

    def forbidden() -> None:
        raise AssertionError("verified handoff unnecessarily regenerated CPU sources")

    monkeypatch.setattr(audit, "derivative_cpu_aot_sources", forbidden)
    assert run(cpu_source_directory=directory) == baseline


@pytest.mark.parametrize("condition", ["absent", "stale"])
def test_audit_fallback_still_exposes_a_real_cpu_generation_defect(
    generated: tuple[Path, tuple[str, ...]],
    source_tree: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    condition: str,
) -> None:
    directory, _ = generated
    run = _light_audit(tmp_path, monkeypatch)
    if condition == "absent":
        (directory / receipt.RECEIPT).unlink()
    else:
        (source_tree / "python/generativeqc_compiler/integral/expr.py").write_text(
            "defect\n"
        )

    def defective() -> None:
        raise RuntimeError("injected CPU generation defect")

    monkeypatch.setattr(audit, "derivative_cpu_aot_sources", defective)
    with pytest.raises(RuntimeError, match="CPU generation defect"):
        run(cpu_source_directory=directory)


@pytest.mark.parametrize("mutation", ["target", "radial", "header", "provenance"])
def test_original_repository_assertions_still_detect_cuda_metadata_defects(
    generated: tuple[Path, tuple[str, ...]],
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    directory, _ = generated
    # Feed a verified source handoff, then mutate only the real CUDA audit
    # output. The repository test's original numerical/metadata assertions
    # must remain effective; its test function is not rewritten here.
    load = receipt.load
    monkeypatch.setattr(receipt, "load", lambda ignored: load(directory))
    monkeypatch.setattr(audit, "_range_programs", lambda radials: ([], ("range",)))
    inventory = audit.package_inventory_payload

    def changed_inventory(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result: dict[str, Any] = inventory(*args, **kwargs)
        if mutation == "target":
            result["target"] = "sm_80"
        elif mutation == "radial":
            result["packages"] = [
                package
                for package in result["packages"]
                if package["scientific_identity_payload"]["radial"]["family"]
                != "short_range"
            ]
        return result

    monkeypatch.setattr(audit, "package_inventory_payload", changed_inventory)
    if mutation == "header":
        monkeypatch.setattr(
            audit, "emit_cuda_derivative_shell_aot_header", lambda *args: ""
        )
    elif mutation == "provenance":
        monkeypatch.setattr(audit, "file_hash", lambda path: "")
    test = runpy.run_path(
        str(audit.ROOT / "tests/python/test_derivative_aot_audit.py")
    )["test_repository_audit_reports_cuda_shell_package_inventory"]
    with pytest.raises(AssertionError):
        test()


@pytest.mark.parametrize(
    "corruption", ["units", "output", "unit_budget", "total_budget"]
)
def test_publication_rejects_incomplete_or_out_of_budget_sources(
    generated: tuple[Path, tuple[str, ...]],
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    directory, sources = generated
    if corruption == "units":
        sources = sources[:-1]
    elif corruption == "output":
        sources = ("incorrect output", *sources[1:])
    elif corruption == "unit_budget":
        monkeypatch.setattr(receipt, "MAX_UNIT_BYTES", 1)
    else:
        monkeypatch.setattr(receipt, "MAX_PROGRAM_BYTES", 1)
    with pytest.raises(ValueError):
        receipt.publish(directory, receipt.input_identity(), sources)


def test_directory_alias_is_not_an_implicit_build_handoff(
    generated: tuple[Path, tuple[str, ...]],
    tmp_path: Path,
) -> None:
    directory, _ = generated
    alias = tmp_path / "alias"
    alias.symlink_to(directory, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        receipt.load(alias)


def test_handoff_helper_is_bound_by_native_source_identity() -> None:
    from generativeqc.autotune import _source_identity_paths

    assert (
        audit.ROOT / "tools/stationary_cpu_source_receipt.py"
        in _source_identity_paths(audit.ROOT)
    )


def test_codegen_runner_records_the_imported_handoff_dependency(
    source_tree: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from generativeqc_compiler.integral import first_derivative_schedule

    from tools import run_codegen

    sources = tuple(((), f"// source {i}\n") for i in range(receipt.CPU_AOT_SHARDS))
    monkeypatch.setattr(
        first_derivative_schedule, "derivative_cpu_aot_sources", lambda: sources
    )
    output = tmp_path / "wrapped"
    depfile = tmp_path / "wrapped.d"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_codegen",
            "--depfile",
            str(depfile),
            "--target",
            str(output / "generativeqc_stationary_cpu_derivative_0.cpp"),
            "--source-root",
            str(audit.ROOT),
            str(audit.ROOT / "tools/generate_stationary_cpu_derivative_aot.py"),
            "--output-directory",
            str(output),
        ],
    )
    run_codegen.main()
    assert receipt.load(output) == tuple(source for _, source in sources)
    assert (
        run_codegen._escape_depfile_path(
            audit.ROOT / "tools/stationary_cpu_source_receipt.py"
        )
        in depfile.read_text()
    )
