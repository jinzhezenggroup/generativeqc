"""Retirement reports and gates retain fresh filesystem and failure semantics."""

from __future__ import annotations

import json
import os
import sys
from typing import TYPE_CHECKING

import pytest

from tools.generativeqc_validation import xc_retirement as retirement

if TYPE_CHECKING:
    from pathlib import Path


def _source(root: Path, relative: str, text: str) -> Path:
    source = root / relative
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(text)
    return source


@pytest.mark.parametrize("report_json", [False, True])
@pytest.mark.parametrize("require_no_consumers", [False, True])
def test_cli_preserves_inventory_and_ordered_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    report_json: bool,
    require_no_consumers: bool,
) -> None:
    legacy_path = "python/generativeqc_compiler/xc/expressions.py"
    allowed_path = "python/generativeqc_compiler/xc/allowed.py"
    unexpected_path = "tools/consumer.py"
    formula_path = "python/generativeqc_compiler/xc/nested/new_expressions.py"
    module = "generativeqc_compiler.xc.expressions"
    _source(tmp_path, legacy_path, "pass\n")
    _source(tmp_path, allowed_path, "from . import expressions\n")
    # Escaped, adjacent literal strings still form a real dynamic import edge.
    _source(
        tmp_path,
        unexpected_path,
        "importlib.import_module('generativeqc_compiler.xc.' 'express\\u0069ons')\n",
    )
    _source(tmp_path, formula_path, "pass\n")
    monkeypatch.setattr(retirement, "LEGACY_CONSUMER_CEILING", {(allowed_path, module)})
    consumers = [
        {"path": allowed_path, "module": module, "line": 1, "kind": "import-from"},
        {
            "path": unexpected_path,
            "module": module,
            "line": 1,
            "kind": "dynamic-import",
        },
    ]
    expected_inventory = {
        "schema": "generativeqc.xc-retirement-inventory.v1",
        "legacy_sources": [
            {"module": module, "path": legacy_path, "bytes": 5, "lines": 1}
        ],
        "consumers": consumers,
        "unexpected_consumers": consumers[1:],
        "unexpected_expression_modules": [formula_path],
    }
    expected_errors = [
        f"{unexpected_path}:1: new legacy XC consumer {module} (dynamic-import)",
        f"{formula_path}: untracked handwritten-looking XC expression module",
    ]
    if require_no_consumers:
        expected_errors.extend(
            f"{path}:1: legacy XC consumer remains {module}"
            for path in (allowed_path, unexpected_path)
        )
    assert retirement.inventory(tmp_path) == expected_inventory
    assert (
        retirement.errors(tmp_path, require_no_consumers=require_no_consumers)
        == expected_errors
    )
    argv = ["xc-retirement", "--root", str(tmp_path)]
    if report_json:
        argv.append("--json")
    if require_no_consumers:
        argv.append("--require-no-consumers")
    monkeypatch.setattr(sys, "argv", argv)
    assert retirement.main() == 1
    expected_output = (
        json.dumps(expected_inventory, indent=2, sort_keys=True) + "\n"
        if report_json
        else ""
    ) + "".join(f"{error}\n" for error in expected_errors)
    assert capsys.readouterr().out == expected_output


def test_separate_calls_see_rewrites_additions_and_removals(tmp_path: Path) -> None:
    source = _source(
        tmp_path,
        "python/generativeqc_compiler/xc/consumer.py",
        "from generativeqc_compiler.xc import expressionX\n",
    )
    assert retirement.inventory(tmp_path)["consumers"] == []
    original_stat = source.stat()
    source.write_text("from generativeqc_compiler.xc import expressions\n")
    os.utime(source, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    assert source.stat().st_size == original_stat.st_size
    assert source.stat().st_mtime_ns == original_stat.st_mtime_ns
    assert len(retirement.inventory(tmp_path)["consumers"]) == 1
    assert len(retirement.unexpected_consumers(tmp_path)) == 1
    assert len(retirement.errors(tmp_path, require_no_consumers=True)) == 2

    added = _source(
        tmp_path,
        "tests/new_consumer.py",
        "import generativeqc_compiler.xc.rsh_expressions\n",
    )
    assert len(retirement.inventory(tmp_path)["consumers"]) == 2
    source.unlink()
    assert len(retirement.errors(tmp_path)) == 1
    added.unlink()
    assert retirement.inventory(tmp_path)["consumers"] == []
    assert retirement.errors(tmp_path, require_no_consumers=True) == []


@pytest.mark.parametrize("operation", ["inventory", "errors", "cli"])
def test_parser_failures_are_not_hidden_or_partially_published(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    operation: str,
) -> None:
    source = _source(tmp_path, "tools/invalid.py", "def invalid(:\n")
    monkeypatch.setattr(
        sys,
        "argv",
        ["xc-retirement", "--root", str(tmp_path), "--json", "--require-no-consumers"],
    )
    with pytest.raises(SyntaxError) as error:
        if operation == "cli":
            retirement.main()
        elif operation == "inventory":
            retirement.inventory(tmp_path)
        else:
            retirement.errors(tmp_path, require_no_consumers=True)
    assert error.value.filename == str(source.relative_to(tmp_path))
    assert capsys.readouterr().out == ""


def test_file_symlinks_are_scanned_and_broken_links_still_fail(tmp_path: Path) -> None:
    outside = _source(
        tmp_path,
        "outside/consumer.py",
        "from generativeqc_compiler.xc import expressions\n",
    )
    tools = tmp_path / "tools"
    tools.mkdir()
    link = tools / "linked.py"
    try:
        link.symlink_to(outside)
        (tools / "linked_directory").symlink_to(
            outside.parent, target_is_directory=True
        )
    except OSError as error:
        pytest.skip(f"symlink creation is unavailable: {error}")
    report = retirement.inventory(tmp_path)
    assert [consumer["path"] for consumer in report["consumers"]] == ["tools/linked.py"]
    assert len(retirement.errors(tmp_path, require_no_consumers=True)) == 2
    outside.unlink()
    with pytest.raises(FileNotFoundError):
        retirement.inventory(tmp_path)
    with pytest.raises(FileNotFoundError):
        retirement.errors(tmp_path, require_no_consumers=True)


def test_successful_cli_rechecks_after_an_edit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _source(tmp_path, "tools/consumer.py", "pass\n")
    monkeypatch.setattr(sys, "argv", ["xc-retirement", "--root", str(tmp_path)])
    assert retirement.main() == 0
    assert capsys.readouterr().out == ""
    source.write_text("import generativeqc_compiler.xc.expressions\n")
    assert retirement.main() == 1
    assert "new legacy XC consumer" in capsys.readouterr().out
