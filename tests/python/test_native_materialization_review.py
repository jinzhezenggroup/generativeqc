"""Candidate review must bind source, not just a count or a line number."""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import pytest

from tools import ratchet_native_materialization as gate

ROOT = Path(__file__).resolve().parents[2]
SOURCE = """
void stage(size_t n, double* out, const double* input) {
  for (size_t i=0; i<n; ++i)
    for (size_t j=0; j<n; ++j)
      for (size_t k=0; k<n; ++k)
        for (size_t l=0; l<n; ++l)
          out[((i*n+j)*n+k)*n+l] += input[((i*n+j)*n+k)*n+l];
  consume(out);
}
"""


def tree(root: Path, source: str = SOURCE) -> dict:
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "include").mkdir(exist_ok=True)
    (root / "src/stage.cpp").write_text(source)
    (root / "src/consumer.cpp").write_text(
        "void consume(double* data) { use(data); }\n"
    )
    return gate.inventory(root)


def manifest(report: dict) -> dict:
    return {
        "schema": gate.MANIFEST_SCHEMA,
        "candidates": [
            {
                "candidate_id": row["candidate_id"],
                "owner": "#1626",
                "disposition": "retained-pending-evidence",
                "reason": "Fusion profitability and consumer lifetime remain unresolved.",
                "producer": {"path": row["path"], "symbol": "stage"},
                "consumers": [{"path": "src/consumer.cpp", "symbol": "consume"}],
                "residency": "host vector",
                "lifetime": "one call",
                "resource_owner": "stage",
                "evidence_gaps": ["Independent parity and complete endpoint timing"],
                "source_bindings": dict(report["source_identities"]),
            }
            for row in report["candidates"]
        ],
    }


def test_comment_line_shifts_keep_review_but_producer_changes_require_it(
    tmp_path: Path,
) -> None:
    before = tree(tmp_path)
    review = manifest(before)
    shifted = tree(tmp_path, "// a new explanation\n\n" + SOURCE)
    result = gate.compare(before, shifted, review)
    assert result["status"] == "PASS"
    assert len(result["unchanged"]) == 1
    assert before["candidates"][0]["line"] != shifted["candidates"][0]["line"]
    after = tree(tmp_path, SOURCE.replace("+= input", "+= 2 * input"))
    result = gate.compare(before, after, review)
    assert result["status"] == "NEEDS_REVIEW"
    assert result["counts"] == {"baseline": 1, "candidate": 1, "delta": 0}
    assert len(result["changed"]) == 1
    assert gate.compare(before, after, manifest(after))["status"] == "PASS"


def test_consumer_changes_in_same_or_declared_external_file_require_review(
    tmp_path: Path,
) -> None:
    before = tree(tmp_path)
    review = manifest(before)
    after = tree(tmp_path, SOURCE.replace("consume(out)", "consume_other(out)"))
    assert gate.compare(before, after, review)["status"] == "NEEDS_REVIEW"
    tree(tmp_path)
    (tmp_path / "src/consumer.cpp").write_text(
        "void consume(double* data) { retain(data); }"
    )
    after = gate.inventory(tmp_path)
    result = gate.compare(before, after, review)
    assert result["status"] == "NEEDS_REVIEW"
    assert len(result["changed"]) == 1
    assert (
        result["changed"][0]["before"]["source_identity"]
        == result["changed"][0]["after"]["source_identity"]
    )


def test_equal_count_replacement_retains_added_and_removed_sites(
    tmp_path: Path,
) -> None:
    before = tree(tmp_path)
    after = tree(tmp_path, SOURCE.replace("out[", "other["))
    result = gate.compare(before, after, manifest(before))
    assert result["status"] == "NEEDS_REVIEW"
    assert result["counts"]["delta"] == 0
    assert len(result["removed"]) == len(result["added"]) == 1
    assert result["removed"][0] == before["candidates"][0]
    assert any("unmatched/stale" in row["reason"] for row in result["review_errors"])
    assert gate.compare(before, after, manifest(after))["status"] == "PASS"


def test_removal_is_evidence_not_retirement_proof_and_stale_record_fails(
    tmp_path: Path,
) -> None:
    before = tree(tmp_path)
    after = tree(tmp_path, "void removed() {}\n")
    result = gate.compare(before, after, manifest(before))
    assert result["status"] == "NEEDS_REVIEW"
    result = gate.compare(before, after, manifest(after))
    assert result["status"] == "PASS"
    assert result["counts"]["delta"] == -1
    assert len(result["removed"]) == 1
    assert "not proof" in result["limitations"][0]


@pytest.mark.parametrize(
    "field",
    ["owner", "reason", "source_bindings", "producer", "consumers", "evidence_gaps"],
)
def test_missing_review_information_fails(tmp_path: Path, field: str) -> None:
    report = tree(tmp_path)
    review = manifest(report)
    del review["candidates"][0][field]
    assert gate.compare(report, report, review)["status"] == "NEEDS_REVIEW"


def test_duplicates_and_unmatched_bindings_fail(tmp_path: Path) -> None:
    report = tree(tmp_path)
    review = manifest(report)
    review["candidates"] *= 2
    assert gate.compare(report, report, review)["status"] == "NEEDS_REVIEW"
    review = manifest(report)
    review["candidates"][0]["source_bindings"]["src/missing.cpp"] = "0" * 64
    assert gate.compare(report, report, review)["status"] == "NEEDS_REVIEW"


@pytest.mark.parametrize(
    "mutation", ["schema", "count", "identity", "source", "analyzer", "empty"]
)
def test_malformed_inventory_never_passes(tmp_path: Path, mutation: str) -> None:
    before = tree(tmp_path)
    after = copy.deepcopy(before)
    if mutation == "schema":
        after["schema"] = "wrong"
    elif mutation == "count":
        after["candidate_count"] = 0
    elif mutation == "identity":
        after["candidates"][0]["candidate_id"] = "0" * 64
    elif mutation == "source":
        after["source_identities"] = {}
    elif mutation == "analyzer":
        after["provenance"]["scanner_digest"] = "0" * 64
    else:
        after = {}
    with pytest.raises(gate.ReviewError):
        gate.compare(before, after, manifest(before))


def test_literal_identity_is_not_masked_or_whitespace_collapsed() -> None:
    assert gate.source_identity('void f() { use("a b"); }') != gate.source_identity(
        'void f() { use("ab"); }'
    )
    assert gate.source_identity("void f() { use('N'); }") != gate.source_identity(
        "void f() { use('T'); }"
    )
    assert gate.source_identity('void f() { use(R"(a  b)"); }') != gate.source_identity(
        'void f() { use(R"(a b)"); }'
    )


def test_cli_missing_baseline_fails_closed_and_publishes_receipt(
    tmp_path: Path,
) -> None:
    output = tmp_path / "report.json"
    assert gate.main(["--output", str(output), "--fail-on-unreviewed"]) == 2
    assert json.loads(output.read_text())["status"] == "INCOMPLETE"
    assert (
        gate.main(
            [
                "--root",
                str(tmp_path),
                "--base-sha",
                "0" * 40,
                "--output",
                str(output),
                "--fail-on-unreviewed",
            ]
        )
        == 2
    )
    assert json.loads(output.read_text())["status"] == "INCOMPLETE"


def test_exact_git_base_uses_current_analyzer_and_allows_manifest_bootstrap(
    tmp_path: Path,
) -> None:
    current = tree(tmp_path)
    for args in (
        ("init",),
        ("add", "src", "include"),
        (
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-m",
            "base",
        ),
    ):
        subprocess.run(
            ["git", "-C", str(tmp_path), *args], check=True, capture_output=True
        )
    base = subprocess.check_output(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"], text=True
    ).strip()
    # Git does not track empty directories.
    (tmp_path / "include/marker.hpp").write_text("// marker\n")
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "include"], check=True, capture_output=True
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-m",
            "include",
        ],
        check=True,
        capture_output=True,
    )
    base = subprocess.check_output(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"], text=True
    ).strip()
    current = gate.inventory(tmp_path)
    review = tmp_path / "review.json"
    review.write_text(json.dumps(manifest(current)))
    result = gate.audit(tmp_path, base, review)
    assert result["status"] == "PASS"
    assert result["base_commit"] == base
    assert (
        result["baseline"]["provenance"]["source_content_basis"]
        == "git-archive-at-base-commit"
    )
    review.write_text("{broken")
    output = tmp_path / "result.json"
    assert (
        gate.main(
            [
                "--root",
                str(tmp_path),
                "--base-sha",
                base,
                "--manifest",
                str(review),
                "--output",
                str(output),
                "--fail-on-unreviewed",
            ]
        )
        == 2
    )
    assert json.loads(output.read_text())["status"] == "INCOMPLETE"


def test_workflow_enforces_review_and_uploads_receipt_after_failure() -> None:
    workflow = (ROOT / ".github/workflows/work-audit.yml").read_text()
    assert "--fail-on-unreviewed" in workflow
    assert workflow.count(".artifacts/native-materialization-review.json") == 2
    assert (
        "if: always()"
        in workflow.split("Require source-bound materialization candidate review", 1)[
            1
        ].split("env:", 1)[0]
    )


@pytest.mark.parametrize(
    "function", [gate.audit_text, gate._functions, gate.structured_audit]
)
def test_stale_import_identity_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, function: object
) -> None:
    tree(tmp_path)
    monkeypatch.setitem(function.__globals__, "_SOURCE_SHA256", "0" * 64)
    with pytest.raises(gate.ReviewError, match="changed after import"):
        gate.inventory(tmp_path)


def test_analyzer_change_during_scan_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tree(tmp_path)
    original = Path.read_bytes
    calls = 0
    target = Path(gate.audit_text.__globals__["__file__"])

    def changed(path: Path) -> bytes:
        nonlocal calls
        data = original(path)
        if path == target:
            calls += 1
            if calls > 1:
                return data + b"\n# edited during scan\n"
        return data

    monkeypatch.setattr(Path, "read_bytes", changed)
    with pytest.raises(gate.ReviewError, match="changed after import"):
        gate.inventory(tmp_path)


def test_baseline_export_attributes_cannot_hide_a_source(tmp_path: Path) -> None:
    tree(tmp_path)
    (tmp_path / "include/marker.hpp").write_text("// marker\n")
    (tmp_path / ".gitattributes").write_text("src/stage.cpp export-ignore\n")
    for args in (
        ("init",),
        ("add", "."),
        (
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-m",
            "base",
        ),
    ):
        subprocess.run(
            ["git", "-C", str(tmp_path), *args], check=True, capture_output=True
        )
    base = subprocess.check_output(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"], text=True
    ).strip()
    with pytest.raises(gate.ReviewError, match="omits source files"):
        gate._baseline_tree(tmp_path, base, tmp_path / "extracted")


@pytest.mark.parametrize("directive", ["#", "%:", "??="])
def test_preprocessor_line_endings_cannot_move_a_consumer_unreviewed(
    tmp_path: Path,
    directive: str,
) -> None:
    source = SOURCE.replace("  consume(out);\n", "")
    source = source.replace(
        "  for (size_t i", "#define EXTRA\n  consume(out);\n  for (size_t i"
    )
    source = source.replace("\n}\n", "\n  EXTRA\n}\n")
    source = source.replace("#define", directive + "define")
    before = tree(tmp_path, source)
    after = tree(
        tmp_path,
        source.replace(
            directive + "define EXTRA\n  consume(out);",
            directive + "define EXTRA consume(out);",
        ),
    )
    assert before["source_identities"] != after["source_identities"]
    assert gate.compare(before, after, manifest(before))["status"] == "NEEDS_REVIEW"
    shifted = tree(tmp_path, "// explanation\n\n" + source)
    assert gate.compare(before, shifted, manifest(before))["status"] == "NEEDS_REVIEW"


def test_real_module_reload_cannot_relabel_retained_callable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib.util
    import sys

    path = tmp_path / "audit_native_complexity.py"
    original = Path(gate.audit_text.__globals__["__file__"]).read_text()
    path.write_text(original)
    spec = importlib.util.spec_from_file_location("_materialization_reload_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(gate, "audit_text", module.audit_text)
    verifier = gate._bind_analyzers()
    verifier()
    retained = gate.audit_text
    path.write_text(original.replace("return tuple(findings)", "return tuple()"))
    spec.loader.exec_module(module)
    assert retained is not module.audit_text
    with pytest.raises(gate.ReviewError, match="changed after import"):
        verifier()


@pytest.mark.parametrize(
    "field",
    [
        "disposition",
        "producer",
        "consumer",
        "candidate_id",
        "source_bindings",
        "owner",
        "consumers",
    ],
)
@pytest.mark.parametrize("invalid", [[], {}])
def test_invalid_json_shapes_produce_failure_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, invalid: object
) -> None:
    report = tree(tmp_path)
    review = manifest(report)
    if field == "producer":
        review["candidates"][0]["producer"]["path"] = invalid
    elif field == "consumer":
        review["candidates"][0]["consumers"][0]["path"] = invalid
    else:
        review["candidates"][0][field] = invalid

    def compare_invalid(*args: object) -> dict:
        return gate.compare(report, report, review)

    monkeypatch.setattr(gate, "audit", compare_invalid)
    output = tmp_path / "receipt.json"
    assert gate.main(
        ["--base-sha", "0" * 40, "--output", str(output), "--fail-on-unreviewed"]
    ) in {1, 2}
    assert json.loads(output.read_text())["status"] in {"NEEDS_REVIEW", "INCOMPLETE"}


@pytest.mark.parametrize(
    "token", ["__LINE__", "__builtin_LINE()", "std::source_location::current()"]
)
def test_line_sensitive_code_requires_review_after_line_insertion(
    tmp_path: Path, token: str
) -> None:
    source = SOURCE.replace("consume(out)", f"consume({token})")
    before = tree(tmp_path, source)
    after = tree(tmp_path, "\n" + source)
    assert gate.compare(before, after, manifest(before))["status"] == "NEEDS_REVIEW"


def test_imported_helper_binding_changes_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def replacement(*args: object) -> int:
        return 0

    monkeypatch.setitem(gate._functions.__globals__, "_matching", replacement)
    with pytest.raises(gate.ReviewError, match="helper changed after import"):
        gate._analyzer_hashes()


def test_dependency_reloaded_before_gate_binding_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib.util
    import sys

    path = tmp_path / "audit_native_complexity.py"
    path.write_text(Path(gate.audit_text.__globals__["__file__"]).read_text())
    spec = importlib.util.spec_from_file_location("_materialization_early_reload", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    old_helper = module._matching
    spec.loader.exec_module(module)
    assert old_helper is not module._matching
    monkeypatch.setattr(gate, "audit_text", module.audit_text)
    monkeypatch.setitem(gate._functions.__globals__, "_matching", old_helper)
    verifier = gate._bind_analyzers()
    with pytest.raises(gate.ReviewError, match="helper changed after import"):
        verifier()


def test_real_reload_between_scan_entry_and_emission_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib.util
    import sys

    tree(tmp_path)
    path = tmp_path / "audit_native_complexity.py"
    path.write_text(Path(gate.audit_text.__globals__["__file__"]).read_text())
    spec = importlib.util.spec_from_file_location(
        "_materialization_midscan_reload", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(gate, "audit_text", module.audit_text)
    verifier = gate._bind_analyzers()
    reloaded = False
    original = Path.read_bytes

    def read_and_reload(source: Path) -> bytes:
        nonlocal reloaded
        data = original(source)
        if source == tmp_path / "src/stage.cpp":
            spec.loader.exec_module(module)
            reloaded = True
        return data

    monkeypatch.setattr(Path, "read_bytes", read_and_reload)
    with pytest.raises(gate.ReviewError, match="changed after import"):
        gate.inventory(tmp_path, _verify=verifier)
    assert reloaded


@pytest.mark.parametrize("directive", ["#", "%:", "??="])
@pytest.mark.parametrize(
    "header", ["dir//old.hpp", "dir/*old*/file.hpp", "dir/*old.hpp"]
)
def test_comment_spelling_in_angle_header_cannot_hide_source_changes(
    directive: str, header: str
) -> None:
    before = f"{directive}include <{header}>\nvoid f() {{ use(old); }}\n"
    assert gate.source_identity(before) != gate.source_identity(
        before.replace("old.hpp", "new.hpp").replace("/*old*/", "/*new*/")
    )
    assert gate.source_identity(before) != gate.source_identity(
        before.replace("use(old)", "use(new)")
    )


def test_multiline_comments_do_not_hide_preprocessor_semantics() -> None:
    before = "#define EXTRA /*\n*/ consume(out);\nEXTRA\n"
    after = "#define EXTRA\nconsume(out);\nEXTRA\n"
    assert gate.source_identity(before) != gate.source_identity(after)


def test_retained_gate_entrypoints_reject_module_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib.util
    import sys

    report = tree(tmp_path)
    path = tmp_path / "ratchet_native_materialization.py"
    original = Path(gate.__file__).read_text()
    path.write_text(original)
    spec = importlib.util.spec_from_file_location("_materialization_gate_reload", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    retained_inventory, retained_compare, retained_audit, retained_main = (
        module.inventory,
        module.compare,
        module.audit,
        module.main,
    )
    path.write_text(original.replace("scanned += 1", "scanned += 7"))
    spec.loader.exec_module(module)
    with pytest.raises(module.ReviewError, match="changed after import"):
        retained_inventory(tmp_path)
    with pytest.raises(module.ReviewError, match="changed after import"):
        retained_compare(report, report, manifest(report))
    with pytest.raises(module.ReviewError, match="changed after import"):
        retained_audit(tmp_path, "0" * 40, tmp_path / "absent.json")
    output = tmp_path / "reload-receipt.json"
    assert (
        retained_main(
            [
                "--root",
                str(tmp_path),
                "--inventory-only",
                "--output",
                str(output),
                "--fail-on-unreviewed",
            ]
        )
        == 2
    )
    assert json.loads(output.read_text())["status"] == "INCOMPLETE"
