"""Common #1631 acceptance and shared policy, with no native execution."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest
from generativeqc_compiler.common.materialization import materialization_diagnostic
from generativeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    analyze_complexity,
    einsum,
    input_tensor,
    jvp,
    prepare_for_backend,
    vjp,
)

from tools.audit_native_work import audit_tree
from tools.audit_structured_materialization import MP2_REPRESENTATION_SOURCE

if TYPE_CHECKING:
    from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OV = """
for (std::size_t i=0; i<occupied; ++i)
 for (std::size_t a=occupied; a<n; ++a)
  for (std::size_t j=0; j<occupied; ++j)
   for (std::size_t b=occupied; b<n; ++b)
    weights[((i*n+a)*n+j)*n+b] += 1.0;
"""
DENSE = OV.replace("<occupied", "<n").replace("=occupied", "=0")
TRIANGLE = "for (std::size_t i=0; i<n; ++i) for (std::size_t j=0; j<=i; ++j) weights[i*n+j] = 1;"


def producer(body: str, shape: str = "n*n*n*n") -> str:
    return f"""std::vector<double> producer(std::size_t n, std::size_t occupied) {{
 std::vector<double> weights({shape}, 0.0);
 {body}
 return weights;
}}"""


def common_native(tmp_path: Path, source: str) -> dict[str, Any]:
    (tmp_path / "case.cpp").write_text(source, encoding="utf-8")
    return audit_tree(tmp_path, paths=("case.cpp",))


def diagnostic(report: dict[str, Any]) -> dict[str, Any]:
    return report["materialization_diagnostics"][0]


def codes(item: dict[str, Any]) -> set[str]:
    return {blocker["code"] for blocker in item["blockers"]}


def test_common_exact_ov_domain_retains_certificate_and_consumer_unknown(
    tmp_path: Path,
) -> None:
    report = common_native(tmp_path, producer(OV))
    item = diagnostic(report)
    assert item["storage"]["dense_elements"] == "n*n*n*n"
    assert item["storage"]["dense_growth_degree"] == 4
    assert (
        item["support"]["written_elements"]
        == "occupied*(n-occupied)*occupied*(n-occupied)"
    )
    assert item["support"]["written_growth_degree"] == 4
    assert item["support"]["status"] == "exact-address-domain"
    assert item["support"]["producer_status"] == "support-certified"
    assert item["support"]["conditions"]
    assert item["expansion_ratio"]
    assert item["recommendation"]
    assert "downstream-layout-unresolved" in codes(item)
    assert "dense-vector-layout" in codes(item)
    assert not item["consumer_abi_verified"]
    assert not item["automatic_rewrite"]
    assert not item["support"]["strict_reduction_proven"]
    finding = next(
        f
        for f in report["findings"]
        if f["rule_id"] == "native.structured-zero-materialization"
    )
    assert finding["details"]["written_count_kind"] == "exact-address-domain"
    assert finding["details"]["materialization_diagnostic"] == item


def test_common_full_domain_never_recommends_sparse_storage(tmp_path: Path) -> None:
    item = diagnostic(common_native(tmp_path, producer(DENSE)))
    assert item["support"]["status"] == "full-domain"
    assert "full-write-domain" in codes(item)
    assert item["recommendation"] is None
    assert item["expansion_ratio"] is None
    assert item["representation_candidates"] == []


@pytest.mark.parametrize("value", ["0.0", "1e-300", "1.0"])
def test_common_support_does_not_depend_on_numerical_magnitude(
    tmp_path: Path, value: str
) -> None:
    structured = diagnostic(common_native(tmp_path, producer(OV.replace("1.0", value))))
    assert structured["support"]["status"] == "exact-address-domain"
    assert (
        structured["support"]["written_elements"]
        == "occupied*(n-occupied)*occupied*(n-occupied)"
    )
    full = diagnostic(common_native(tmp_path, producer(DENSE.replace("1.0", value))))
    assert full["support"]["status"] == "full-domain"
    assert full["recommendation"] is None


def test_common_lower_triangle_is_packed_quadratic_support(tmp_path: Path) -> None:
    item = diagnostic(common_native(tmp_path, producer(TRIANGLE, "n*n")))
    assert item["support"]["written_elements"] == "n*(n+1)/2"
    assert item["support"]["written_growth_degree"] == 2
    assert item["storage"]["dense_growth_degree"] == 2
    assert item["support"]["domains"][0]["kind"] == "lower-triangle"
    assert item["representation_candidates"] == ["packed"]
    assert "threshold" not in json.dumps(item)


def test_common_union_bound_and_unsupported_producer_do_not_gain_proof(
    tmp_path: Path,
) -> None:
    union = diagnostic(
        common_native(tmp_path, producer(OV + OV.replace("i<occupied", "i<n")))
    )
    assert union["support"]["status"] == "union-upper-bound"
    assert "union-cardinality-unresolved" in codes(union)
    assert union["recommendation"] is union["expansion_ratio"] is None
    unknown = diagnostic(common_native(tmp_path, producer("mutate(weights);" + OV)))
    assert unknown["support"]["status"] == "unknown"
    assert "unsupported-write-support" in codes(unknown)
    assert unknown["recommendation"] is unknown["expansion_ratio"] is None


def test_common_mp2_source_census_is_visible_without_sparse_or_abi_claim() -> None:
    report = audit_tree(ROOT, paths=(MP2_REPRESENTATION_SOURCE,))
    boundary = report["production_boundaries"][0]
    assert boundary["status"] == "SOURCE_VISIBLE"
    assert len(boundary["observed_source_roles"]) == 8
    assert not boundary["exact_write_support_proven"]
    assert not boundary["consumer_abi_verified"]
    assert not boundary["runtime_endpoint_selection_proven"]
    item = boundary["materialization_diagnostic"]
    assert item in report["materialization_diagnostics"]
    assert item["storage"]["dense_growth_degree"] is None
    assert "fourth_power(n)" in item["storage"]["dense_elements"]
    assert item["observed_source_roles"] == boundary["observed_source_roles"]
    assert item["recommendation"] is None
    assert "unsupported-write-support" in codes(item)
    assert "aggregate-member-layout" in codes(item)
    candidate = next(
        f["details"]
        for f in report["findings"]
        if f["rule_id"] == "native.structured-zero-materialization"
        and f["function"] == "initial_orbital_weights"
        and f["details"]["buffer"] == "result.two_electron"
    )
    assert candidate["classification"] == "unknown"
    assert candidate["recommendation"] is None
    assert "assign freshness" in candidate["unknown_reason"]


def test_mp2_role_anchors_do_not_certify_helper_arithmetic(tmp_path: Path) -> None:
    source = (ROOT / MP2_REPRESENTATION_SOURCE).read_text(encoding="utf-8")
    anchor = "return posthf::checked_mul(value, value);"
    assert anchor in source
    # The return anchor survives, but this is no longer the square of the input.
    changed = source.replace(anchor, "++value; " + anchor)
    target = tmp_path / MP2_REPRESENTATION_SOURCE
    target.parent.mkdir(parents=True)
    target.write_text(changed, encoding="utf-8")
    boundary = audit_tree(tmp_path, paths=(MP2_REPRESENTATION_SOURCE,))[
        "production_boundaries"
    ][0]
    assert boundary["status"] == "SOURCE_VISIBLE"
    item = boundary["materialization_diagnostic"]
    assert "fourth_power(n)" in item["storage"]["dense_elements"]
    assert item["storage"]["dense_growth_degree"] is None
    assert item["support"]["status"] == "unknown"
    assert item["recommendation"] is None
    assert not boundary["exact_write_support_proven"]
    assert not boundary["consumer_abi_verified"]
    assert not boundary["runtime_endpoint_selection_proven"]


@pytest.mark.parametrize("mutation", ["changed", "missing", "malformed"])
def test_common_mp2_role_changes_fail_closed(tmp_path: Path, mutation: str) -> None:
    source = (ROOT / MP2_REPRESENTATION_SOURCE).read_text(encoding="utf-8")
    anchor = "result.two_electron.assign(fourth_power(n), 0.0);"
    assert anchor in source
    if mutation == "changed":
        source = source.replace(anchor, anchor.replace("fourth_power(n)", "square(n)"))
    elif mutation == "missing":
        source = source.replace(anchor, "")
    else:
        source = source.replace(anchor, anchor.replace("assign(", "assign["))
    target = tmp_path / MP2_REPRESENTATION_SOURCE
    target.parent.mkdir(parents=True)
    target.write_text(source, encoding="utf-8")
    boundary = audit_tree(tmp_path, paths=(MP2_REPRESENTATION_SOURCE,))[
        "production_boundaries"
    ][0]
    assert boundary["status"] == "INCOMPLETE"
    assert boundary["missing_roles"]
    assert not boundary["exact_write_support_proven"]
    assert not boundary["consumer_abi_verified"]
    assert not boundary["runtime_endpoint_selection_proven"]
    assert boundary["materialization_diagnostic"]["recommendation"] is None
    assert (
        boundary["materialization_diagnostic"]["storage"]["dense_growth_degree"] is None
    )


def test_common_python_uses_same_policy_and_preserves_upper_bound(
    tmp_path: Path,
) -> None:
    source = tmp_path / "case.py"
    source.write_text(
        "import numpy as np\ndef f(n):\n x = np.zeros((n, n))\n x[np.diag_indices(n)] = 1\n return x\n",
        encoding="utf-8",
    )
    report = audit_tree(tmp_path, paths=("case.py",))
    item = diagnostic(report)
    assert item["schema"] == "generativeqc.materialization-diagnostic.v1"
    assert item["origin"] == "python-source"
    assert item["support"]["status"] == "union-upper-bound"
    assert item["recommendation"] is None
    assert item["expansion_ratio"] is None
    assert "ordinary NumPy semantics" in item["support"]["certificate_scope"]
    details = report["findings"][0]["details"]
    assert item["support"]["written_elements"] == details["support_upper_bound"]
    assert item["support"]["domains"] == details["write_support"]


def test_common_provenance_hashes_sources_and_every_analyzer(tmp_path: Path) -> None:
    source = producer(OV)
    report = common_native(tmp_path, source)
    provenance = report["provenance"]
    assert provenance["source_hashes"] == {
        "case.cpp": hashlib.sha256(source.encode()).hexdigest()
    }
    hashes = provenance["scanner_hashes"]
    assert set(hashes) == {
        "audit_structured_materialization.py",
        "audit_native_work.py",
        "audit_native_complexity.py",
        "work_audit_python.py",
        "python/generativeqc_compiler/common/materialization.py",
    }
    for name, digest in hashes.items():
        path = ROOT / name if name.startswith("python/") else ROOT / "tools" / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert (
        provenance["scanner_digest"]
        == hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    )
    assert provenance["analyzer_digest"] == provenance["scanner_digest"]
    assert provenance["scanned_source_dirty"] is None


def test_common_strict_path_selection_and_alias_deduplication(tmp_path: Path) -> None:
    source = tmp_path / "case.cpp"
    source.write_text(producer(OV), encoding="utf-8")
    alias = tmp_path / "alias.cpp"
    alias.symlink_to(source)
    report = audit_tree(tmp_path, paths=("case.cpp", "alias.cpp", "case.cpp"))
    assert report["scanned_files"] == {"native_lexical": 1}
    assert len(report["materialization_diagnostics"]) == 1
    assert report["provenance"]["alias_topology_unverified"]
    for path in ("missing.cpp", "../outside.cpp"):
        with pytest.raises(ValueError):
            audit_tree(tmp_path, paths=(path,))
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "leak.cpp").symlink_to(source)
    with pytest.raises(ValueError):
        audit_tree(outside, paths=(".",))


def test_common_git_unavailable_preserves_unknown_provenance(tmp_path: Path) -> None:
    with patch(
        "tools.audit_structured_materialization.subprocess.run",
        side_effect=OSError("git unavailable"),
    ):
        provenance = common_native(tmp_path, producer(OV))["provenance"]
    for key in ("commit", "tree", "working_tree_dirty", "scanned_source_dirty"):
        assert provenance[key] is None


def test_common_consumes_snapshot_without_reading_replaced_outside_symlink(
    tmp_path: Path,
) -> None:
    from tools import audit_structured_materialization as structured

    root = tmp_path / "root"
    root.mkdir()
    source = root / "case.cpp"
    original = producer(OV)
    source.write_text(original, encoding="utf-8")
    outside = tmp_path / "outside.cpp"
    # Identical bytes defeat a hash-only check of a second filesystem read.
    outside.write_text(original, encoding="utf-8")
    actual_audit = structured.audit_tree
    original_read = Path.read_bytes
    reads = []

    def capture_read(path: Path) -> bytes:
        if path == source:
            assert not path.is_symlink(), "the common audit reread the replaced path"
            reads.append(path)
        assert path != outside, "outside-root bytes must never be read"
        return original_read(path)

    def replace_after_scan(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = actual_audit(*args, **kwargs)
        source.unlink()
        source.symlink_to(outside)
        return result

    with (
        patch.object(structured, "audit_tree", replace_after_scan),
        patch.object(Path, "read_bytes", capture_read),
    ):
        report = audit_tree(root, paths=("case.cpp",))
    assert reads == [source]
    assert source.is_symlink()
    assert diagnostic(report)["support"]["status"] == "exact-address-domain"
    assert (
        report["provenance"]["source_hashes"]["case.cpp"]
        == hashlib.sha256(original.encode()).hexdigest()
    )
    assert report["provenance"]["source_content_basis"] == "captured-source-bytes"
    assert report["provenance"]["filesystem_state_basis"] == "scan-time-observation"
    assert report["provenance"]["scanned_source_dirty"] is None


def test_long_lived_import_cannot_hash_new_policy_and_execute_old_policy(
    tmp_path: Path,
) -> None:
    copy = tmp_path / "checkout"
    paths = [
        "tools/audit_native_work.py",
        "tools/audit_native_complexity.py",
        "tools/audit_structured_materialization.py",
        "tools/work_audit_python.py",
        "python/generativeqc_compiler/__init__.py",
        "python/generativeqc_compiler/common/__init__.py",
        "python/generativeqc_compiler/common/materialization.py",
    ]
    for relative in paths:
        target = copy / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    (copy / "case.cpp").write_text(producer(OV), encoding="utf-8")
    script = """
from pathlib import Path
from generativeqc_compiler.common import materialization
from tools.audit_native_work import audit_tree
root = Path.cwd()
first = audit_tree(root, paths=('case.cpp',))
assert first['materialization_diagnostics'][0]['recommendation'].startswith('Review ')
policy = Path(materialization.__file__)
text = policy.read_text()
assert '"Review "' in text
policy.write_text(text.replace('"Review "', '"CHANGED_POLICY "'))
try:
    audit_tree(root, paths=('case.cpp',))
except ValueError as error:
    assert 'policy source changed after import' in str(error)
else:
    raise AssertionError('stale policy must not produce a new-source receipt')

# Explicit reload during a multi-source scan must not mix policy generations.
import importlib
from tools.audit_structured_materialization import audit_tree as structured_audit
importlib.reload(materialization)
(root / 'second.cpp').write_bytes((root / 'case.cpp').read_bytes())
def reload_on_second_source(relative, data):
    if relative == 'second.cpp':
        policy.write_text(policy.read_text().replace('CHANGED_POLICY ', 'RELOADED_POLICY '))
        importlib.reload(materialization)
try:
    structured_audit(root, paths=('case.cpp', 'second.cpp'), _source_visitor=reload_on_second_source)
except ValueError as error:
    assert 'policy changed during scan' in str(error)
else:
    raise AssertionError('mixed policy generations must not emit a receipt')
"""
    environment = {**os.environ, "PYTHONPATH": str(copy / "python")}
    subprocess.run(
        [sys.executable, "-S", "-c", script],
        cwd=copy,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )


def test_stdlib_only_cli_and_canonical_policy_import(tmp_path: Path) -> None:
    (tmp_path / "case.cpp").write_text(producer(OV), encoding="utf-8")
    (tmp_path / "case.py").write_text(
        "import numpy as np\ndef f(n):\n x=np.zeros((n,n))\n x[np.diag_indices(n)]=1\n return x\n",
        encoding="utf-8",
    )
    script = f"""
import json, runpy, sys
sys.argv = ['audit_native_work.py', '--root', sys.argv[1], '--path', '.', '--format', 'json']
runpy.run_path({str(ROOT / "tools/audit_native_work.py")!r}, run_name='__main__')
assert not any(name.split('.')[0] in {{'generativeqc', 'numpy', 'pyscf', 'torch', 'cupy'}} for name in sys.modules)
assert 'generativeqc_compiler.common.materialization' in sys.modules
"""
    # The tool directory is the normal sys.path[0] of direct script execution.
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "tools")}
    result = subprocess.run(
        [sys.executable, "-S", "-c", script, str(tmp_path)],
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    report = json.loads(result.stdout)
    assert {item["origin"] for item in report["materialization_diagnostics"]} == {
        "native-source",
        "python-source",
    }


def test_tensorir_shared_diagnostics_leave_equation_ad_and_preparation_unchanged() -> (
    None
):
    space = IndexSpace("ao", "ao", 2)
    matrix = input_tensor(
        "a",
        TensorSpec(
            (Index("i", space), Index("j", space)), role="input", differentiable=True
        ),
    )
    program = Program({"quartic": einsum("ij,kl->ijkl", matrix, matrix)})
    payload = program.to_payload()
    logical_hash = program.logical_hash
    feeds = {"a": np.arange(4, dtype=float).reshape(2, 2)}
    tangents = {"a": np.ones((2, 2))}
    cotangents = {"quartic": np.ones((2, 2, 2, 2))}
    forward = jvp(program, feeds, tangents)
    reverse = vjp(program, feeds, cotangents)
    prepared = prepare_for_backend(program, "cpu")
    report = analyze_complexity(program)
    summary = report.summary_payload()
    items = report.materialization_diagnostics()
    retained = next(item for item in items if item["storage"]["retained_output"])
    assert retained["schema"] == "generativeqc.materialization-diagnostic.v1"
    assert retained["storage"]["dense_elements"] == 16
    assert retained["storage"]["dense_growth_degree"] == 4
    assert retained["support"]["status"] == "missing-structured-ir"
    assert retained["support"]["written_elements"] is None
    assert {"missing-exact-support", "retained-output-layout"} <= codes(retained)
    assert all(item["recommendation"] is None for item in items)
    assert report.summary_payload() == summary
    assert "materialization" not in program.provenance
    assert program.to_payload() == payload
    assert program.logical_hash == logical_hash
    assert prepare_for_backend(program, "cpu").to_payload() == prepared.to_payload()
    assert jvp(program, feeds, tangents).derivative_hash == forward.derivative_hash
    assert vjp(program, feeds, cotangents).derivative_hash == reverse.derivative_hash
    np.testing.assert_array_equal(
        jvp(program, feeds, tangents).output_tangents["quartic"],
        forward.output_tangents["quartic"],
    )
    np.testing.assert_array_equal(
        vjp(program, feeds, cotangents).input_cotangents["a"],
        reverse.input_cotangents["a"],
    )


def test_shared_policy_rejects_fabricated_empty_support() -> None:
    with pytest.raises(ValueError, match="analyzer-owned"):
        materialization_diagnostic(
            origin="test",
            subject={},
            dense_elements=16,
            support_kind="exact-address-domain",
            certificate_scope="none",
        )
