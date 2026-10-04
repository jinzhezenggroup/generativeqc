"""Mutations must not hide new method-local CUDA vendor submissions."""

from __future__ import annotations

import json
import typing

import pytest

from tools.check_vendor_boundaries import MANIFEST, audit_vendor_boundaries

if typing.TYPE_CHECKING:
    from pathlib import Path


def _fixture(
    root: Path,
    source: str,
    *,
    path: str = "src/cc/example.cu",
    category: str = "migration",
) -> Path:
    file = root / path
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(source)
    manifest = root / MANIFEST
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "schema": 1,
                "files": {
                    path: {
                        "classification": category,
                        "contract": "#1890",
                        "reason": "Fixture migration debt.",
                        "references": {"cublasDgemm": 1},
                    }
                },
            }
        )
    )
    return file


def test_repository_inventory_is_complete() -> None:
    assert audit_vendor_boundaries()["errors"] == []


@pytest.mark.parametrize(
    "expression",
    [
        "cublasDgemm(h);",
        "auto call = &cublasDgemm;",
        "#define SUBMIT cublasDgemm",
        "cutensorContract(h);",
        "cublasLtMatmul(h);",
        "cusolverDnXsyevd(h);",
        "cusparseSpMM(h);",
        "ncclAllReduce(h);",
        "cub :: BlockReduce<double> reduction;",
        "cutlass::gemm::device::Gemm<> operation;",
        "cute::gemm(a, b);",
        "using namespace cub;",
        "namespace backend = cutlass;",
    ],
)
def test_new_native_files_are_not_exempt(tmp_path: Path, expression: str) -> None:
    _fixture(tmp_path, "cublasDgemm(h);")
    # Even a file under the provider directory needs explicit classification.
    new = tmp_path / "src/tensor/unclassified.cuh"
    new.parent.mkdir(parents=True)
    new.write_text(expression)
    errors = audit_vendor_boundaries(tmp_path)["errors"]
    assert len(errors) == 1 and "unclassified vendor references" in errors[0]


def test_same_symbol_growth_and_stale_debt_both_fail(tmp_path: Path) -> None:
    file = _fixture(tmp_path, "cublasDgemm(h); cublasDgemm(other);")
    assert "count 2 != classified 1" in audit_vendor_boundaries(tmp_path)["errors"][0]
    file.write_text("semantic_contraction(h);")
    assert "remove retired" in audit_vendor_boundaries(tmp_path)["errors"][0]


def test_provider_still_requires_new_symbols_to_be_classified(tmp_path: Path) -> None:
    file = _fixture(
        tmp_path, "cublasDgemm(h); cublasDgemm(other);", category="provider"
    )
    assert audit_vendor_boundaries(tmp_path)["errors"] == []
    file.write_text("cublasDgemm(h); cublasSgemm(other);")
    assert (
        "unclassified vendor reference cublasSgemm"
        in audit_vendor_boundaries(tmp_path)["errors"][0]
    )


@pytest.mark.parametrize(
    "path",
    [
        "python/generativeqc_compiler/method/new.py",
        "tools/generate_new.py",
    ],
)
def test_generator_fragments_cannot_hide_submissions(tmp_path: Path, path: str) -> None:
    file = _fixture(tmp_path, 'lines = [f"cublasDgemm({handle});"]', path=path)
    assert audit_vendor_boundaries(tmp_path)["errors"] == []
    file.write_text(
        'lines = ["// open comment", f"cublasDgemm({handle});", "cublasSgemm(h);"]'
    )
    assert "cublasSgemm" in audit_vendor_boundaries(tmp_path)["errors"][0]


def test_native_comments_strings_includes_and_digit_separators(tmp_path: Path) -> None:
    _fixture(
        tmp_path,
        """
#include <cusolverDn.h>
// cublasSgemm(h);
/* cutensorContract(h); */
const char* diagnostic = R"tag(cublasLtMatmul(h); //)tag";
const char* quoted = "cusparseSpMM(h);";
auto size = 1'024;
cublasDgemm(h);
""",
    )
    assert audit_vendor_boundaries(tmp_path)["errors"] == []


def test_generator_docstrings_are_not_submissions(tmp_path: Path) -> None:
    _fixture(
        tmp_path,
        '''"""Do not emit cublasSgemm(h)."""
def generate():
    """Emit a contraction; no cutensorContract(h)."""
    return "cublasDgemm(h);"
''',
        path="tools/generate_new.py",
    )
    assert audit_vendor_boundaries(tmp_path)["errors"] == []


def test_diagnostic_exception_does_not_authorize_new_calls(tmp_path: Path) -> None:
    _fixture(tmp_path, "cublasDgemm(h); cublasDgemm(other);", category="diagnostic")
    assert audit_vendor_boundaries(tmp_path)["errors"]


def test_classification_requires_a_contract(tmp_path: Path) -> None:
    _fixture(tmp_path, "cublasDgemm(h);")
    manifest = tmp_path / MANIFEST
    data = json.loads(manifest.read_text())
    data["files"]["src/cc/example.cu"].pop("contract")
    manifest.write_text(json.dumps(data))
    assert (
        "contract and reason are required"
        in audit_vendor_boundaries(tmp_path)["errors"][0]
    )
