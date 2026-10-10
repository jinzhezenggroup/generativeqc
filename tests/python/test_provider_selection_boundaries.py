"""Provider implementation selectors must not spread into new owners."""

from __future__ import annotations

import json
import typing

import pytest

from tools.check_provider_selection_boundaries import (
    MANIFEST,
    audit_provider_selection_boundaries,
)

if typing.TYPE_CHECKING:
    from pathlib import Path


def _fixture(
    root: Path,
    source: str,
    *,
    path: str = "src/cc/example.cu",
    selector: str = "use_cublas",
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
                        "classification": "migration",
                        "contract": "#1890",
                        "reason": "Fixture migration debt.",
                        "selectors": {selector: 1},
                    }
                },
            }
        )
    )
    return file


def test_repository_provider_selection_inventory_is_complete() -> None:
    assert audit_provider_selection_boundaries()["errors"] == []


@pytest.mark.parametrize(
    "selector",
    [
        "reduction_provider",
        "matrix_gemm",
        "df_matrix_gemm",
        "df_replay_matrix_gemm",
        "lambda_matrix_gemm",
        "conventional_matrix_gemm",
        "use_cublas",
        "use_cublaslt",
        "use_cutensor",
        "use_cutlass",
        "use_cub",
        "use_cusolver",
        "use_cusparse",
        "use_nccl",
    ],
)
def test_new_native_selector_surfaces_require_classification(
    tmp_path: Path, selector: str
) -> None:
    _fixture(tmp_path, "bool use_cublas = true;")
    new = tmp_path / "src/methods/new.cu"
    new.parent.mkdir(parents=True)
    new.write_text(f"bool {selector} = false;")
    errors = audit_provider_selection_boundaries(tmp_path)["errors"]
    assert len(errors) == 1
    assert "unclassified provider-selection identifiers" in errors[0]


def test_same_selector_growth_and_retirement_both_fail(tmp_path: Path) -> None:
    file = _fixture(tmp_path, "bool use_cublas = true;")
    file.write_text("bool use_cublas = true; if (use_cublas) {}")
    assert (
        "count 2 != classified 1"
        in audit_provider_selection_boundaries(tmp_path)["errors"][0]
    )
    file.write_text("semantic_binding.execute();")
    assert (
        "remove retired" in audit_provider_selection_boundaries(tmp_path)["errors"][0]
    )


def test_native_comments_and_strings_are_not_selector_surfaces(tmp_path: Path) -> None:
    _fixture(
        tmp_path,
        """
// use_cutensor
/* use_cublaslt */
const char* diagnostic = "use_cusolver";
bool use_cublas = true;
""",
    )
    assert audit_provider_selection_boundaries(tmp_path)["errors"] == []


def test_python_schedule_keys_are_selector_surfaces(tmp_path: Path) -> None:
    _fixture(
        tmp_path,
        'schedule = {"reduction_provider": "generated"}',
        path="python/generativeqc_compiler/tensor/example.py",
        selector="reduction_provider",
    )
    assert audit_provider_selection_boundaries(tmp_path)["errors"] == []


def test_generator_fragments_cannot_hide_selector_surfaces(tmp_path: Path) -> None:
    _fixture(
        tmp_path,
        'return "bool use_cublas = true;"',
        path="tools/generate_new.py",
        selector="use_cublas",
    )
    assert audit_provider_selection_boundaries(tmp_path)["errors"] == []


def test_python_docstrings_do_not_create_selector_debt(tmp_path: Path) -> None:
    _fixture(
        tmp_path,
        '''"""Do not add use_cutensor switches."""
def choose():
    """reduction_provider is historical terminology."""
    return {"reduction_provider": "generated"}
''',
        path="python/generativeqc_compiler/tensor/example.py",
        selector="reduction_provider",
    )
    assert audit_provider_selection_boundaries(tmp_path)["errors"] == []


def test_classification_requires_contract_and_reason(tmp_path: Path) -> None:
    _fixture(tmp_path, "bool use_cublas = true;")
    manifest = tmp_path / MANIFEST
    data = json.loads(manifest.read_text())
    data["files"]["src/cc/example.cu"].pop("contract")
    manifest.write_text(json.dumps(data))
    assert (
        "classification, contract and reason are required"
        in audit_provider_selection_boundaries(tmp_path)["errors"][0]
    )
