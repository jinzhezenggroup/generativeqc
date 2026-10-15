"""Explicit AD API control discovery for the default-promotion inventory."""

from pathlib import Path

import pytest

from tools.check_default_promotion_inventory import _discover_tensor_ad


@pytest.mark.parametrize("default", ["False", "True"])
def test_boolean_api_controls_are_discovered(tmp_path: Path, default: str) -> None:
    relative = "python/generativeqc_compiler/tensor/ad_program.py"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_text(
        f"def linearize(*, allow_external_mask_where: bool = {default}, max_elements: int = 3):\n    pass\n",
        encoding="utf-8",
    )
    assert _discover_tensor_ad(tmp_path) == {
        "tensor-ad:linearize.allow_external_mask_where": relative
    }


def test_ambiguous_boolean_api_default_rejects(tmp_path: Path) -> None:
    path = tmp_path / "python/generativeqc_compiler/tensor/ad_program.py"
    path.parent.mkdir(parents=True)
    path.write_text(
        "def linearize(*, opt_in: bool = None):\n    pass\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Boolean default"):
        _discover_tensor_ad(tmp_path)
