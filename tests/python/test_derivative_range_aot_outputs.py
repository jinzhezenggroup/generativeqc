"""Range AOT output identities cannot overwrite another radial program."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from tools import generate_derivative_range_aot as generator

if TYPE_CHECKING:
    from pathlib import Path

    from generativeqc_compiler.integral.range_separation import CoulombKernel


def _arguments(tmp_path: Path, entries: list[dict[str, object]]) -> list[str]:
    manifest = tmp_path / "radials.json"
    manifest.write_text(
        json.dumps(
            {"schema": "generativeqc.derivative-aot.radials.v1", "entries": entries}
        ),
        encoding="utf-8",
    )
    return [
        "generate",
        "--radial-manifest",
        str(manifest),
        "--output-directory",
        str(tmp_path / "output"),
    ]


@pytest.mark.parametrize("family", ["short_range", "long_range"])
@pytest.mark.parametrize("reverse", [False, True])
def test_colliding_radials_fail_before_generation_or_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    family: str,
    reverse: bool,
) -> None:
    entries = [
        {"backend": "cpu", "family": family, "omega": omega} for omega in (0.3, 0.4)
    ]
    if reverse:
        entries.reverse()
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "retained.cpp"
    sentinel.write_text("retained source\n", encoding="utf-8")
    emit = Mock(return_value=("unexpected overwritten source", (0,), "entry"))
    monkeypatch.setattr(generator, "program_source", emit)
    monkeypatch.setattr(sys, "argv", _arguments(tmp_path, entries))

    with pytest.raises(RuntimeError, match="one omega per radial family"):
        generator.main()

    emit.assert_not_called()
    assert list(output.iterdir()) == [sentinel]
    assert sentinel.read_text(encoding="utf-8") == "retained source\n"


@pytest.mark.parametrize("omegas", [(0.3, 0.3), (0.4, 0.7)])
def test_distinct_families_preserve_all_radial_output_contents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    omegas: tuple[float, float],
) -> None:
    entries = [
        {"backend": "cpu", "family": family, "omega": omega}
        for family, omega in zip(("short_range", "long_range"), omegas, strict=True)
    ]

    def emit(
        radial: CoulombKernel,
        angular: tuple[int, int, int, int],
        group_index: int,
    ) -> tuple[str, tuple[int, ...], str]:
        return (
            json.dumps(
                {"omega": radial.omega, "angular": angular, "group": group_index}
            ),
            generator.component_groups(angular)[group_index],
            "entry",
        )

    monkeypatch.setattr(generator, "program_source", emit)
    monkeypatch.setattr(sys, "argv", _arguments(tmp_path, entries))
    generator.main()

    files = tuple((tmp_path / "output").glob("*.cpp"))
    assert len(files) == 34
    for tag, omega in zip(("sr", "lr"), omegas, strict=True):
        selected = tuple(path for path in files if f"range_{tag}_" in path.name)
        assert len(selected) == 17
        assert all(
            json.loads(path.read_text(encoding="utf-8"))["omega"] == omega
            for path in selected
        )
