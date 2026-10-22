"""Frozen campaign inputs fail closed before native execution."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from benchmarks.cc_cupti_campaign import INPUTS, frozen_inputs


def specification(root: Path) -> dict:
    spec = {"sha256": {}}
    for name in INPUTS:
        path = root / name
        path.write_bytes(name.encode())
        spec[name] = str(path.resolve())
        spec["sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return spec


@pytest.mark.parametrize("name", INPUTS)
def test_changed_input_bytes_are_rejected(tmp_path: Path, name: str) -> None:
    spec = specification(tmp_path)
    frozen_inputs(spec)
    Path(spec[name]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="input changed"):
        frozen_inputs(spec)


@pytest.mark.parametrize("name", INPUTS)
def test_same_bytes_replacement_changes_the_frozen_identity(
    tmp_path: Path, name: str
) -> None:
    spec = specification(tmp_path)
    before = frozen_inputs(spec)
    path = Path(spec[name])
    replacement = tmp_path / "replacement"
    replacement.write_bytes(path.read_bytes())
    replacement.replace(path)
    assert frozen_inputs(spec) != before


def test_directory_input_is_not_accepted_as_a_file(tmp_path: Path) -> None:
    spec = specification(tmp_path)
    path = Path(spec["backend"])
    path.unlink()
    path.mkdir()
    with pytest.raises(ValueError, match="absolute regular file"):
        frozen_inputs(spec)
