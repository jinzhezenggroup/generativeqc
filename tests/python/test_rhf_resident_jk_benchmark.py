"""The action diagnostic consumes basis-only inputs without an orbital oracle."""

from pathlib import Path

import pytest

from benchmarks.rhf_resident_jk import read_input


@pytest.mark.parametrize("displacement", [0.0, 0.05])
def test_basis_input_preserves_atoms_shell_order_and_primitives(
    tmp_path: Path, displacement: float
) -> None:
    """Geometry changes move one coordinate and preserve both basis spaces."""
    text = """2 2 1 1073741824
8 0.0 0.1 -0.2
1 1.4 0.1 1.1
0 0 2
2.0 0.7
0.5 -0.2
1 3 1
0.8 1.0
0 1 1
0.4 1.0
"""
    path = tmp_path / "basis.input"
    path.write_text(text)
    actual = read_input(path, displacement)
    assert actual["inputs"]["atomic_numbers"] == [8, 1]
    assert actual["inputs"]["coordinates"] == [
        [displacement, 0.1, -0.2],
        [1.4, 0.1, 1.1],
    ]
    assert actual["inputs"]["basis_representation"] == "real_spherical"
    assert actual["inputs"]["shells"] == [
        {
            "atom_index": 0,
            "angular_momentum": 0,
            "primitives": [[2.0, 0.7], [0.5, -0.2]],
        },
        {"atom_index": 1, "angular_momentum": 3, "primitives": [[0.8, 1.0]]},
    ]
    assert actual["auxiliary_shells"] == [
        {"atom_index": 0, "angular_momentum": 1, "primitives": [[0.4, 1.0]]}
    ]
    assert read_input(path, 0.0)["inputs"]["coordinates"][0][0] == 0.0
    assert path.read_text() == text
