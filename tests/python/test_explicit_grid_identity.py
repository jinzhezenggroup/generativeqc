"""Versioned exact-grid identity without Python-object quadrature expansion."""

import json
import typing

import numpy as np
import pytest
from generativeqc_compiler.common.provenance import canonical_hash
from generativeqc_compiler.dft.grid import ExplicitGrid


def make_grid(
    *,
    version: int = 2,
    points: typing.Any = None,
    weights: typing.Any = None,
    owners: typing.Any = None,
    provenance: typing.Any = None,
) -> ExplicitGrid:
    """Build unequal finite arrays, including signed zero and frozen metadata."""
    return ExplicitGrid(
        np.arange(12, dtype=float).reshape(4, 3) if points is None else points,
        np.arange(1, 5, dtype=float) if weights is None else weights,
        (0, 0, 1, 1) if owners is None else owners,
        {"source": "test", "nested": {"epoch": 7}}
        if provenance is None
        else provenance,
        identity_version=version,
    )


def test_typed_construction_does_not_materialize_interchange(
    monkeypatch: typing.Any,
) -> None:
    def forbidden_record(self: ExplicitGrid) -> typing.Any:
        raise AssertionError("typed identity must not materialize a JSON grid")

    monkeypatch.setattr(ExplicitGrid, "record", forbidden_record)
    first = make_grid()
    assert len(first.identity) == 64
    assert make_grid().identity == first.identity


@pytest.mark.parametrize("version", [1, 2])
def test_roundtrip_and_content_tampering(tmp_path: typing.Any, version: int) -> None:
    grid = make_grid(version=version)
    path = tmp_path / "grid.json"
    grid.write(path)
    record = json.loads(path.read_text())
    assert record["version"] == version
    assert ExplicitGrid.read(path).identity == grid.identity
    record["weights_bohr3"][0] += 1
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="hash mismatch"):
        ExplicitGrid.read(path)


def test_legacy_identity_remains_canonical_json_and_migration_is_explicit() -> None:
    legacy = make_grid(version=1)
    assert legacy.identity == canonical_hash(legacy.record())
    assert legacy.identity != make_grid(version=2).identity


def test_typed_v2_digest_is_a_stable_portable_contract() -> None:
    """Independent whole-buffer encoding of the documented v2 metadata/payload."""
    assert make_grid().identity == (
        "4833bdb5b125e11031658df322ae8aebfc03d0f3bb60906126459cd741d07b48"
    )


@pytest.mark.parametrize("version", [1, 2])
def test_interchange_version_cannot_be_relabelled(
    tmp_path: typing.Any, version: int
) -> None:
    grid = make_grid(version=version)
    path = tmp_path / "grid.json"
    grid.write(path)
    record = json.loads(path.read_text())
    record["version"] = 3 - version
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="hash mismatch"):
        ExplicitGrid.read(path)


@pytest.mark.parametrize(
    "component", ["points", "weights", "owners", "provenance", "shape", "signed-zero"]
)
def test_typed_identity_covers_every_scientific_input(component: str) -> None:
    original = make_grid()
    points, weights = original.points.copy(), original.weights.copy()
    owners, provenance = original.owners, {"source": "test", "nested": {"epoch": 7}}
    if component == "points":
        points[0, 0] = np.nextafter(0.0, 1.0)
    elif component == "weights":
        weights[-1] = np.nextafter(weights[-1], 0.0)
    elif component == "owners":
        owners = (0, 0, 1, 2)
    elif component == "provenance":
        provenance["nested"]["epoch"] = 8
    elif component == "shape":
        points, weights, owners = points[:-1], weights[:-1], owners[:-1]
    else:
        points[0, 0] = -0.0
    changed = make_grid(
        points=points, weights=weights, owners=owners, provenance=provenance
    )
    assert changed.identity != original.identity


@pytest.mark.parametrize("dtype", ["<f8", ">f8", "<f4", ">f4", "<i4"])
def test_typed_identity_normalizes_dtype_endianness_and_layout(dtype: str) -> None:
    reference = make_grid()
    points = np.asfortranarray(reference.points.astype(dtype))
    weights = reference.weights.astype(dtype)
    assert make_grid(points=points, weights=weights).identity == reference.identity


def test_typed_grid_owns_buffers_and_freezes_nested_provenance() -> None:
    points = np.arange(12, dtype=float).reshape(4, 3)
    weights = np.arange(1, 5, dtype=float)
    provenance = {"source": "test", "nested": {"epoch": 7}}
    grid = make_grid(points=points, weights=weights, provenance=provenance)
    identity, record = grid.identity, grid.record()
    points[:] = -1
    weights[:] = -1
    provenance["nested"]["epoch"] = 8
    assert grid.identity == identity and grid.record() == record
    with pytest.raises(ValueError):
        grid.points.setflags(write=True)
    with pytest.raises(ValueError):
        grid.weights.setflags(write=True)


@pytest.mark.parametrize("version", [0, 3, True, 2.0, "2"])
def test_unknown_or_ambiguous_identity_versions_fail_closed(
    version: typing.Any,
) -> None:
    with pytest.raises(ValueError, match="identity version"):
        make_grid(version=version)


@pytest.mark.parametrize("owner", [-1, 2**31, False, 0.5, np.int64(0)])
def test_typed_owner_encoding_rejects_truncation_and_overflow(
    owner: typing.Any,
) -> None:
    with pytest.raises(ValueError, match="point owner"):
        make_grid(owners=(0, 0, 1, owner))


def test_typed_owner_encoding_accepts_the_full_original_integer_domain() -> None:
    first = make_grid(owners=(0, 0, 1, 2**31 - 1))
    assert first.identity != make_grid().identity


def test_chunk_boundaries_do_not_change_typed_content_identity() -> None:
    points = np.arange(3 * 140_000, dtype=float).reshape(-1, 3)
    weights = np.arange(len(points), dtype=float)
    owners = (0,) * len(points)
    first = make_grid(points=points, weights=weights, owners=owners)
    second = make_grid(
        points=points[::-1][::-1], weights=weights[::-1][::-1], owners=owners
    )
    assert first.identity == second.identity
    points[-1, -1] += 1
    assert (
        make_grid(points=points, weights=weights, owners=owners).identity
        != first.identity
    )
