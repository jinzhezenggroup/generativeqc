"""Offline oracle acceptance must never turn absent evidence into a skip."""

import json

import pytest

from tools.check_native_stationary_cuda_oracles import EXPECTED_CASES, read_receipts


def receipts() -> list[dict]:
    return [
        {
            "profile": profile,
            "fixture": fixture,
            "atoms": atoms,
            "aos": 6 if fixture == "lih_sp" else atoms,
            "energy": -0.5,
            "gradient": [0.0] * (3 * atoms),
            "density": [
                [0.0] * (6 if fixture == "lih_sp" else atoms) ** 2
                for _ in range(1 if profile.endswith("rks") else 2)
            ],
            "weighted_density": [
                [0.0] * (6 if fixture == "lih_sp" else atoms) ** 2
                for _ in range(1 if profile.endswith("rks") else 2)
            ],
        }
        for profile, fixture, atoms in sorted(EXPECTED_CASES)
    ]


def log(records: list[dict]) -> str:
    return "\n".join(
        "11: native_stationary_oracle " + json.dumps(record) for record in records
    )


def test_complete_native_oracle_inventory_is_required() -> None:
    records = receipts()
    assert read_receipts(log(records)) == records
    for incomplete in (records[:-1], records[1:], []):
        with pytest.raises(ValueError, match="incomplete"):
            read_receipts(log(incomplete))
    with pytest.raises(ValueError, match="duplicate"):
        read_receipts(log(records + records[:1]))


@pytest.mark.parametrize(
    "field",
    (
        "profile",
        "fixture",
        "atoms",
        "aos",
        "energy",
        "gradient",
        "density",
        "weighted_density",
    ),
)
def test_missing_native_oracle_field_fails_closed(field: str) -> None:
    records = receipts()
    del records[0][field]
    with pytest.raises(ValueError, match="fields"):
        read_receipts(log(records))


@pytest.mark.parametrize(
    "corruption",
    (
        "profile",
        "fixture",
        "shape",
        "gradient",
        "energy",
        "atoms",
        "aos",
        "aos_type",
        "density_shape",
        "density",
        "weighted_shape",
        "weighted_density",
    ),
)
def test_invalid_native_oracle_receipt_fails_closed(corruption: str) -> None:
    records = receipts()
    if corruption == "profile":
        records[0]["profile"] = "unqualified"
    elif corruption == "fixture":
        records[0]["fixture"] = "unqualified"
    elif corruption == "shape":
        records[0]["gradient"] = [0.0]
    elif corruption == "gradient":
        records[0]["gradient"][0] = float("nan")
    elif corruption == "energy":
        records[0]["energy"] = float("inf")
    elif corruption == "aos":
        records[0]["aos"] += 1
    elif corruption == "aos_type":
        records[0]["aos"] = True
    elif corruption == "density_shape":
        records[0]["density"] = [[0.0]]
    elif corruption == "density":
        records[0]["density"][0][0] = float("nan")
    elif corruption == "weighted_shape":
        records[0]["weighted_density"] = [[0.0]]
    elif corruption == "weighted_density":
        records[0]["weighted_density"][0][0] = float("inf")
    else:
        records[0]["atoms"] = True
    with pytest.raises(ValueError):
        read_receipts(log(records))
