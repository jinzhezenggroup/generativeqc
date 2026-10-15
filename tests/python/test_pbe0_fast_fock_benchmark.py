"""Budget-Fock timing claims require numerical and executed-arithmetic gates."""

import copy
import json
import math

import pytest

from benchmarks.pbe0_fast_fock import check_sample, portable_report


def test_rejected_nonfinite_diagnostic_remains_portable_and_observable() -> None:
    report = {"records": [{"energy": -1.0, "history": [{"hartree": float("nan")}]}]}
    portable = portable_report(report)
    assert portable["records"][0]["history"][0]["hartree"] is None
    assert portable["nonfinite_values"] == {
        "$report.records[0].history[0].hartree": "nan"
    }
    assert math.isnan(report["records"][0]["history"][0]["hartree"])
    json.dumps(portable, allow_nan=False)
    assert portable_report({"energy": -1.0}) == {"energy": -1.0}


def test_fast_claim_requires_fp32_exchange_and_one_endpoint() -> None:
    oracle = {"energy": -1.0, "forces": [[0.0, 0.0, 0.0]]}
    row = {
        **oracle,
        "precision_validation": {"status": "qualified_budget_fock", "solves": [{}]},
        "precision": {
            "operators": [
                {"name": name, "compute": "fp32", "accumulation": "fp64", "count": 3}
                for name in ("coulomb_j", "exchange_k", "coulomb_recurrence")
            ]
        },
    }
    assert check_sample(row, oracle, "fast")["energy_Eh"] == 0.0
    for defect in (
        "strict_k",
        "empty_k",
        "extra_solve",
        "force_error",
        "energy_error",
        "nan",
    ):
        invalid = copy.deepcopy(row)
        if defect == "strict_k":
            invalid["precision"]["operators"][1]["compute"] = "fp64"
        elif defect == "empty_k":
            invalid["precision"]["operators"][1]["count"] = 0
        elif defect == "extra_solve":
            invalid["precision_validation"]["solves"].append({})
        elif defect == "force_error":
            invalid["forces"][0][0] = 1e-3
        elif defect == "energy_error":
            invalid["energy"] += 1e-3
        else:
            invalid["energy"] = float("nan")
        with pytest.raises((RuntimeError, ValueError)):
            check_sample(invalid, oracle, "fast")
