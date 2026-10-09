"""Current OH comparison policy over independently captured CPU evidence."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from tools.oh_uhf_policy import POLICY_ID, evaluate_oh_density
from tools.oh_uhf_symmetry import raw_comparison


def _evidence() -> dict:
    return json.loads(
        (Path(__file__).parents[1] / "data/oh_uhf_symmetry_1791.json").read_text()
    )


@pytest.mark.parametrize("index", range(4))
def test_actual_cpu_states_admit_equivalence_without_promoting_raw(index: int) -> None:
    row = _evidence()["rows"][index]
    inputs = copy.deepcopy(row["certificate_inputs"])
    before = copy.deepcopy(inputs)
    result = evaluate_oh_density(**inputs)
    assert result["policy_id"] == POLICY_ID
    assert result["equivalence_decision"] == "ACCEPTED_EQUIVALENT"
    assert result["canonical_determinant_required"] is False
    assert result["raw"]["status"] == row["diagnostic"]["raw"]["status"] == "FAIL"
    assert result["raw"]["tolerance"] == 1e-7
    assert result["symmetry"]["status"] == "PASS"
    assert result["production_density_modified"] is False
    assert inputs == before
    # Retained capture text remains a historical statement, not current policy.
    assert "unresolved" in row["diagnostic"]["canonical_determinant_policy"]


@pytest.mark.parametrize("index", range(4))
def test_same_state_passes_raw_and_equivalence(index: int) -> None:
    inputs = copy.deepcopy(_evidence()["rows"][index]["certificate_inputs"])
    inputs["reference"] = copy.deepcopy(inputs["density"])
    inputs["reference_energy"] = inputs["endpoint_energy"]
    inputs["angle"] = 0.0
    result = evaluate_oh_density(**inputs)
    assert result["raw"]["status"] == "PASS"
    assert result["equivalence_decision"] == "ACCEPTED_EQUIVALENT"


@pytest.mark.parametrize("index", range(4))
@pytest.mark.parametrize("mutation", ["hamiltonian", "density", "energy"])
def test_retained_pass_cannot_override_new_tensor_failure(
    index: int, mutation: str
) -> None:
    row = _evidence()["rows"][index]
    assert row["diagnostic"]["symmetry"]["status"] == "PASS"
    inputs = copy.deepcopy(row["certificate_inputs"])
    if mutation == "hamiltonian":
        inputs["hcore"][2][2] += 1e-3
    elif mutation == "density":
        inputs["density"][1][2][2] += 1e-3
    else:
        inputs["endpoint_energy"] += 1e-3
    result = evaluate_oh_density(**inputs)
    assert result["equivalence_decision"] == "REJECTED"
    assert result["symmetry"]["status"] == "FAIL"


def test_saved_certificate_is_not_a_policy_input() -> None:
    with pytest.raises(TypeError):
        evaluate_oh_density(**_evidence()["rows"][0]["diagnostic"])


def test_nondegenerate_fixture_remains_outside_policy() -> None:
    evidence = _evidence()
    h2 = evidence["nondegenerate_fixture"]
    assert h2["orbital_gap_hartree"] > 1e-3
    d = np.asarray(h2["density"])
    assert raw_comparison(d, d, h2["overlap"])["status"] == "PASS"
    inputs = copy.deepcopy(evidence["rows"][0]["certificate_inputs"])
    inputs.update(density=d, reference=d, overlap=h2["overlap"])
    with pytest.raises(ValueError, match="shape"):
        evaluate_oh_density(**inputs)


def test_unsupported_geometry_and_nonfinite_inputs_fail_closed() -> None:
    inputs = copy.deepcopy(_evidence()["rows"][0]["certificate_inputs"])
    inputs["coordinates"][1][2] = 2.0
    with pytest.raises(ValueError, match="only supports"):
        evaluate_oh_density(**inputs)
    inputs = copy.deepcopy(_evidence()["rows"][0]["certificate_inputs"])
    inputs["density"][0][0][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        evaluate_oh_density(**inputs)
