from __future__ import annotations

import json
from pathlib import Path

from generativeqc import _generated_methods

ROOT = Path(__file__).resolve().parents[2]


def test_df_rccsdt_manifest_is_distinct_energy_only_method() -> None:
    payload = json.loads((ROOT / "manifests/public_methods.json").read_text())
    row = next(method for method in payload["methods"] if method["name"] == "df-rccsd(t)")
    assert row == {
        "name": "df-rccsd(t)",
        "symbol": "DF_RCCSD_T",
        "abi_id": 19,
        "family": "coupled_cluster",
        "provider": "df_rccsdt",
        "properties": ["energy"],
        "supports_batch": False,
        "aliases": ["df-ccsd(t)"],
    }
    assert _generated_methods.METHOD_NAME_TO_ID["df-rccsd(t)"] == 19
    assert _generated_methods.METHOD_NAME_TO_ID["df-ccsd(t)"] == 19


def test_df_rccsdt_public_owner_stays_force_fail_closed() -> None:
    source = (ROOT / "src/methods/df_rccsdt_method.cpp").read_text()
    assert "run_df_ccsdt_native(execution_, system_, auxiliary_, descriptor_, false)" in source
    assert "public DF-RCCSD(T) forces remain unqualified" in source
    assert "descriptor_.density_fitting_mode = GENERATIVEQC_DENSITY_FITTING_NONE" in source
    assert "descriptor_.density_fitting_auxiliary_basis = nullptr" in source
