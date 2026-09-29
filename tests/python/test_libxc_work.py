from __future__ import annotations

import pytest
from generativeqc_compiler.xc.libxc_maple import MapleImportError
from generativeqc_compiler.xc.libxc_work import (
    LIBXC_WORK_DOMAIN,
    LIBXC_WORK_DOMAIN_VERSION,
    automatic_work_policy,
    polarized_work_setup,
)


def test_gga_shared_boundary_cases_use_one_generic_work_policy() -> None:
    for name in ("GGA_C_AM05", "GGA_X_AK13"):
        policy = automatic_work_policy(name)
        lines, arguments = polarized_work_setup(
            policy,
            ("rho_a", "rho_b", "sigma_aa", "sigma_ab", "sigma_bb"),
        )
        source = "\n".join(lines)

        assert policy.family == "gga"
        assert policy.density_threshold > 0.0
        assert policy.sigma_floor is not None and policy.sigma_floor > 0.0
        assert "work_rho_b = fmax" in source
        assert "work_sigma_aa = fmax" in source
        assert "sigma_average" in source
        assert arguments == (
            "work_rho_a",
            "work_rho_b",
            "work_sigma_aa",
            "work_sigma_ab",
            "work_sigma_bb",
        )


def test_mgga_work_policy_owns_tau_and_fhc_boundary() -> None:
    policy = automatic_work_policy("MGGA_X_LTA")
    lines, arguments = polarized_work_setup(
        policy,
        (
            "rho_a",
            "rho_b",
            "sigma_aa",
            "sigma_ab",
            "sigma_bb",
            "tau_a",
            "tau_b",
        ),
    )
    source = "\n".join(lines)

    assert policy.family == "mgga"
    assert policy.tau_floor == pytest.approx(1.0e-20)
    assert policy.enforce_fhc
    assert "work_tau_a = fmax" in source
    assert "8.0 * work_rho_a * work_tau_a" in source
    assert arguments[-2:] == ("work_tau_a", "work_tau_b")


def test_work_domain_has_distinct_native_identity() -> None:
    assert LIBXC_WORK_DOMAIN == "libxc-7.0/work-semilocal-v1"
    assert LIBXC_WORK_DOMAIN_VERSION == 4


def test_work_policy_rejects_wrong_feature_layout() -> None:
    policy = automatic_work_policy("GGA_X_AK13")
    with pytest.raises(MapleImportError, match="expected"):
        polarized_work_setup(policy, ("rho_a", "rho_b"))
