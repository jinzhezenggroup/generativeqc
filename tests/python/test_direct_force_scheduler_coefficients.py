"""Guard the method-neutral native Direct force coefficient routing."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_native_force_contractions_expose_scaled_entry_points() -> None:
    expected = {
        "src/scf/cuda/direct_force_low_order.cuh": (
            "contract_two_electron_force_ssss_task_scaled",
            "contract_two_electron_force_psss_task_scaled",
        ),
        "src/scf/cuda/direct_force_order2.cuh": (
            "contract_two_electron_force_psps_task_scaled",
            "contract_two_electron_force_pair_order2_task_scaled",
        ),
        "src/scf/cuda/direct_force_order3.cuh": (
            "contract_two_electron_force_order3_task_scaled",
        ),
        "src/scf/cuda/direct_force_quartet.cuh": (
            "contract_two_electron_force_quartet_subtile_scaled",
        ),
    }
    for path, symbols in expected.items():
        source = _source(path)
        for symbol in symbols:
            assert symbol in source
        assert "direct_force_density_coefficient_scaled<Unrestricted>" in source


def test_bounded_native_launchers_forward_explicit_force_coefficients() -> None:
    expected = {
        "src/scf/cuda/direct_bounded_exact_force.hpp": "launch_contract_bounded_exact_low_order_force_page_kernel_scaled",
        "src/scf/cuda/direct_bounded_fallback.hpp": "launch_bounded_direct_shell_quartet_kernel_scaled",
        "src/scf/cuda/direct_bounded_dddd.hpp": "launch_bounded_direct_dddd_streaming_kernel_scaled",
    }
    for path, symbol in expected.items():
        source = _source(path)
        assert symbol in source
        assert "double coulomb_coefficient" in source
        assert "double exchange_coefficient" in source


def test_hf_compatibility_wrappers_keep_historical_coefficients() -> None:
    for path in (
        "src/scf/cuda/direct_force_low_order.cuh",
        "src/scf/cuda/direct_force_order2.cuh",
        "src/scf/cuda/direct_force_order3.cuh",
        "src/scf/cuda/direct_force_quartet.cuh",
    ):
        source = _source(path)
        assert "Unrestricted ? -1.0 : -0.5" in source
        assert ", 1.0," in source


def test_top_level_shell_force_dispatch_forwards_explicit_coefficients() -> None:
    header = _source("src/scf/cuda/direct_angular_force.hpp")
    source = _source("src/scf/cuda/direct_angular_force.cu")
    for symbol in (
        "launch_two_electron_force_psss_resident_bra_kernel_scaled",
        "dispatch_angular_force_quartets_scaled",
    ):
        assert symbol in header
        assert symbol in source
    assert "double coulomb_coefficient" in header
    assert "double exchange_coefficient" in header
    for symbol in (
        "contract_two_electron_force_ssss_task_scaled",
        "contract_two_electron_force_psss_task_scaled",
        "contract_two_electron_force_psps_task_scaled",
        "contract_two_electron_force_pair_order2_task_scaled",
        "contract_two_electron_force_order3_task_scaled",
        "contract_two_electron_force_quartet_subtile_scaled",
    ):
        assert symbol in source
    assert "contract_two_electron_force_ssss_task<" not in source
    assert "contract_two_electron_force_psss_task<" not in source
    assert "contract_two_electron_force_psps_task<" not in source
    assert "contract_two_electron_force_pair_order2_task<" not in source
    assert "contract_two_electron_force_order3_task<" not in source
    assert "contract_two_electron_force_quartet_subtile<" not in source


def test_top_level_hf_shell_dispatch_pins_historical_coefficients() -> None:
    source = _source("src/scf/cuda/direct_angular_force.cu")
    assert source.count("unrestricted ? -1.0 : -0.5") >= 2
    assert "generated_shell_class_mask, 1.0, exchange_coefficient" in source
