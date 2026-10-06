"""Guard optional pair-derivative dispatch independently of screening purpose."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_pair_derivative_selector_preserves_spin_screening_and_force_owner() -> None:
    """Both optional specializations write forces for either screening purpose."""
    source = (ROOT / "src/scf/cuda/direct_bounded_fallback.cu").read_text()
    begin = source.index("void launch_bounded_direct_shell_quartet_kernel_scaled(")
    end = source.index("void launch_bounded_direct_fock_shell_quartet_kernel(", begin)
    dispatch = source[begin:end]
    assert (
        "bounded_direct_shell_quartet_kernel<Unrestricted, Purpose, true, -1, -1, PairDerivatives>"
        in dispatch
    )
    for spin in ("true", "false"):
        for purpose in ("Fock", "Force"):
            assert (
                f"select.template operator()<{spin}, DirectScreeningPurpose::{purpose}>();"
                in dispatch
            )
    assert "if (materialized_pair_derivative_available(batch))" in dispatch
    for enabled in ("true", "false"):
        assert (
            f"launch.template operator()<Unrestricted, Purpose, {enabled}>();"
            in dispatch
        )
