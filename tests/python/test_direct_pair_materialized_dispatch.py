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
    assert (
        "separate_sources && materialized_pair_derivative_available(batch)" in dispatch
    )
    for enabled in ("true", "false"):
        assert (
            f"launch.template operator()<Unrestricted, Purpose, {enabled}>();"
            in dispatch
        )


def test_full_range_force_promotes_qualified_static_128_thread_schedule() -> None:
    """Keep #1978's measured CTA width as the generic full-range force default."""
    source = (ROOT / "src/scf/cuda/direct_bounded_fallback.cu").read_text()
    constants = (ROOT / "src/scf/cuda/direct_constants.hpp").read_text()
    begin = source.index("void launch_bounded_direct_shell_quartet_kernel_scaled(")
    end = source.index("void launch_bounded_direct_range_exchange_force_kernel(", begin)
    dispatch = source[begin:end]

    assert "constexpr unsigned kBoundedDirectForceThreads = 128;" in constants
    assert "Force ? blockDim.x : detail::kBoundedDirectQueueCapacity" in source
    assert "slot += blockDim.x / detail::kDirectQuartetThreads" in source
    assert "block.x == kBoundedDirectThreads" in dispatch
    assert "block.x = kBoundedDirectForceThreads;" in dispatch
    assert "GENERATIVEQC_EXPERIMENT_DIRECT_FORCE_CTA_THREADS" not in source
