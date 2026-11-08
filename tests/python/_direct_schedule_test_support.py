"""Generate real Direct-HF policy inputs for host-only compiled probes."""

from pathlib import Path

from generativeqc_compiler.integral.direct_resident_schedule import (
    emit_direct_resident_psss_schedule_header,
)
from generativeqc_compiler.integral.one_electron_derivative_policy_cuda import (
    emit_one_electron_derivative_policy_cuda,
)


def write_direct_policy_headers(directory: Path) -> None:
    """Emit policy only, without CUDA mathematics, device probes or runtime imports."""
    (directory / "generated_direct_resident_psss_schedule.cuh").write_text(
        emit_direct_resident_psss_schedule_header(), encoding="utf-8"
    )
    (directory / "generated_one_electron_derivative_policy.cuh").write_text(
        emit_one_electron_derivative_policy_cuda(), encoding="utf-8"
    )
