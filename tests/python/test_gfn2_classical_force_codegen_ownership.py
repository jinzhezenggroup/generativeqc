from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = (ROOT / "src/xtb/native/src/backends/cuda/gfn2_classical_force.cu").read_text(
    encoding="utf-8"
)


def _between(start: str, end: str) -> str:
    assert start in SOURCE
    assert end in SOURCE
    return SOURCE.split(start, 1)[1].split(end, 1)[0]


def test_classical_repulsion_force_is_compiler_owned() -> None:
    assert '#include "generated_gfn2_pair_native.hpp"' in SOURCE
    body = _between(
        "__global__ void repulsion_gradient_kernel",
        "__global__ void es2_topology_preflight_kernel",
    )
    assert "evaluate_gfn2_repulsion_pair" in body
    assert "pair.distance_derivative" in body
    for forbidden in (
        "exp(",
        "repulsion_klight",
        "repulsion_kexp",
        "distance_power",
        "pair_energy",
    ):
        assert forbidden not in body


def test_classical_es2_force_is_compiler_owned() -> None:
    assert '#include "generated_gfn2_es2_native.hpp"' in SOURCE
    body = _between(
        "__global__ void es2_gradient_kernel",
        "__global__ void prepare_primitive_stage_kernel",
    )
    assert "evaluate_gfn2_es2_cached_gradient_weight" in body
    assert "project_gfn2_es2_gradient" in body
    assert "term *= kernel" not in body
    assert "sign * weighted * displacement" not in body
