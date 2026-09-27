"""Device-free guards for CUDA grid lease synchronization policy."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _selected_execution() -> str:
    source = (ROOT / "src/dft/cuda_grid.cu").read_text()
    return source.split("int grid_cuda_run_selected_v1", 1)[1].split(
        "int grid_cuda_run_v1", 1
    )[0]


def test_device_feature_lease_skips_detailed_section_fences() -> None:
    block = _selected_execution()
    assert (
        "!(npoint && features && !feature_output && !jet_output);" in block
    )
    output = block.index("ctx.section(true, ctx.metrics.output_ms")
    before_output = block[:output]
    assert "ctx.section(true," not in before_output
    assert before_output.count("ctx.section(detailed_profile,") == 9


def test_device_feature_lease_keeps_final_error_publication() -> None:
    block = _selected_execution()
    output = block.split("ctx.section(true, ctx.metrics.output_ms", 1)[1]
    assert "cudaMemcpyAsync(&failure, ctx.error" in output
    assert (
        'if (failure) throw std::runtime_error("nonfinite CUDA AO/density output")'
        in output
    )
