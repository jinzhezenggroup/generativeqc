"""Device-free guards for CUDA grid lease synchronization policy."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _selected_execution() -> str:
    source = (ROOT / "src/dft/cuda_grid.cu").read_text()
    return source.split("static int grid_cuda_run_selected_impl", 1)[1].split(
        "int grid_cuda_run_selected_v1", 1
    )[0]


def test_device_feature_lease_skips_detailed_section_fences() -> None:
    block = _selected_execution()
    assert (
        "!defer_error_to_consumer &&" in block
        and "!(npoint && features && !feature_output && !jet_output);" in block
    )
    output = block.index("ctx.section(true, ctx.metrics.output_ms")
    before_output = block[:output]
    assert "ctx.section(true," not in before_output
    assert before_output.count("ctx.section(detailed_profile,") == 9


def test_synchronous_device_feature_lease_keeps_final_error_publication() -> None:
    block = _selected_execution()
    output = block.split("ctx.section(true, ctx.metrics.output_ms", 1)[1]
    assert "cudaMemcpyAsync(&failure, ctx.error" in output
    assert (
        'if (failure) throw std::runtime_error("nonfinite CUDA AO/density output")'
        in output
    )


def test_deferred_feature_lease_hands_error_to_same_stream_consumer() -> None:
    source = (ROOT / "src/dft/cuda_grid.cu").read_text()
    block = _selected_execution()
    deferred = block.split("if (defer_error_to_consumer)", 1)[1].split(
        "int failure = 0;", 1
    )[0]
    assert "return;" in deferred
    assert "cudaMemcpyDeviceToHost" not in deferred
    assert (
        "deferred CUDA grid errors require a device-only local feature lease" in block
    )
    wrapper = source.split("int grid_cuda_run_selected_deferred_v1", 1)[1].split(
        "int grid_cuda_run_v1", 1
    )[0]
    assert "grid_cuda_run_selected_impl" in wrapper
    assert "jet_output, 1, error, size" in wrapper


def test_stationary_consumer_explicitly_owns_deferred_error_gate() -> None:
    grid = (ROOT / "python/vibeqc_compiler/dft/cuda.py").read_text()
    stationary = (ROOT / "python/vibeqc/_stationary_cuda.py").read_text()
    assert "grid_cuda_run_selected_deferred_v1" in grid
    assert "_defer_error_to_consumer=defer_error_to_consumer" in grid
    assert "defer_error_to_consumer=True" in stationary
