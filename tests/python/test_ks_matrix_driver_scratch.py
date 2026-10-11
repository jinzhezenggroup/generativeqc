"""Source-level driver scratch wiring; not CUDA/Graph device qualification."""

from pathlib import Path

import pytest

DRIVER = Path(__file__).resolve().parents[1] / "native/test_ks_matrix_provider_cuda.cu"


def _body(source: str, marker: str) -> str:
    begin = source.index("{", source.index(marker)) + 1
    end, depth = begin, 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[begin : end - 1]


def _compact(source: str) -> str:
    return "".join(source.split())


def test_driver_owns_exact_disjoint_scratch_outside_enqueue() -> None:
    source = DRIVER.read_text()
    case = _body(source, "void run_case(")
    allocation = "masked_output(output_size)"
    prefix = case[: case.index(allocation)]
    # Function scope keeps the independent cudaMalloc alive until run_case exits.
    assert prefix.count("{") == prefix.count("}")
    assert "DeviceBuffer<double>" in prefix.split(";")[-1]
    compact = _compact(case)
    assert "conststd::size_tmatrix_size=static_cast<std::size_t>(n)*n;" in compact
    assert "conststd::size_toutput_size=batch*spins*matrix_size;" in compact
    assert "dout(output_size),masked_output(output_size);" in compact
    assert case.index(allocation) < case.index("cudaEventCreate")
    assert case.index(allocation) < case.index("const auto enqueue")
    assert case.index(allocation) < case.index("cudaStreamBeginCapture")
    # No reassignment, explicit free, aliasing or additional scratch use can hide
    # between the allocation and the two provider views checked below.
    assert case.count("masked_output") == 3
    assert "cudaFree" not in case
    buffer = _compact(_body(source, "struct DeviceBuffer"))
    assert "check(cudaMalloc(&data,count*sizeof(T)))" in buffer
    assert "~DeviceBuffer(){if(data)(void)cudaFree(data);}" in buffer


@pytest.mark.parametrize("ordinary", [True, False])
def test_both_driver_calls_pass_exact_scratch_span(ordinary: bool) -> None:
    case = _body(DRIVER.read_text(), "void run_case(")
    enqueue = _compact(_body(case, "const auto enqueue"))
    if ordinary:
        expected = (
            "matrix::launch_matrix_product("
            "owner.view(masked_output.data,output_size),batch,n,dl.data,"
            "transpose,dr.data,da.data,dout.data,library,scale)"
        )
    else:
        expected = (
            "matrix::launch_spin_matrix_product("
            "owner.view(masked_output.data,output_size),batch,spins,n,dl.data,"
            "left_spin,transpose,dr.data,right_spin,da.data,dout.data,library)"
        )
    assert expected in enqueue
    assert enqueue.count("owner.view(masked_output.data,output_size)") == 2
    assert "owner.view()" not in enqueue


def test_graph_and_ordinary_replays_share_scratch_until_synchronized_cleanup() -> None:
    case = _body(DRIVER.read_text(), "void run_case(")
    capture = _compact(_body(case, "if (graph_mode)"))
    assert "cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal)" in capture
    assert capture.index("cudaStreamBeginCapture") < capture.index("enqueue();")
    assert capture.index("enqueue();") < capture.index("cudaStreamEndCapture")
    replay = _compact(_body(case, "for (int repeat"))
    assert "repeat < 2" in case
    assert (
        "if(graph_mode)check(cudaGraphLaunch(executable,stream));elseenqueue();"
        in replay
    )
    assert replay.index("cudaEventSynchronize(end)") < replay.index("cudaMemcpyAsync")
    assert replay.index("cudaMemcpyAsync") < replay.index(
        "cudaStreamSynchronize(stream)"
    )
    cleanup = case[case.index("if (executable)") :]
    assert cleanup.index("cudaGraphExecDestroy") < cleanup.index("cudaGraphDestroy")
    assert cleanup.index("cudaGraphDestroy") < cleanup.index("cudaEventDestroy")
    assert case.index("for (int repeat") < case.index("if (executable)")
