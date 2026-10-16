"""Host-only guards for CUDA placement and shared integral AD source generation."""

import ctypes as ct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from generativeqc._cuda_hessian_execution import CudaHessianExecution
from generativeqc_compiler.common.array_execution import (
    array_execution,
    arrays,
    current_execution,
)
from generativeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from generativeqc_compiler.common.native_runtime import compile_runtime
from generativeqc_compiler.common.paths import asset_path
from generativeqc_compiler.integral.array_graph_cuda import emit_array_graph
from generativeqc_compiler.integral.expr import Graph
from generativeqc_compiler.integral.resident_second_cuda import emit_resident_second
from generativeqc_compiler.integral.second_derivatives import (
    build_one_electron_second_ir,
    build_second_derivative_kernel,
)
from generativeqc_compiler.integral.second_derivatives_native import (
    emit_second_derivative_primitive,
)


def test_array_execution_is_scoped_and_thread_local() -> None:
    sentinel = SimpleNamespace(namespace=SimpleNamespace(placement="device"))
    with array_execution(sentinel):
        assert arrays.placement == "device"
        with ThreadPoolExecutor(max_workers=1) as executor:
            assert executor.submit(current_execution).result() is None
        with pytest.raises(RuntimeError), array_execution(None):
            assert current_execution() is None
            raise RuntimeError("restore on failure")
        assert current_execution() is sentinel
    assert current_execution() is None
    assert arrays.asarray([1]).shape == (1,)


def test_cuda_hessian_missing_array_dependency_does_not_fall_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "cupy", None)
    with pytest.raises(RuntimeError, match="does not fall back"):
        CudaHessianExecution(None, device_id=0)


def test_scalar_cuda_graph_preserves_lazy_piecewise_lowering() -> None:
    graph = Graph()
    variable = graph.variable("value")
    root = graph.select_le(
        variable,
        graph.constant(0),
        graph.constant(0),
        graph.stable_unary("log", variable),
    )
    source = emit_array_graph(graph, (root,), ("value",))
    assert "if (" in source
    assert "cudaStream_t stream" in source
    assert "atomicExch(status, 1)" in source


def test_resident_integral_emits_value_gradient_and_hvp_from_one_graph(
    tmp_path: Path,
) -> None:
    integral = build_one_electron_second_ir("overlap", (0, 0), output="weighted_hvp")
    kernel = build_second_derivative_kernel(integral, (0,))
    source = emit_second_derivative_primitive(
        kernel, backend="cpu", include_first_order=True
    )
    source += """
extern "C" int probe(const double* exponents, const double* centers,
    const double* direction, double* output) {
  double unit=1;
  return generativeqc::scf::generated_second::primitive(exponents,centers,&unit,direction,output) ? 0 : 1;
}
"""
    path = tmp_path / "primitive.cpp"
    path.write_text(source)
    headers = tuple(
        asset_path("src/integrals/" + name)
        for name in ("eri_geometry.hpp", "range_moments.hpp")
    )
    artifact = compile_runtime(
        CppCompilerAdapter(Path("c++")),
        tmp_path / "cache",
        path,
        headers=headers,
        options=("-I" + str(headers[0].parents[1]),),
    )
    library = ct.CDLL(str(artifact.library))
    function = library.probe
    function.argtypes, function.restype = [ct.c_void_p] * 4, ct.c_int
    exponents = np.array([0.7, 1.1])
    centers = np.array([[0.1, -0.3, 0.2], [0.4, 0.2, 1.3]])
    direction = np.array([[0.04, -0.03, 0.02], [-0.01, 0.02, -0.03]])
    output = np.empty(13)
    assert (
        function(
            exponents.ctypes.data,
            centers.ctypes.data,
            direction.ctypes.data,
            output.ctypes.data,
        )
        == 0
    )
    reduced = np.prod(exponents) / exponents.sum()
    delta, moving = centers[0] - centers[1], direction[0] - direction[1]
    value = (np.pi / exponents.sum()) ** 1.5 * np.exp(-reduced * delta @ delta)
    gradient = -2 * reduced * value * delta
    hvp = value * (-2 * reduced * moving + 4 * reduced**2 * delta * (delta @ moving))
    np.testing.assert_allclose(
        output,
        np.concatenate(([value], gradient, -gradient, hvp, -hvp)),
        atol=2e-14,
        rtol=2e-14,
    )
    device_source = emit_resident_second(integral, (0,))
    assert "double result[13]" in device_source
    assert "molecular_direction[3*atoms[center]+axis]" in device_source
