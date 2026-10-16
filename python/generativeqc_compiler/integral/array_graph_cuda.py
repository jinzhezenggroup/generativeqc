"""CUDA array lowering of the shared scalar graph, including lazy branches."""

from typing import Any

from .cuda import CudaEmitter


def emit_array_graph(graph: Any, roots: tuple, names: tuple[str, ...]) -> str:
    """Emit a shape-independent FP64 lane program with caller-owned storage."""
    emitter = CudaEmitter(
        graph, {name: f"inputs[{index}*size+lane]" for index, name in enumerate(names)}
    )
    emitter.emit(roots)
    stores = "\n".join(
        f"outputs[{index}*size+lane]={emitter.reference(root)};"
        f"if (!isfinite(outputs[{index}*size+lane])) atomicExch(status, 1);"
        for index, root in enumerate(roots)
    )
    return (
        "#include <cuda_runtime.h>\n#include <cmath>\n"
        "__global__ void array_graph(const double* inputs, size_t size, double* outputs, int* status) {\n"
        "for (size_t lane=size_t(blockIdx.x)*blockDim.x+threadIdx.x; lane<size; "
        "lane+=size_t(blockDim.x)*gridDim.x) {\n"
        + "\n".join(emitter.lines)
        + "\n"
        + stores
        + "\n}}\n"
        'extern "C" int generativeqc_array_graph(const double* inputs, size_t size, '
        "double* outputs, int* status, cudaStream_t stream) {\n"
        "if (size) array_graph<<<(size+127)/128,128,0,stream>>>(inputs,size,outputs,status);\n"
        "return static_cast<int>(cudaGetLastError());\n}\n"
    )
