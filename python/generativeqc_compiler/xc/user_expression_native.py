"""Standalone native CPU point/batch adapter for sealed LDA polynomials."""

from __future__ import annotations

import ctypes as ct
from hashlib import sha256
from pathlib import Path

import numpy as np

from generativeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from generativeqc_compiler.common.native_runtime import compile_runtime
from generativeqc_compiler.common.provenance import canonical_hash, file_hash
from generativeqc_compiler.integral.scalar_c import ScalarCEmitter

from .native import _cache_source
from .user_expression import FEATURES
from .user_expression_program import MAX_POINTS, UserExpressionProgram, validate_inputs

MAX_SOURCE_BYTES = 262144


def emit_native(program: UserExpressionProgram) -> tuple[str, dict]:
    """Lower canonical roots, with checks independent of the Python boundary."""
    if type(program) is not UserExpressionProgram:
        raise TypeError("native expression requires UserExpressionProgram")
    graph, roots = program.scalar_graph()
    emitter = ScalarCEmitter(
        graph, {key: f"input[{i} * npoint + point]" for i, key in enumerate(FEATURES)}
    )
    emitter.emit(roots)
    lines = [
        "// Generated bounded user LDA polynomial; no Libxc or SCF admission.",
        "#include <cmath>",
        "#include <cstddef>",
        'extern "C" int xc_user_expression(const double* input, size_t input_count, double* output, size_t output_count, size_t npoint) noexcept {',
        f"  if (npoint > {MAX_POINTS} || input_count != 2 * npoint || output_count != {len(roots)} * npoint) return -1;",
        "  if (!npoint) return 0;",
        "  if (!input || !output) return -1;",
        "  for (size_t i = 0; i < input_count; ++i)",
        "    if (!std::isfinite(input[i]) || input[i] < 0.0) return -2;",
        "  for (size_t point = 0; point < npoint; ++point) {",
        *emitter.lines,
    ]
    for row, root in enumerate(roots):
        lines.extend(
            [
                f"    output[{row} * npoint + point] = {emitter.reference(root)};",
                f"    if (!std::isfinite(output[{row} * npoint + point])) return {row + 1};",
            ]
        )
    lines.extend(["  }", "  return 0;", "}", ""])
    source = "\n".join(lines)
    if len(source.encode()) > MAX_SOURCE_BYTES:
        raise ValueError("native expression exceeds source budget")
    return source, {
        "schema": "generativeqc.xc-expression-native.v1",
        "program": program.to_payload(),
        "source_sha256": sha256(source.encode()).hexdigest(),
        "max_source_bytes": MAX_SOURCE_BYTES,
    }


class NativeUserExpression:
    """Verified compiled program; each call owns its input and output storage.

    The private native ABI may fill scratch before reporting overflow. No scratch
    buffer is published on failure and there is no caller output/donation API.
    Successful arrays have immutable byte storage independent of future calls.
    """

    def __init__(
        self,
        program: UserExpressionProgram,
        *,
        compiler: CppCompilerAdapter,
        cache: str | Path,
    ) -> None:
        if not isinstance(compiler, CppCompilerAdapter):
            raise TypeError("native expression requires CppCompilerAdapter")
        source, metadata = emit_native(program)
        self._program = program
        cache = Path(cache).resolve()
        directory = cache / "source" / canonical_hash(metadata)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "xc_user_expression.cpp"
        _cache_source(path, source)
        self.artifact = compile_runtime(
            compiler, cache, path, options=("-ffp-contract=off",)
        )
        if self.artifact.metadata["identity"]["source"] != metadata["source_sha256"]:
            raise ValueError("native expression source identity mismatch")
        if file_hash(self.artifact.library) != self.artifact.metadata["binary_sha256"]:
            raise ValueError("native expression binary identity mismatch")
        self._library = ct.CDLL(str(self.artifact.library))
        self._function = self._library.xc_user_expression
        self._function.argtypes = [
            ct.POINTER(ct.c_double),
            ct.c_size_t,
            ct.POINTER(ct.c_double),
            ct.c_size_t,
            ct.c_size_t,
        ]
        self._function.restype = ct.c_int

    @property
    def program(self) -> UserExpressionProgram:
        return self._program

    def evaluate(self, features: np.ndarray) -> np.ndarray:
        """Return [output, point] FP64 values, including zero-density derivatives."""
        x = validate_inputs(features)
        result = np.empty((len(self.program.outputs), x.shape[1]), dtype=np.float64)
        status = self._function(
            x.ctypes.data_as(ct.POINTER(ct.c_double)),
            x.size,
            result.ctypes.data_as(ct.POINTER(ct.c_double)),
            result.size,
            x.shape[1],
        )
        if status:
            raise ArithmeticError(
                f"native expression failed (status {status}); no result published"
            )
        return np.frombuffer(result.tobytes(), dtype=np.float64).reshape(result.shape)

    def point(self, *, rho_a: float, rho_b: float) -> np.ndarray:
        """Single-point endpoint with the declaration's Python scalar contract."""
        if any(type(v) not in (int, float) for v in (rho_a, rho_b)):
            raise TypeError("point inputs require Python FP64 floats or integers")
        try:
            x = np.array([[rho_a], [rho_b]], dtype=np.float64)
        except OverflowError as exc:
            raise ValueError("rho inputs exceed FP64") from exc
        return self.evaluate(x)[:, 0]
