"""Scalar emission and artifact compilation; execution has a separate owner."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

from generativeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from generativeqc_compiler.common.native_runtime import compile_runtime
from generativeqc_compiler.common.paths import asset_path
from generativeqc_compiler.common.provenance import canonical_hash, file_hash
from generativeqc_compiler.common.source_cache import cache_source
from generativeqc_compiler.integral.scalar_c import ScalarCEmitter

from .user_expression import FEATURES
from .user_expression_program import MAX_POINTS, UserExpressionProgram

if TYPE_CHECKING:
    from generativeqc_compiler.common.cuda_runtime import CudaArtifact

MAX_SOURCE_BYTES = 262144
RUNTIME_HEADER = "src/dft/xc_user_expression_point.hpp"


def emit_native(program: UserExpressionProgram) -> tuple[str, dict]:
    """Emit canonical scalar roots into the independently owned point runtime."""
    if type(program) is not UserExpressionProgram:
        raise TypeError("native expression requires UserExpressionProgram")
    graph, roots = program.scalar_graph()
    emitter = ScalarCEmitter(
        graph, {key: f"rho[{i}]" for i, key in enumerate(FEATURES)}
    )
    emitter.emit(roots)
    lines = [
        "// Generated bounded user LDA polynomial; no Libxc or SCF admission.",
        '#include "xc_user_expression_point.hpp"',
        'extern "C" int xc_user_expression(const double* input, size_t input_count, double* output, size_t output_count, size_t npoint) noexcept {',
        f"  return generativeqc::dft::user_expression::points<{len(roots)}, {MAX_POINTS}>(",
        "      input, input_count, output, output_count, npoint,",
        "      [](const double* rho, double* values) noexcept {",
        *emitter.lines,
        *[
            f"    values[{row}] = {emitter.reference(root)};"
            for row, root in enumerate(roots)
        ],
        "  });",
        "}",
        "",
    ]
    source = "\n".join(lines)
    if len(source.encode()) > MAX_SOURCE_BYTES:
        raise ValueError("native expression exceeds source budget")
    header = asset_path(RUNTIME_HEADER).read_text(encoding="utf-8")
    return source, {
        "schema": "generativeqc.xc-expression-native.v2",
        "program": program.to_payload(),
        "source_sha256": sha256(source.encode()).hexdigest(),
        "runtime_headers": {RUNTIME_HEADER: sha256(header.encode()).hexdigest()},
        "max_source_bytes": MAX_SOURCE_BYTES,
    }


def compile_native(
    program: UserExpressionProgram, *, compiler: CppCompilerAdapter, cache: str | Path
) -> CudaArtifact:
    """Compile and verify an artifact without loading a native library.

    Stage the checked runtime header beside generated source so compiler artifact
    identity uses the same relative header name for checkout and installed use.
    A stale or altered staged source/header is rejected, never silently repaired.
    """
    if not isinstance(compiler, CppCompilerAdapter):
        raise TypeError("native expression requires CppCompilerAdapter")
    source, metadata = emit_native(program)
    header = asset_path(RUNTIME_HEADER).read_text(encoding="utf-8")
    if (
        sha256(header.encode()).hexdigest()
        != metadata["runtime_headers"][RUNTIME_HEADER]
    ):
        raise ValueError("native expression runtime header changed during preparation")
    cache = Path(cache).resolve()
    directory = cache / "source" / canonical_hash(metadata)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "xc_user_expression.cpp"
    staged_header = directory / "xc_user_expression_point.hpp"
    cache_source(path, source)
    cache_source(staged_header, header)
    artifact = compile_runtime(
        compiler, cache, path, headers=(staged_header,), options=("-ffp-contract=off",)
    )
    identity = artifact.metadata["identity"]
    if identity["source"] != metadata["source_sha256"] or identity["headers"] != {
        staged_header.name: metadata["runtime_headers"][RUNTIME_HEADER]
    }:
        raise ValueError("native expression source/header identity mismatch")
    if file_hash(artifact.library) != artifact.metadata["binary_sha256"]:
        raise ValueError("native expression binary identity mismatch")
    return artifact
