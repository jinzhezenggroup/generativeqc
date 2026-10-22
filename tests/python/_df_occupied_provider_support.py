"""Real borrowed column adapters for vendor-injected host translation units."""

from pathlib import Path

from _cpp_source_support import cpp_function_definition, cpp_record_definition


def column_adapters(*, witness: bool = False) -> str:
    """Keep the production records/forwarders intact except an optional witness."""
    header = (
        Path(__file__).resolve().parents[2] / "src/tensor/cuda_runtime.cuh"
    ).read_text()
    result = "namespace generativeqc_tensor {\n"
    for record, function in (
        ("ColumnGemmView", "submit_column_gemm"),
        ("ColumnBatchedGemmView", "submit_column_batched_gemm"),
    ):
        body = cpp_function_definition(header, function)
        if witness:
            body = body.replace("{", "{ provider_entry = true;", 1)
        result += cpp_record_definition(header, record) + ";\n" + body + "\n"
    return result + "}\n"
