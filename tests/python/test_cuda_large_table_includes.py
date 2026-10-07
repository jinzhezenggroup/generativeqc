from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HEADERS = {
    "dft/dispersion/d4_data.hpp",
    "dft/dispersion/d4_eeq_data.hpp",
    "dft/dispersion/d4_eeq_r2scan3c_c6.hpp",
}

def test_production_cuda_translation_units_do_not_include_large_generated_tables() -> None:
    offenders = []
    for path in (ROOT / "src").rglob("*.cu"):
        source = path.read_text(encoding="utf-8")
        for header in HEADERS:
            if f'#include "{header}"' in source:
                offenders.append(f"{path.relative_to(ROOT)} -> {header}")
    assert offenders == []

def test_cuda_facing_d4_headers_are_table_data_free() -> None:
    for relative in (
        "src/dft/dispersion/d4_reference.hpp",
        "src/dft/dispersion/d4_eeq.hpp",
        "src/dft/dispersion/d4_cuda.hpp",
        "src/dft/dispersion/d4_runtime.hpp",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        for header in HEADERS:
            assert f'#include "{header}"' not in source, f"{relative} -> {header}"

def test_large_d4_tables_have_one_production_cpp_owner() -> None:
    source = (ROOT / "src/dft/dispersion/d4_host_tables.cpp").read_text(encoding="utf-8")
    for header in HEADERS:
        assert f'#include "{header}"' in source
