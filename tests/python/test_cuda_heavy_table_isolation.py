import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
HEAVY_DATA_HEADERS = {
    (SRC / "dft/dispersion/d4_data.hpp").resolve(),
    (SRC / "dft/dispersion/d4_eeq_data.hpp").resolve(),
    (SRC / "dft/dispersion/d4_eeq_r2scan3c_c6.hpp").resolve(),
}
INCLUDE_RE = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.MULTILINE)
INCLUDE_ROOTS = (
    SRC,
    SRC / "xtb/native",
    SRC / "xtb/native/src",
)
CUDA_SOURCE_ROOTS = (
    SRC,
    ROOT / "tests",
    ROOT / "benchmarks",
)


def _resolve_include(owner: Path, include: str) -> Path | None:
    for candidate in (owner.parent / include, *(root / include for root in INCLUDE_ROOTS)):
        if candidate.is_file():
            return candidate.resolve()
    return None


def _heavy_chain(root: Path) -> list[Path] | None:
    stack = [(root.resolve(), [root.resolve()])]
    visited: set[Path] = set()
    while stack:
        path, chain = stack.pop()
        if path in HEAVY_DATA_HEADERS:
            return chain
        if path in visited:
            continue
        visited.add(path)
        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for include in INCLUDE_RE.findall(source):
            child = _resolve_include(path, include)
            if child is not None:
                stack.append((child, [*chain, child]))
    return None


def test_cuda_translation_units_do_not_parse_large_immutable_d4_tables() -> None:
    offenders = []
    for root in CUDA_SOURCE_ROOTS:
        for source in root.rglob("*.cu"):
            chain = _heavy_chain(source)
            if chain is not None:
                offenders.append(" -> ".join(path.relative_to(ROOT).as_posix() for path in chain))
    assert offenders == []
