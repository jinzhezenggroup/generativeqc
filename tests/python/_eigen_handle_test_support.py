"""Link real empty solver ownership into non-numerical host probes."""

from pathlib import Path
from shutil import copyfile

ROOT = Path(__file__).resolve().parents[2]


def empty_eigen_owner_units(directory: Path) -> tuple[Path, Path]:
    """Stub only vendor calls; any call from an empty owner fails the probe."""
    fixture = ROOT / "tests/native/fixtures/shared_eigen_provider"
    copyfile(fixture / "cusolverDn.h", directory / "cusolverDn.h")
    return (
        ROOT / "src/solver/cuda/symmetric_eigen_handles.cpp",
        fixture / "empty_handles.cpp",
    )
