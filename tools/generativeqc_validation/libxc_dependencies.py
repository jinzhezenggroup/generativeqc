"""Source dependencies for the retained direct/all-electron Libxc E/F domain.

Whole-build hashes authenticate the measured binary; they are not qualification
invalidation keys. Unknown native inputs deliberately affect both backends and
products. Only explicitly disjoint consumers or backend/product-owned inputs
receive narrower scope. Additions and deletions participate in comparisons.
"""

from __future__ import annotations

import hashlib
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from generativeqc_compiler.common.provenance import canonical_hash, file_hash

from tools.generate_build_identity import _inventory

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

SCHEMA = "generativeqc.libxc-native-dependencies.v1"
NATIVE_GENERATORS = (
    "tools/generate_native_stationary_cuda.py",
    "tools/generate_libxc_semilocal_cpu_registry.py",
    "tools/generate_libxc_semilocal_cuda_registry.py",
)
DISJOINT_ROOTS = (
    "src/cc/",
    "src/posthf/mp2_",
    "src/posthf/ump2_",
    "src/posthf/ri_mp2_",
    "src/xtb/",
    "src/cli/",
    "python/generativeqc_compiler/cc/",
    "python/generativeqc_compiler/mp2/",
    "python/generativeqc_compiler/geometry/",
    "tools/generativeqc_cc/",
    "tools/generativeqc_mp2/",
)
POINT_CONTRACT_INPUTS = frozenset(
    "python/generativeqc_compiler/xc/" + name
    for name in (
        "libxc_bulk_catalog.json",
        "b3lyp_production_policy.py",
    )
)
CUDA_OWNERS = (
    "src/dft/",
    "src/scf/",
    "src/integrals/",
    "src/runtime/",
    "src/tensor/",
    "src/solver/",
    "python/generativeqc_compiler/xc/",
    "python/generativeqc_compiler/dft/",
    "python/generativeqc_compiler/method/",
    "python/generativeqc_compiler/integral/",
    "tools/generate_",
)


def native_inventory(root: Path) -> dict[str, str]:
    """Include native generators and assets, not just C++/CUDA source files."""
    return {
        path.as_posix(): file_hash(root / path)
        for path in _inventory(root, root / "cmake/GenerativeQCSourceIdentity.json")
    }


def authenticated_inputs(bundle: Mapping, inputs: Mapping) -> dict[str, str]:
    """Authenticate a measured census against both immutable receipt digests.

    Legacy campaigns may supply this census through a hash-pinned manifest
    sidecar recovered from their measured Git tree. Neither the campaign nor its
    ELF/Graph identities are rewritten during that recovery.
    Generator hashes omitted by the old build manifest are supplemental inputs
    from the declared measured Git revision, protected by the sidecar's manifest
    digest. Future embedded native identities bind those generators directly.
    """
    if inputs.get("schema") != SCHEMA:
        raise ValueError("native dependency inventory schema mismatch")
    sources, native = inputs["sources"], inputs["native_sources"]
    supplemental = inputs.get("generator_sources", {})
    if not supplemental.keys() <= set(NATIVE_GENERATORS):
        raise ValueError("unknown supplemental native generator input")
    if supplemental and inputs.get("source_commit") != bundle["source_commit"]:
        raise ValueError("native generator inventory/source revision mismatch")
    if (
        len(sources) != bundle["source_input_count"]
        or canonical_hash(sources) != bundle["source_identity"]
    ):
        raise ValueError("native dependency inventory/source mismatch")
    for paths in (sources, native, supplemental):
        for name, digest in paths.items():
            path = PurePosixPath(name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or path.as_posix() != name
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError("invalid native dependency input")
    identity = hashlib.sha256(
        "".join(
            f"{name}:{digest}\n" for name, digest in sorted(native.items())
        ).encode()
    ).hexdigest()
    if identity != bundle["native_build_verification"]["measured_input_identity"]:
        raise ValueError("native dependency inventory/embedded identity mismatch")
    if any(sources[name] != native[name] for name in sources.keys() & native.keys()):
        raise ValueError("conflicting native dependency inputs")
    measured = {**sources, **native}
    if any(
        name in measured and measured[name] != digest
        for name, digest in supplemental.items()
    ):
        raise ValueError("conflicting native generator inputs")
    measured.update(supplemental)
    if not set(NATIVE_GENERATORS) <= measured.keys():
        raise ValueError("missing native generator dependencies")
    return measured


def affects(path: str, *, backend: str, product: str) -> bool:
    """Select dependencies of the measured FP64 direct semilocal context.

    XC Graph/point contracts are checked per registration by the receipt
    validators, rather than by hashing unrelated scalar compiler modules. Build
    configuration and ABI inputs remain conservative shared dependencies.
    Spatial XC E/V contractions are energy dependencies, not force-only owners.
    """
    if path.startswith(DISJOINT_ROOTS) or PurePosixPath(path).suffix in (".md", ".rst"):
        return False
    if path in POINT_CONTRACT_INPUTS:
        return False
    if path == "cmake/GenerativeQCSourceIdentity.json":
        return False
    if path == "CMakeLists.txt" or path.startswith(
        ("include/", "cmake/", "python/generativeqc_compiler/common/")
    ):
        return True
    if product == "compiled" and path.startswith(
        ("src/scf/", "src/integrals/", "src/molecule/", "src/hf/", "src/solver/")
    ):
        return False
    name = PurePosixPath(path)
    stem = name.stem.lower()
    parts = name.parts
    cuda = path.startswith(CUDA_OWNERS) and (
        name.suffix in (".cu", ".cuh")
        or "cuda" in parts
        or stem.startswith("cuda_")
        or stem == "cuda"
        or "_cuda" in stem
    )
    if cuda and backend == "cpu":
        return False
    force_owner = path.startswith(
        (
            "src/dft/",
            "src/scf/gradient/",
            "src/integrals/",
            "python/generativeqc_compiler/method/",
            "python/generativeqc_compiler/dft/",
            "tools/generate_",
        )
    ) or path in (
        "src/scf/stationary_integral_gradient.cpp",
        "src/scf/stationary_integral_gradient.hpp",
    )
    force = force_owner and (
        any(word in stem for word in ("gradient", "derivative", "stationary", "force"))
        or "gradient" in parts
    )
    return product == "forces" or not force


def changed_inputs(
    measured: Mapping[str, str], live: Mapping[str, str], *, backend: str, product: str
) -> list[str]:
    """Return an actionable, deterministic delta, including missing/new inputs."""
    return sorted(
        path
        for path in measured.keys() | live.keys()
        if measured.get(path) != live.get(path)
        and affects(path, backend=backend, product=product)
    )
