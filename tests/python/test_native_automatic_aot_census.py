"""Artifact economics retain spill evidence and cannot promote molecular science."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import pytest
from generativeqc_compiler.xc.libxc_bulk_capabilities import functional_capability

from tools.census_native_automatic_aot import (
    compiled_capability_evidence,
    device_image,
    spill_records,
)

if TYPE_CHECKING:
    from pathlib import Path

    from conftest import NativeCxx


def test_installed_device_image_checks_bytes_independently_of_host_metadata(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    """An unchanged resource table cannot hide a changed executable image."""
    payload = b"test-CUDA-image-0123456789"
    source = tmp_path / "image.cpp"
    source.write_text(
        'extern "C" __attribute__((used, section(".nv_fatbin"))) '
        "const unsigned char image[] = {" + ",".join(map(str, payload)) + "};\n"
    )
    binary = native_cxx.build_shared([source], tmp_path / "image.so")
    expected = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    assert device_image(binary) == expected
    original = binary.read_bytes()
    binary.write_bytes(original + b"unrelated host metadata")
    assert device_image(binary) == expected
    assert original.count(payload) == 1
    binary.write_bytes(original.replace(payload, b"X" + payload[1:]))
    assert device_image(binary) != expected


def test_compiled_evidence_reaches_existing_promotion_owner_without_force_claim() -> (
    None
):
    library = {"sha256": "a" * 64, "bytes": 4096, "path": "core.so"}
    evidence = compiled_capability_evidence(
        "LDA_C_VWN_4",
        census={"library": library, "installed_library": library},
        reference="qualified-census.json",
    )
    capability = functional_capability("LDA_C_VWN_4", evidence=evidence)
    assert "compiled-cuda" in capability.qualified_stages
    assert not {"molecular-scf", "forces", "public-method"} & set(
        capability.qualified_stages
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        functional_capability("LDA_C_VWN_3", evidence=evidence)


def test_ptxas_spills_are_measured_separately_from_stack_and_local_storage() -> None:
    name = "_ZN_generativeqc_native_shared_0123_geometry"
    log = f"""ptxas info    : Function properties for {name}
    128 bytes stack frame, 16 bytes spill stores, 24 bytes spill loads
ptxas info    : Used 64 registers, 0 bytes smem
"""
    assert spill_records(log)[name] == {
        "stack_bytes": 128,
        "spill_store_bytes": 16,
        "spill_load_bytes": 24,
    }
    with pytest.raises(ValueError, match="ambiguous"):
        spill_records(log + log.replace("16 bytes spill", "32 bytes spill"))
    assert spill_records("LOCAL:0 STACK:128") == {}
