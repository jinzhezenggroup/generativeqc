"""Protect common automatic point semantics and complete CPU source parity."""

import hashlib
import os
import re
from dataclasses import replace

import pytest
from generativeqc_compiler.xc.automatic_point import emit_automatic_point
from generativeqc_compiler.xc.libxc_maple import MapleImportError
from generativeqc_compiler.xc.spec import UnsupportedXC

from tools import generate_libxc_semilocal_cpu_registry as registry


@pytest.mark.parametrize("name", ["LDA_C_VWN_4", "GGA_X_APBE", "MGGA_X_LTA"])
def test_backends_share_semantics_but_not_emitted_source_identity(name: str) -> None:
    host = emit_automatic_point(name)
    device = emit_automatic_point(name, backend="cuda")

    assert host.expression_identity == device.expression_identity
    assert host.features == device.features
    assert (
        host.outputs
        == device.outputs
        == ((), *((axis,) for axis in range(len(host.features))))
    )
    assert host.ingredient_mask == device.ingredient_mask
    assert host.source_identity != device.source_identity
    assert emit_automatic_point(name) == host
    assert emit_automatic_point(name, backend="cuda") == device
    assert "throw std::domain_error" in host.source
    assert "throw" not in device.source
    assert "invalid.valid = false" in device.source
    assert "__device__ inline DevicePointValue" in device.source
    assert "total_density <" in device.source


def test_exact_emitted_contract_changes_invalidate_source_identity() -> None:
    source = emit_automatic_point("GGA_X_APBE")
    for changed in (
        replace(source, source=source.source + "\n"),
        replace(source, expression_identity="0" * 64),
        replace(source, outputs=source.outputs[:-1]),
        replace(source, features=source.features[::-1]),
        replace(source, ingredient_mask=1),
        replace(source, backend="cuda"),
    ):
        assert source.source_identity != changed.source_identity


def test_unsupported_backend_or_registration_never_emits_a_point() -> None:
    with pytest.raises(ValueError, match="backend must be cpu or cuda"):
        emit_automatic_point("GGA_X_APBE", backend="metal")  # type: ignore[arg-type]
    with pytest.raises((UnsupportedXC, MapleImportError, ValueError)):
        emit_automatic_point("GGA_X_NOT_REGISTERED", backend="cuda")
    with pytest.raises(MapleImportError, match="compile-time scalars"):
        emit_automatic_point("MGGA_X_TPSS", backend="cuda")


@pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_POINT_MIGRATION_BASELINE") != "1",
    reason="historical byte-migration audit; executable numerical gates own correctness",
)
def test_all_cpu_shards_match_explicit_spin_screening_baseline() -> None:
    # The common-owner refactor preserves mathematical bytes; the subsequent
    # physical spin-coordinate fix deliberately updates this complete baseline.
    # Ignore only source-closure descriptor hashes, never executable statements.
    expected = {
        "generated_libxc_semilocal_0.cpp": "492256f2fd3e59fb5ebcc7e3078d374eda6f89078fd7179d7a712a70f5c4ce3f",
        "generated_libxc_semilocal_1.cpp": "5fad24955de2655f3cdedfbd915ed63efaf8bc12f3e1bc3e173f76f2c050d479",
        "generated_libxc_semilocal_2.cpp": "05f39614bd76a35cdc187f09cbacdf3e1f3e8d1ec1db4f58c5f3083d0c36571e",
        "generated_libxc_semilocal_3.cpp": "4ee464923310e1844e134bd85c0b1b73653f2dbf6c528b05d387484234bebc28",
        "generated_libxc_semilocal_4.cpp": "9b97be92956af089ff21e71eae737638bc17b0c65fcbc080289dabf3c69d3cbd",
        "generated_libxc_semilocal_5.cpp": "a4ec5822d97d4bdac3a1fe34a87ad1a8b5ff1bf580f66de9f834b225ba19202d",
        "generated_libxc_semilocal_6.cpp": "f04e1ad92dc8368dc48d3445aef5d03c3d39fd2703097f2fae29d9a1dff5b06f",
        "generated_libxc_semilocal_7.cpp": "62f57528ed6f1a2de0939e9dc9590cfe466194a670ea203af90a83b55dd8cffc",
        "generated_libxc_semilocal_registry.cpp": "c83c06f9a5cc4c197f349b9b234fba9483c231734eb015f53cc7d40deec8f2bf",
        "generated_libxc_semilocal_registry.hpp": "28f5cedefc994cc62faecaf9990c56cf5a035b16a2db6092893e50e8f6f4fa63",
    }
    entries = registry.registry_entries()
    sources = {
        "generated_libxc_semilocal_registry.cpp": registry.emit_registry(),
        "generated_libxc_semilocal_registry.hpp": registry.emit_header(),
        **{
            f"generated_libxc_semilocal_{index}.cpp": registry.emit_shard(
                index, entries
            )
            for index in range(registry.SHARD_COUNT)
        },
    }
    for name, source in sources.items():
        normalized = re.sub(
            r'(kAutomatic_\w+_ExpressionIdentity = ")[a-f0-9]{64}(";)',
            r"\1<source-bound-identity>\2",
            source,
        )
        assert hashlib.sha256(normalized.encode()).hexdigest() == expected[name], name
