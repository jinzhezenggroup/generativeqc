"""One physical CUDA tile contract, independently of point mathematical policy."""

from dataclasses import replace

import pytest
from generativeqc_compiler.dft.ao_cuda import emit_native_xc_contraction_kernels
from generativeqc_compiler.dft.xc_point_cuda import (
    emit_device_point_value,
    emit_point_traversal,
)
from generativeqc_compiler.xc.automatic_cuda import (
    emit_automatic_cuda_descriptor,
    emit_automatic_cuda_provider,
)
from generativeqc_compiler.xc.automatic_point import emit_automatic_point
from generativeqc_compiler.xc.automatic_semilocal import automatic_registry_entries

from tools import generate_libxc_semilocal_cpu_registry as cpu_registry
from tools import generate_libxc_semilocal_cuda_registry as cuda_registry


@pytest.mark.parametrize(
    ("name", "feature_terms"),
    (("LDA_C_VWN_4", 1), ("GGA_X_APBE", 4), ("MGGA_X_LTA", 5)),
)
def test_automatic_and_curated_consumers_share_one_physical_traversal(
    name: str, feature_terms: int
) -> None:
    provider = emit_automatic_cuda_provider(name)
    curated = emit_native_xc_contraction_kernels()
    assert provider.point == emit_automatic_point(name, backend="cuda")
    assert (
        provider.point.expression_identity
        == emit_automatic_point(name).expression_identity
    )
    assert provider.feature_terms == feature_terms
    assert provider.source.count(emit_device_point_value()) == 1
    assert curated.count(emit_device_point_value()) == 1
    assert provider.source.count(emit_point_traversal()) == 1
    assert curated.count(emit_point_traversal()) == 1
    assert (
        "evaluate_point_tiles<feature_terms, response, batched, CuratedPointEvaluator<static_family>>"
        in curated
    )
    assert (
        "evaluate_point_tiles<feature_terms, false, batched, PointEvaluator>"
        in provider.source
    )
    assert "evaluate_semilocal_point(" not in provider.source
    assert (
        "response_point("
        not in provider.source.split("struct " + provider.provider_name, 1)[1].split(
            "};", 1
        )[0]
    )
    assert emit_automatic_cuda_provider(name) == provider


def test_provider_source_identity_covers_layout_and_emitted_contract() -> None:
    provider = emit_automatic_cuda_provider("GGA_X_APBE")
    for altered in (
        replace(provider, source=provider.source + "\n"),
        replace(provider, provider_name="another_provider"),
        replace(provider, feature_terms=1),
        replace(provider, point=replace(provider.point, expression_identity="0" * 64)),
    ):
        assert altered.source_identity != provider.source_identity


def test_emission_does_not_grant_public_cuda_admission() -> None:
    from generativeqc_compiler.method.bulk_ks import resolve_bulk_ks
    from generativeqc_compiler.method.spec import UnsupportedMethod

    emit_automatic_cuda_provider("GGA_X_APBE")
    with pytest.raises(UnsupportedMethod, match="supports CPU only"):
        resolve_bulk_ks("GGA_X_APBE", backend="cuda")


def test_cpu_and_cuda_share_structural_registry_without_a_positive_whitelist() -> None:
    entries = automatic_registry_entries()
    assert entries == cpu_registry.registry_entries()
    assert len(entries) == 197
    assert len({entry.code for entry in entries}) == len(entries)
    assert cuda_registry.automatic_registry_entries is automatic_registry_entries
    assert cuda_registry.SHARD_COUNT == 32


def test_aot_descriptor_binds_shared_science_and_exact_launcher_source() -> None:
    entry = next(
        item for item in automatic_registry_entries() if item.name == "GGA_X_APBE"
    )
    provider = emit_automatic_cuda_provider(entry.name)
    descriptor = emit_automatic_cuda_descriptor(provider)
    shard = cuda_registry.emit_shard(0, (entry,))
    assert provider.source in shard
    assert descriptor in shard
    assert provider.source_identity in descriptor
    assert provider.point.expression_identity in descriptor
    assert f"0x{entry.code:x}U" in descriptor
    assert "7U," in descriptor
    assert "&launch, &launch_batch, {}" in descriptor
    assert "delta) throw" in provider.source
    assert "exchange_scale != 1.0" in provider.source
    assert "count > std::size_t{std::numeric_limits<I>::max()}" in provider.source
    assert "default: return nullptr;" in shard
    assert (
        "automatic_cuda_point_entry(std::uint32_t code) noexcept"
        in cuda_registry.emit_header()
    )


def test_disabled_bundle_has_no_scientific_entries_or_shard_dependencies() -> None:
    disabled = cuda_registry.emit_registry(enabled=False)
    assert "return nullptr;" in disabled
    assert "automatic_cuda_point_shard" not in disabled
    assert "automatic_evaluate_points" not in disabled
