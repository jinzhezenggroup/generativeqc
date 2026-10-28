"""One physical CUDA tile contract, independently of point mathematical policy."""

import sys
from dataclasses import replace
from pathlib import Path

import pytest
from generativeqc_compiler.xc.automatic_cuda import (
    emit_automatic_cuda_descriptor,
    emit_automatic_cuda_provider,
)
from generativeqc_compiler.xc.automatic_cuda_package import automatic_cuda_package
from generativeqc_compiler.xc.automatic_point import emit_automatic_point
from generativeqc_compiler.xc.automatic_semilocal import (
    AutomaticRegistryEntry,
    automatic_registry_entries,
)

from tools import generate_libxc_semilocal_cpu_registry as cpu_registry
from tools import generate_libxc_semilocal_cuda_registry as cuda_registry


@pytest.mark.parametrize(
    ("name", "feature_terms"),
    (("LDA_C_VWN_4", 1), ("GGA_X_APBE", 4), ("MGGA_X_LTA", 5)),
)
def test_automatic_provider_preserves_mathematics_and_physical_feature_contract(
    name: str, feature_terms: int
) -> None:
    provider = emit_automatic_cuda_provider(name)
    assert provider.point == emit_automatic_point(name, backend="cuda")
    assert (
        provider.point.expression_identity
        == emit_automatic_point(name).expression_identity
    )
    assert provider.feature_terms == feature_terms
    assert emit_automatic_cuda_provider(name) == provider


def test_provider_source_identity_covers_layout_and_emitted_contract() -> None:
    provider = emit_automatic_cuda_provider("GGA_X_APBE")
    for altered in (
        replace(provider, source=provider.source + "\n"),
        replace(provider, provider_name="another_provider"),
        replace(provider, point=replace(provider.point, expression_identity="0" * 64)),
    ):
        assert altered.source_identity != provider.source_identity
    with pytest.raises(ValueError, match="restricted layout"):
        replace(provider, feature_terms=1)


def test_semantic_cuda_resolution_does_not_grant_optional_mechanisms() -> None:
    from generativeqc_compiler.method.bulk_ks import resolve_bulk_ks

    provider = emit_automatic_cuda_provider("GGA_X_APBE")
    resolved = resolve_bulk_ks("GGA_X_APBE", backend="cuda")
    assert resolved.backend == "cuda"
    assert "&launch, &launch_batch, {}" in emit_automatic_cuda_descriptor(provider)


def test_cpu_and_cuda_share_structural_registry_without_a_positive_whitelist() -> None:
    entries = automatic_registry_entries()
    assert entries == cpu_registry.registry_entries()
    assert len(entries) == 197
    assert len({entry.code for entry in entries}) == len(entries)
    assert cuda_registry.automatic_registry_entries is automatic_registry_entries
    assert cuda_registry.SHARD_COUNT == 32


def _assert_shared_package_closure(
    entries: tuple[AutomaticRegistryEntry, ...],
) -> tuple[str, ...]:
    """Every complete worker and scientific registration has one shard owner."""
    workers, registrations = automatic_cuda_package(
        tuple(entry.name for entry in entries)
    )
    shards = tuple(
        cuda_registry.emit_shard(index, entries)
        for index in range(cuda_registry.SHARD_COUNT)
    )
    assert set(registrations) == {entry.name for entry in entries}
    assert tuple(workers) == tuple(sorted(workers))
    for ordinal, worker in enumerate(workers.values()):
        assert tuple(source.count(worker.source) for source in shards) == tuple(
            int(index == ordinal % cuda_registry.SHARD_COUNT)
            for index in range(cuda_registry.SHARD_COUNT)
        )
    for ordinal, entry in enumerate(entries):
        owner = ordinal % cuda_registry.SHARD_COUNT
        registration = registrations[entry.name]
        provider = emit_automatic_cuda_provider(entry.name)
        assert provider.expression_identity in registration
        assert provider.point.expression_identity in registration
        if provider.feature_terms == 1:
            assert provider.restricted.expression_identity in registration
        assert tuple(source.count(registration) for source in shards) == tuple(
            int(index == owner) for index in range(cuda_registry.SHARD_COUNT)
        )
        dispatch = f"case 0x{entry.code:x}U:"
        assert tuple(source.count(dispatch) for source in shards) == tuple(
            int(index == owner) for index in range(cuda_registry.SHARD_COUNT)
        )
    assert sum(source.count("    case ") for source in shards) == len(entries)
    for source in shards:
        unowned = source
        for worker in workers.values():
            unowned = unowned.replace(worker.source, "")
        for registration in registrations.values():
            unowned = unowned.replace(registration, "")
        assert "automatic_evaluate_points" not in unowned
    return shards


def test_aot_descriptor_binds_shared_science_and_exact_launcher_source() -> None:
    entry = next(
        item for item in automatic_registry_entries() if item.name == "GGA_X_APBE"
    )
    provider = emit_automatic_cuda_provider(entry.name)
    descriptor = emit_automatic_cuda_descriptor(provider)
    shard = _assert_shared_package_closure((entry,))[0]
    assert provider.source_identity in descriptor
    assert provider.expression_identity in descriptor
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


def test_explicit_package_preserves_registration_order_and_identity() -> None:
    """Packaging is not a named science branch or a provenance alias merge."""
    names = ["MGGA_X_LTA", "GGA_X_APBE", "LDA_C_VWN_4", "GGA_X_APBE"]
    expected = tuple(
        entry for entry in automatic_registry_entries() if entry.name in names
    )
    assert cuda_registry.select_entries() == automatic_registry_entries()
    assert cuda_registry.select_entries(names) == expected
    assert cuda_registry.select_entries(list(reversed(names))) == expected
    assert len(expected) == 3
    shards = _assert_shared_package_closure(expected)
    for index, entry in enumerate(expected):
        shard = shards[index]
        for absent in expected:
            if absent != entry:
                assert f"case 0x{absent.code:x}U:" not in shard


@pytest.mark.parametrize("names", ([], ["UNKNOWN"], ["GGA_X_APBE", "MGGA_X_BR89"]))
def test_explicit_package_rejects_unknown_and_unrepresentable_names(
    names: list[str],
) -> None:
    with pytest.raises(ValueError, match="unregistered automatic CUDA AOT"):
        cuda_registry.select_entries(names)


@pytest.mark.parametrize(
    "options",
    (("--functional", "UNKNOWN"), ("--disabled", "--functional", "GGA_X_APBE")),
)
def test_invalid_package_cli_writes_no_partial_outputs(
    options: tuple[str, ...], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["generate", "--output-directory", str(tmp_path / "output"), *options],
    )
    with pytest.raises(SystemExit) as failure:
        cuda_registry.main()
    assert failure.value.code == 2
    assert not (tmp_path / "output").exists()


def test_selected_cli_replaces_stale_shards_without_extra_point_programs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Shrinking a package must remove omitted executable dispatch cases."""
    output = tmp_path / "output"
    output.mkdir()
    for index in range(cuda_registry.SHARD_COUNT):
        (output / f"generated_libxc_semilocal_cuda_{index}.cu").write_text("stale")
    monkeypatch.setattr(
        sys,
        "argv",
        ["generate", "--output-directory", str(output), "--functional", "GGA_X_APBE"],
    )
    cuda_registry.main()
    selected = cuda_registry.select_entries(["GGA_X_APBE"])[0]
    sources = [path.read_text() for path in sorted(output.glob("*.cu"))]
    assert len(sources) == cuda_registry.SHARD_COUNT
    assert sum(source.count("    case ") for source in sources) == 1
    assert sum(f"case 0x{selected.code:x}U:" in source for source in sources) == 1
    assert "stale" not in "".join(sources)
    mtimes = {path: path.stat().st_mtime_ns for path in output.iterdir()}
    cuda_registry.main()
    assert {path: path.stat().st_mtime_ns for path in output.iterdir()} == mtimes
