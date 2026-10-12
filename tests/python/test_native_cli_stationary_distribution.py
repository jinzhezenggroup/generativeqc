"""Host-only regression checks for the standalone CUDA force link contract."""

from __future__ import annotations

import unittest

from tools.check_native_cli_stationary import compiler_native_profiles, verify_contract


def fixtures() -> tuple[str, str, str]:
    cache = (
        "GENERATIVEQC_MONOLITHIC_CLI:BOOL=ON\n"
        "GENERATIVEQC_ENABLE_NATIVE_STATIONARY_CUDA:BOOL=ON\n"
        "GENERATIVEQC_ENABLE_STATIONARY_FORCE_AOT:BOOL=OFF"
    )
    registry = "\n".join(
        f"    &native_stationary_profile_{profile}(),"
        for profile in compiler_native_profiles()
    )
    symbols = (
        "\n".join(
            f"00000000 T generativeqc::dft::native_stationary_profile_{profile}()"
            for profile in compiler_native_profiles()
        )
        + "\n00000000 T generativeqc::dft::cuda_stationary_bindings()"\n        + "\n00000000 T generativeqc::dft::launch_native_stationary_source_transport(...)"\n        + "\n00000000 T generativeqc::dft::launch_native_stationary_source_reduction(...)"
    )
    return cache, registry, symbols


class StandaloneCudaForceTests(unittest.TestCase):
    def test_compiler_graph_inventory_static_contract(self) -> None:
        verify_contract(*fixtures())

    def test_disabled_native_profiles_fail(self) -> None:
        cache, registry, symbols = fixtures()
        with self.assertRaisesRegex(ValueError, "NATIVE_STATIONARY_CUDA"):
            verify_contract(
                cache.replace(
                    "GENERATIVEQC_ENABLE_NATIVE_STATIONARY_CUDA:BOOL=ON",
                    "GENERATIVEQC_ENABLE_NATIVE_STATIONARY_CUDA:BOOL=OFF",
                ),
                registry,
                symbols,
            )

    def test_python_shared_artifacts_are_not_distribution_dependencies(self) -> None:
        cache, registry, symbols = fixtures()
        with self.assertRaisesRegex(ValueError, "STATIONARY_FORCE_AOT"):
            verify_contract(
                cache.replace(
                    "GENERATIVEQC_ENABLE_STATIONARY_FORCE_AOT:BOOL=OFF",
                    "GENERATIVEQC_ENABLE_STATIONARY_FORCE_AOT:BOOL=ON",
                ),
                registry,
                symbols,
            )

    def test_missing_registered_profile_fails(self) -> None:
        cache, registry, symbols = fixtures()
        with self.assertRaisesRegex(ValueError, "registry mismatch"):
            verify_contract(
                cache,
                registry.replace(
                    f"&native_stationary_profile_{compiler_native_profiles()[-1]}(),",
                    "",
                ),
                symbols,
            )

    def test_linker_discarded_profile_fails(self) -> None:
        cache, registry, symbols = fixtures()
        with self.assertRaisesRegex(ValueError, "missing native force symbol"):
            verify_contract(
                cache,
                registry,
                symbols.replace(
                    f"generativeqc::dft::native_stationary_profile_{compiler_native_profiles()[0]}()",
                    "discarded()",
                ),
            )

    def test_missing_shared_cuda_kernel_fails(self) -> None:
        cache, registry, symbols = fixtures()
        with self.assertRaisesRegex(ValueError, "missing shared CUDA kernel launcher"):
            verify_contract(
                cache,
                registry,
                symbols.replace("launch_native_stationary_source_reduction(", "discarded("),
            )

    def test_missing_registry_entry_point_fails(self) -> None:
        cache, registry, symbols = fixtures()
        with self.assertRaisesRegex(ValueError, "registry entry point"):
            verify_contract(
                cache,
                registry,
                symbols.replace(
                    "generativeqc::dft::cuda_stationary_bindings()",
                    "discarded()",
                ),
            )


if __name__ == "__main__":
    unittest.main()
