"""Host-only regression checks for the standalone CUDA force link contract."""

from __future__ import annotations

import unittest

from tools.check_native_cli_stationary import EXPECTED_PROFILES, verify_contract


def fixtures() -> tuple[str, str, str]:
    cache = "\n".join(
        (
            "GENERATIVEQC_MONOLITHIC_CLI:BOOL=ON",
            "GENERATIVEQC_ENABLE_NATIVE_STATIONARY_CUDA:BOOL=ON",
            "GENERATIVEQC_ENABLE_STATIONARY_FORCE_AOT:BOOL=OFF",
        )
    )
    registry = "\n".join(
        f"    &native_stationary_profile_{profile}(),"
        for profile in EXPECTED_PROFILES
    )
    symbols = "\n".join(
        f"00000000 T generativeqc::dft::native_stationary_profile_{profile}()"
        for profile in EXPECTED_PROFILES
    ) + "\n00000000 T generativeqc::dft::cuda_stationary_bindings()"
    return cache, registry, symbols


class StandaloneCudaForceTests(unittest.TestCase):
    def test_exact_ten_profile_static_contract(self) -> None:
        verify_contract(*fixtures())

    def test_disabled_native_profiles_fail(self) -> None:
        cache, registry, symbols = fixtures()
        with self.assertRaisesRegex(ValueError, "NATIVE_STATIONARY_CUDA"):
            verify_contract(
                cache.replace(
                    "GENERATIVEQC_ENABLE_NATIVE_STATIONARY_CUDA:BOOL=ON",
                    "GENERATIVEQC_ENABLE_NATIVE_STATIONARY_CUDA:BOOL=OFF",
                ), registry, symbols
            )

    def test_python_shared_artifacts_are_not_distribution_dependencies(self) -> None:
        cache, registry, symbols = fixtures()
        with self.assertRaisesRegex(ValueError, "STATIONARY_FORCE_AOT"):
            verify_contract(
                cache.replace(
                    "GENERATIVEQC_ENABLE_STATIONARY_FORCE_AOT:BOOL=OFF",
                    "GENERATIVEQC_ENABLE_STATIONARY_FORCE_AOT:BOOL=ON",
                ), registry, symbols
            )

    def test_missing_registered_profile_fails(self) -> None:
        cache, registry, symbols = fixtures()
        with self.assertRaisesRegex(ValueError, "registry mismatch"):
            verify_contract(cache, registry.replace("&native_stationary_profile_b3lyp_uks(),", ""), symbols)

    def test_linker_discarded_profile_fails(self) -> None:
        cache, registry, symbols = fixtures()
        with self.assertRaisesRegex(ValueError, "missing native force symbol"):
            verify_contract(cache, registry, symbols.replace(
                "generativeqc::dft::native_stationary_profile_r2scan_uks()", "discarded()",
            ))

    def test_missing_registry_entry_point_fails(self) -> None:
        cache, registry, symbols = fixtures()
        with self.assertRaisesRegex(ValueError, "registry entry point"):
            verify_contract(cache, registry, symbols.replace(
                "generativeqc::dft::cuda_stationary_bindings()", "discarded()",
            ))


if __name__ == "__main__":
    unittest.main()
