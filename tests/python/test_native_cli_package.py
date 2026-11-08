"""Distribution tests needing neither CUDA nor the full C++ build."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

from tools.package_native_cli import package_native_cli


@unittest.skipUnless(sys.platform == "linux", "Native distribution targets Linux ELF")
class NativeCliPackageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.prefix = self.root / "install"
        self.source = self.root / "source"
        (self.prefix / "bin").mkdir(parents=True)
        (self.source / "LICENSES").mkdir(parents=True)
        (self.source / "LICENSE").write_text("test license\n")
        (self.source / "THIRD_PARTY_NOTICES.md").write_text("test notices\n")
        (self.source / "LICENSES" / "dependency.txt").write_text("dependency license\n")
        # Packaging needs real ELF bytes, not another uncached fixture compile.
        shutil.copy2("/bin/true", self.prefix / "bin" / "generativeqc")
        self.build_metadata = (
            self.prefix / "share" / "generativeqc" / "native-cli-build.json"
        )
        self.build_metadata.parent.mkdir(parents=True)
        self.configuration = {
            "schema_version": 1,
            "version": "0.1.0",
            "system_name": "Linux",
            "system_processor": "x86_64",
            "cuda_compiler_id": "NVIDIA",
            "cuda_toolkit": "12.9.86",
            "cuda_architectures": ["120"],
        }
        self.build_metadata.write_text(json.dumps(self.configuration))

    def build(self, destination: str, release: bool = False) -> Path:
        return package_native_cli(
            prefix=self.prefix,
            source_root=self.source,
            output_dir=self.root / destination,
            version="0.1.0",
            commit="1234567890abcdef" * 2 + "12345678",
            release=release,
            source_date_epoch=1000000000,
        )

    def test_package_contains_one_cli_not_static_archives(self) -> None:
        libs = self.prefix / "lib"
        libs.mkdir()
        (libs / "libgenerativeqc.a").write_text("static archive should not ship")
        archive = self.build("dist", release=True)
        with tarfile.open(archive, "r:gz") as bundle:
            names = set(bundle.getnames())
            root = archive.name.removesuffix(".tar.gz")
            self.assertIn(f"{root}/bin/generativeqc", names)
            self.assertIn(f"{root}/LICENSE", names)
            self.assertIn(f"{root}/LICENSES/dependency.txt", names)
            self.assertNotIn(f"{root}/lib/libgenerativeqc.a", names)
            self.assertFalse(any(".so" in name for name in names))
            self.assertEqual(
                [name for name in names if name.endswith("/bin/generativeqc")],
                [f"{root}/bin/generativeqc"],
            )
            metadata = json.load(bundle.extractfile(f"{root}/build-metadata.json"))
            self.assertEqual(metadata["core_linkage"], "static")
            self.assertEqual(metadata["cuda_toolkit"], "12.9.86")
            self.assertEqual(metadata["cuda_architectures"], ["120"])
            self.assertFalse(metadata["generativeqc_shared_core_required"])
            self.assertFalse(metadata["separately_packaged_aot_plugins"])

        checksum = (archive.parent / "SHA256SUMS").read_text()
        self.assertEqual(
            checksum,
            f"{hashlib.sha256(archive.read_bytes()).hexdigest()}  {archive.name}\n",
        )

    def test_cuda_release_cli_packages_entire_automatic_xc_catalog(self) -> None:
        """Release artifacts must not silently omit automatic CUDA XC programs."""
        root = Path(__file__).resolve().parents[2]
        presets = json.loads((root / "CMakePresets.json").read_text())
        by_name = {preset["name"]: preset for preset in presets["configurePresets"]}
        release_cli = by_name["cuda-release-cli"]
        self.assertEqual(release_cli["inherits"], "cuda-release")
        self.assertEqual(
            by_name["cuda-release"]["cacheVariables"]["GENERATIVEQC_AOT_PROFILE"],
            "auto",
        )
        for preset in by_name.values():
            self.assertNotIn(
                "GENERATIVEQC_CUDA_COMPILE_ARCHITECTURES",
                preset["cacheVariables"],
            )
            self.assertNotIn("CMAKE_CUDA_ARCHITECTURES", preset["cacheVariables"])
        options = release_cli["cacheVariables"]
        self.assertEqual(options["GENERATIVEQC_ENABLE_AUTOMATIC_XC_CUDA_AOT"], "ON")
        self.assertEqual(options["GENERATIVEQC_AUTOMATIC_XC_CUDA_FUNCTIONALS"], "")

    def test_cuda_catalog_verification_respects_docs_only_cli_skip(self) -> None:
        """Catalog verification must follow the optional CLI build tier."""
        root = Path(__file__).resolve().parents[2]
        workflow = (root / ".github/workflows/ci.yml").read_text()
        step = workflow.split(
            "      - name: Verify full automatic CUDA XC catalog is compiled for distribution\n",
            1,
        )[1].split("\n      - ", 1)[0]
        self.assertIn("        if: steps.cli_scope.outputs.tier != 'skip'\n", step)

    def test_rejects_unexpected_shared_aot_plugin(self) -> None:
        libs = self.prefix / "lib"
        libs.mkdir()
        (libs / "libgenerativeqc_stationary_pbe_rks.so").write_bytes(b"ELF fixture")
        with self.assertRaisesRegex(ValueError, "unexpected shared object"):
            self.build("unexpected")

    def test_same_inputs_generate_identical_archive(self) -> None:
        first = self.build("dist1")
        second = self.build("dist2")
        self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_requires_elf_executable(self) -> None:
        (self.prefix / "bin" / "generativeqc").write_text("#!/bin/sh\nexit 0\n")
        with self.assertRaisesRegex(ValueError, "ELF CLI"):
            self.build("invalid")

    def test_cmake_targets_drive_package_identity(self) -> None:
        """Exercise the actual CMake producer and archive consumer without CUDA."""
        root = Path(__file__).resolve().parents[2]
        cases = (
            ("12.9.86", "120", "cuda12.9-sm120"),
            (
                "13.0.88",
                "80;90-real;120-virtual",
                "cuda13.0-sm80+sm90-real+sm120-virtual",
            ),
            ("13.1.0", "90-real;90-virtual", "cuda13.1-sm90-real+sm90-virtual"),
        )
        for index, (toolkit, architectures, label) in enumerate(cases):
            with self.subTest(architectures=architectures):
                script = self.root / "metadata.cmake"
                script.write_text(
                    "cmake_minimum_required(VERSION 3.24)\n"
                    'set(PROJECT_VERSION "0.1.0")\n'
                    'set(CMAKE_SYSTEM_NAME "Linux")\n'
                    'set(CMAKE_SYSTEM_PROCESSOR "x86_64")\n'
                    'set(CMAKE_CUDA_COMPILER_ID "NVIDIA")\n'
                    f'set(CMAKE_CUDA_COMPILER_VERSION "{toolkit}")\n'
                    f'set(CMAKE_CUDA_ARCHITECTURES "{architectures}")\n'
                    f'include("{root / "cmake/GenerativeQCNativeCLIMetadata.cmake"}")\n'
                    f'generativeqc_write_native_cli_metadata("{self.build_metadata}")\n'
                )
                subprocess.run(["cmake", "-P", str(script)], check=True, timeout=30)
                archive = self.build(f"target-{index}", release=True)
                self.assertEqual(
                    archive.name, f"generativeqc-v0.1.0-linux-x86_64-{label}.tar.gz"
                )
                with tarfile.open(archive, "r:gz") as bundle:
                    package_root = archive.name.removesuffix(".tar.gz")
                    metadata = json.load(
                        bundle.extractfile(f"{package_root}/build-metadata.json")
                    )
                    self.assertEqual(metadata["cuda_toolkit"], toolkit)
                    self.assertEqual(
                        metadata["cuda_architectures"], architectures.split(";")
                    )
                    readme = (
                        bundle.extractfile(f"{package_root}/README.txt").read().decode()
                    )
                    self.assertIn(f"CUDA {toolkit}", readme)
                    self.assertIn(", ".join(architectures.split(";")), readme)
                    if "120" not in architectures:
                        self.assertNotIn("Blackwell", readme)
                        self.assertNotIn("sm_120", readme)

    def test_requires_installed_build_metadata(self) -> None:
        self.build_metadata.unlink()
        with self.assertRaisesRegex(ValueError, "build metadata is required"):
            self.build("missing-metadata")
        self.assertFalse((self.root / "missing-metadata").exists())

    def test_rejects_ambiguous_or_mismatched_build_metadata(self) -> None:
        cases = (
            ("schema_version", 2, "schema"),
            ("version", "0.2.0", "version does not match"),
            ("system_name", "Darwin", "Linux x86_64"),
            ("system_processor", "aarch64", "Linux x86_64"),
            ("cuda_compiler_id", "Clang", "NVIDIA CUDA"),
            ("cuda_toolkit", "", "numeric CUDA toolkit"),
            ("cuda_toolkit", "13.0/../../", "numeric CUDA toolkit"),
            ("cuda_architectures", [], "resolved CUDA architectures"),
            ("cuda_architectures", "120", "resolved CUDA architectures"),
            ("cuda_architectures", [120], "resolved CUDA architectures"),
            ("cuda_architectures", ["native"], "resolved CUDA architectures"),
            ("cuda_architectures", ["all"], "resolved CUDA architectures"),
            ("cuda_architectures", ["sm_120"], "resolved CUDA architectures"),
            ("cuda_architectures", ["120-invalid"], "resolved CUDA architectures"),
            ("cuda_architectures", ["120", "../90"], "resolved CUDA architectures"),
        )
        for key, value, error in cases:
            with self.subTest(key=key, value=value):
                self.build_metadata.write_text(
                    json.dumps({**self.configuration, key: value})
                )
                with self.assertRaisesRegex(ValueError, error):
                    self.build("invalid-metadata")
                self.assertFalse((self.root / "invalid-metadata").exists())


if __name__ == "__main__":
    unittest.main()
