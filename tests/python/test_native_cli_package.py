"""Distribution tests needing neither CUDA nor the full C++ build."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path

from tools.package_native_cli import package_native_cli


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
        native = self.root / "native.c"
        native.write_text("int main(void) { return 0; }\n")
        subprocess.run(
            ["cc", "-o", str(self.prefix / "bin" / "generativeqc"), str(native)],
            check=True,
        )

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
            self.assertFalse(metadata["generativeqc_shared_core_required"])
            self.assertFalse(metadata["separately_packaged_aot_plugins"])

        checksum = (archive.parent / "SHA256SUMS").read_text()
        self.assertEqual(
            checksum,
            f"{hashlib.sha256(archive.read_bytes()).hexdigest()}  {archive.name}\n",
        )

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


if __name__ == "__main__":
    unittest.main()
