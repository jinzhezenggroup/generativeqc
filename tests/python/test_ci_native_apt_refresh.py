"""Guard the optional native-dependency APT refresh against unsafe CI skips."""

from __future__ import annotations

import unittest
from pathlib import Path

from test_ci_prose_workflow import job, steps

ROOT = Path(__file__).resolve().parents[2]
ACTION = ROOT / ".github/actions/install-native-cli-deps/action.yml"
CUDA_OWNERS = ("cuda-compile", "cuda-resources", "cuda-direct-aot")
CPU_OWNERS = ("master-cpu-cache", "cpu", "python")


class NativeAptRefreshTests(unittest.TestCase):
    def test_default_installs_refresh_apt_before_native_libraries(self) -> None:
        action = ACTION.read_text(encoding="utf-8")
        self.assertIn("apt-index-refreshed:", action)
        self.assertIn("default: 'false'", action)
        self.assertIn("if: inputs.apt-index-refreshed != 'true'", action)
        self.assertIn("sudo apt-get update -o Acquire::Retries=3", action)
        self.assertIn(
            "sudo apt-get install --yes nlohmann-json3-dev libssl-dev libarchive-dev",
            action,
        )
        self.assertLess(
            action.index("- name: Refresh APT indices when required"),
            action.index("- name: Install native CLI libraries"),
        )
        for owner in CPU_OWNERS:
            with self.subTest(owner=owner):
                native = steps(owner)["Install native CLI dependencies"]
                self.assertIn("uses: ./.github/actions/install-native-cli-deps", native)
                self.assertNotIn("apt-index-refreshed:", native)

    def test_cuda_jobs_skip_only_the_second_apt_refresh(self) -> None:
        for owner in CUDA_OWNERS:
            with self.subTest(owner=owner):
                section = job(owner)
                native = steps(owner)["Install native CLI dependencies"]
                self.assertIn("uses: ./.github/actions/install-native-cli-deps", native)
                self.assertIn("apt-index-refreshed: 'true'", native)
                add_repo = section.index("- name: Add NVIDIA CUDA repository")
                install_cuda = section.index("- name: Install CUDA 12.9 build dependencies")
                install_native = section.index("- name: Install native CLI dependencies")
                self.assertLess(add_repo, install_cuda)
                self.assertLess(install_cuda, install_native)
                self.assertIn("sudo apt-get update", section[add_repo:install_cuda])
                self.assertIn("sudo apt-get install --yes", section[install_cuda:install_native])


if __name__ == "__main__":
    unittest.main()
