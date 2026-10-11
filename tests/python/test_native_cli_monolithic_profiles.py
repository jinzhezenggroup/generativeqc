"""Host-only profile admission through the real native profile implementation."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(
    sys.platform != "linux",
    reason="Native distribution and ELF interposition target Linux",
)


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@pytest.fixture(scope="module")
def profile_commands(native_cxx: object, tmp_path_factory: object) -> dict:
    root = tmp_path_factory.mktemp("native-profile-linkage")
    # profiles.cpp only needs this opaque pointer type from options.hpp. No
    # method routing or scientific implementation is replaced by this fixture.
    (root / "cli_method_catalog.hpp").write_text(
        "namespace generativeqc::cli::method_generated { struct Method; }\n"
    )
    harness = root / "profile_command.cpp"
    harness.write_text(
        '#include "cli/profiles.cpp"\n'
        "int main(int argc, char** argv) {\n"
        "  using namespace generativeqc::cli;\n"
        "  try {\n"
        '    if (argc > 1 && std::string_view(argv[1]) == "identity")\n'
        '      std::cout << compatible_identity(probe_device(0)).dump() << "\\n";\n'
        '    else if (argc > 1 && std::string_view(argv[1]) == "activate") {\n'
        "      activate_local_profile(argc, argv, 0);\n"
        '      std::cout << "activated\\n";\n'
        "    } else return profile_bundle_command(argc, argv);\n"
        "    return 0;\n"
        "  } catch (const std::exception& error) {\n"
        '    std::cerr << error.what() << "\\n"; return 2;\n'
        "  }\n}\n"
    )
    includes = ["-I", str(ROOT / "src"), "-I", str(ROOT / "include"), "-I", str(root)]
    if not Path("/usr/include/nlohmann/json.hpp").exists():
        json_headers = pytest.importorskip("nlohmann_json")
        includes += ["-I", str(json_headers.get_include())]
    flags = ["-std=c++20", *includes]
    json_object = root / "json.o"
    native_cxx.compile_object(ROOT / "src/cli/json.cpp", json_object, args=flags)
    probe = ROOT / "tests/native/cli_profile_probe.cpp"
    shared_core = root / "libgenerativeqc.so"
    native_cxx.build_shared([probe], shared_core, compile_args=flags)
    probe_object = root / "profile_probe.o"
    native_cxx.compile_object(probe, probe_object, args=flags)
    commands = {"library": shared_core}
    for monolithic in (False, True):
        name = "monolithic" if monolithic else "shared"
        obj = root / f"{name}.o"
        native_cxx.compile_object(
            harness,
            obj,
            args=[*flags, f"-DGENERATIVEQC_MONOLITHIC_CLI={int(monolithic)}"],
        )
        output = root / name
        native_cxx.link(
            [obj, json_object, probe_object if monolithic else shared_core],
            output,
            args=["-lcrypto", "-larchive", "-ldl", f"-Wl,-rpath,{root}"],
        )
        commands[name] = output
    return commands


@pytest.fixture
def profile_bundle(profile_commands: dict, tmp_path: Path) -> tuple:
    env = {
        **os.environ,
        "GENERATIVEQC_PROFILE_CACHE": str(tmp_path / "cache"),
        "PATH": str(tmp_path / "no-compiler"),
    }
    for name in (
        "GENERATIVEQC_PROFILE",
        "LD_PRELOAD",
        "GENERATIVEQC_NATIVE_PROFILE_REEXEC",
    ):
        env.pop(name, None)
    identity = json.loads(
        subprocess.check_output([profile_commands["monolithic"], "identity"], env=env)
    )
    schedule = {"kind": "shell_task", "block_threads": 128}
    kernel = {
        "shell_class": "ssss",
        "consumer": "fock",
        "schedule": schedule,
        "schedule_hash": _digest(schedule),
        "source_hash": "b" * 64,
        "gates": dict.fromkeys(
            ("target", "resources", "numerical", "performance", "endpoint"), "pass"
        ),
    }
    device = identity["device"]
    evidence = {
        "endpoint": {"passed": True},
        "candidates": [
            {
                "shell_class": "ssss",
                "consumer": "fock",
                "accepted": True,
                "selected_trial": "fixture",
                "tuning": {
                    "winners": [
                        {
                            "trial_key": "fixture",
                            "schedule": schedule,
                            "production_validation": {"accepted": True},
                        }
                    ]
                },
                "isolated": {
                    "passed": True,
                    "source_hash": kernel["source_hash"],
                    "schedule_hash": kernel["schedule_hash"],
                    "device": {
                        "name": device["name"],
                        "major": device["major"],
                        "minor": device["minor"],
                        "runtime": device["runtime_version"],
                        "driver": device["driver_version"],
                    },
                    "errors": {"synthetic": {"passed": True}},
                },
                "endpoint": {"passed": True},
            }
        ],
    }
    directory = tmp_path / "bundle"
    directory.mkdir()
    shutil.copyfile(profile_commands["library"], directory / "libgenerativeqc.so")
    (directory / "manifest.json").write_text("{}")
    (directory / "evidence.json").write_text(json.dumps(evidence))
    profile = {
        "schema": "generativeqc.local_profile",
        "schema_version": 1,
        "identity": identity,
        "toolchain": {"nvcc": "synthetic", "ptxas": "synthetic"},
        "kernels": [kernel],
        "artifacts": {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ("libgenerativeqc.so", "manifest.json", "evidence.json")
        },
    }
    (directory / "profile.json").write_text(json.dumps(profile))
    return directory, profile, env


def _run(command: Path, env: dict, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [command, *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_monolithic_explicit_profile_refuses_without_reexec(
    profile_commands: dict, profile_bundle: tuple
) -> None:
    directory, _, env = profile_bundle
    env["GENERATIVEQC_PROFILE"] = str(directory)
    result = _run(profile_commands["monolithic"], env, "activate")
    assert result.returncode == 2
    assert "use the shared-core CLI" in result.stderr
    assert "after native re-execution" not in result.stderr
    assert result.stdout == ""
    diagnostic = _run(profile_commands["monolithic"], env, "profile", "diagnose")
    payload = json.loads(diagnostic.stdout)
    assert payload["source"] == "official"
    assert payload["binary_profile_requires_shared_core"] is True
    assert payload["kernels"] == []


def test_monolithic_auto_profile_keeps_official_core(
    profile_commands: dict, profile_bundle: tuple
) -> None:
    directory, profile, env = profile_bundle
    cache = Path(env["GENERATIVEQC_PROFILE_CACHE"])
    destination = cache / "bundles" / _digest(profile)
    shutil.copytree(directory, destination)
    active = cache / "active.json"
    active.write_text(json.dumps({_digest(profile["identity"]): _digest(profile)}))
    before = active.read_bytes()
    result = _run(profile_commands["monolithic"], env, "activate")
    assert result.returncode == 0, result.stderr
    assert result.stdout == "activated\n"
    diagnostic = _run(profile_commands["monolithic"], env, "profile", "diagnose")
    payload = json.loads(diagnostic.stdout)
    assert payload["source"] == "official"
    assert "Monolithic CLI" in payload["rejected"][0]
    assert active.read_bytes() == before


def test_monolithic_install_preserves_existing_cache(
    profile_commands: dict, profile_bundle: tuple, tmp_path: Path
) -> None:
    directory, _, env = profile_bundle
    archive = tmp_path / "profile.zip"
    exported = _run(
        profile_commands["shared"],
        env,
        "profile",
        "export",
        str(directory),
        str(archive),
    )
    assert exported.returncode == 0, exported.stderr
    active = Path(env["GENERATIVEQC_PROFILE_CACHE"]) / "active.json"
    active.parent.mkdir()
    active.write_text('{"keep":"prior"}')
    before = active.read_bytes()
    result = _run(
        profile_commands["monolithic"], env, "profile", "install", str(archive)
    )
    assert result.returncode == 2
    assert "use the shared-core CLI" in result.stderr
    assert active.read_bytes() == before
    assert not (active.parent / "bundles").exists()


def test_shared_profile_activation_is_unchanged(
    profile_commands: dict, profile_bundle: tuple
) -> None:
    directory, _, env = profile_bundle
    env["GENERATIVEQC_PROFILE"] = str(directory)
    diagnostic = _run(profile_commands["shared"], env, "profile", "diagnose")
    assert json.loads(diagnostic.stdout)["source"] == "local", diagnostic.stderr
    result = _run(profile_commands["shared"], env, "activate")
    assert result.returncode == 0, result.stderr
    assert result.stdout == "activated\n"


def test_monolithic_profile_off_retains_core(
    profile_commands: dict, profile_bundle: tuple
) -> None:
    _, _, env = profile_bundle
    env["GENERATIVEQC_PROFILE"] = "off"
    result = _run(profile_commands["monolithic"], env, "activate")
    assert result.returncode == 0, result.stderr
