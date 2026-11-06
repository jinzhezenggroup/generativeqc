"""Opt-in six-site CPU qualification; never a production source rewrite gate.

The explicit alternatives below are disposable experimental translation units.
Production objects are not overwritten. Exact generated sources, compiler/link
commands, raw component records and complete endpoint observations are retained.
No candidate timing or inventory count grants scientific acceptance by itself.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import lzma
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "python"), str(ROOT / "tests/python")]

from generativeqc_compiler.common.materialization import MaterializationRuleRequest

HERE = Path(__file__).resolve().parent
COMMON = ROOT / "src/posthf/mp2_derivative_common.cpp"
MODES = {
    "retain": 0,
    "first": 1,
    "second": 2,
    "third": 3,
    "local-view": 4,
    "helper-retain": 5,
}


def run(command: list[str], *, cwd: Path = ROOT) -> str:
    """Bound each compiler/probe/endpoint invocation and retain stderr on failure."""
    result = subprocess.run(
        command, cwd=cwd, text=True, capture_output=True, timeout=300, check=False
    )
    if result.returncode:
        raise RuntimeError(f"{shlex.join(command)}\n{result.stdout}\n{result.stderr}")
    return result.stdout


def variant(mode: int, *, observed: bool) -> str:
    """Apply a reviewed, site-specific alternative, not arbitrary C++ rewriting."""
    probe = importlib.import_module("test_mp2_derivative_pullback_probe")
    text = probe.instrument(COMMON.read_text()) if observed else COMMON.read_text()
    if mode == 0:
        return text
    begin = text.index("void transform_remaining_shells(")
    end = text.index("\n}  // namespace", begin)
    helper = (HERE / "remaining_shells.cpp.in").read_text().replace("@MODE@", str(mode))
    text = text[:begin] + helper + text[end:]
    call = "dense_two, compact_first);"
    assert text.count(call) == 1
    text = text.replace(call, "dense_two, compact_first, weights);")
    if mode == 1:
        begin = text.index("    std::vector<double> first(compact_first")
        end = text.index("    transform_remaining_shells(", begin)
        text = text[:begin] + "    std::vector<double> first;\n" + text[end:]
    counter = "(++probe::counts[INDEX])" if observed else "((void)0)"
    return "#define MATERIALIZATION_COUNT(INDEX) " + counter + "\n" + text


def qualification_probe(directory: Path, launcher: str) -> dict[str, Any]:
    """Run the real callback contracts against direct and libcint derivatives."""
    probe_module = importlib.import_module("test_mp2_derivative_pullback_probe")
    text = (ROOT / "tests/native/mp2_derivative_pullback_probe.cpp").read_text()
    text = text.replace(
        'require(allocation::live == persistent + shell_bytes, "active quartet allocation ownership");',
        'require(allocation::live >= persistent, "persistent allocation ownership");',
    )
    removed = (
        'require(allocation::peak == e.peak, "allocation high-water mismatch");',
        'require(allocation::calls == e.allocation_calls, "allocation call mismatch");',
        'require(allocation::total == e.total_bytes, "allocation total mismatch");',
        'require(measured_shell_peak == e.shell_peak, "shell allocation peak mismatch");',
        'if (probe::instrumented)\n        require(probe::counts == e.iterations, "executed accumulation count mismatch");',
    )
    for anchor in removed:
        assert text.count(anchor) == 1, anchor
        text = text.replace(anchor, "")
    harness = directory / "probe.cpp"
    harness.write_text(text)
    (directory / "probe_counts.hpp").write_text(
        "#pragma once\n#include <array>\n#include <cstddef>\n"
        "namespace probe { inline std::array<std::size_t,12> counts{}; "
        "inline constexpr bool instrumented = true; }\n"
    )
    flags = [
        "-std=c++20",
        "-O2",
        "-ffunction-sections",
        "-fdata-sections",
        "-DGENERATIVEQC_HAS_OPENBLAS=0",
        "-I" + str(ROOT / "src"),
        "-I" + str(ROOT / "include"),
        "-I" + str(directory),
    ]
    objects = []
    for source in [
        harness,
        ROOT / "src/molecule/basis.cpp",
        ROOT / "src/tensor/cpu_linalg.cpp",
        ROOT / "src/posthf/mp2_gradient.cpp",
    ]:
        target = directory / (source.stem + ".o")
        run([launcher, "c++", *flags, "-c", str(source), "-o", str(target)])
        objects.append(str(target))
    executables = []
    records = {}
    for name, mode in MODES.items():
        source = directory / (name + "-observed.cpp")
        source.write_text(variant(mode, observed=True))
        target = directory / (name + "-observed.o")
        executable = directory / (name + "-probe")
        run([launcher, "c++", *flags, "-c", str(source), "-o", str(target)])
        run(["c++", *objects, str(target), "-Wl,--gc-sections", "-o", str(executable)])
        executables.append(executable)
        records[name] = []
        for case in range(8):
            for representation in (0, 1, 3, 2):
                for sparse in (0, 1):
                    record = json.loads(
                        run(
                            [
                                str(executable),
                                str(case),
                                str(representation),
                                str(sparse),
                            ]
                        )
                    )
                    records[name].append(
                        {
                            "case": case,
                            "representation": representation,
                            "sparse": sparse,
                            **record,
                        }
                    )
    physical = directory / "physical"
    physical.mkdir()
    for case in range(8):
        for sparse in (False, True):
            probe_module.test_physical_eri_derivative_pullback(
                executables, physical, case, sparse
            )
    return {
        "physical_libcint_cases": 8,
        "physical_sparse_populations": 2,
        "physical_representations": [1, 3, 2],
        "records": records,
    }


def library_variants(build: Path, directory: Path) -> dict[str, Path]:
    """Relink only the experimental object; keep all production build inputs."""
    commands = run(["ninja", "-C", str(build), "-t", "commands"]).splitlines()
    common = next(
        command for command in commands if command.endswith("-c " + str(COMMON))
    )
    compile_args = shlex.split(common)
    common_object = compile_args[compile_args.index("-o") + 1]
    link = next(
        command
        for command in commands
        if " -shared " in command and " -o libgenerativeqc.so.0.1.0 " in command
    )
    link_args = shlex.split(link.split("&&")[1])
    force_source = ROOT / "src/posthf/mp2_force.cpp"
    force_command = shlex.split(
        next(
            command
            for command in commands
            if command.endswith("-c " + str(force_source))
        )
    )
    force_object = force_command[force_command.index("-o") + 1]
    all_commands = []

    def compile_source(command: list[str], source: Path, target: Path) -> None:
        args = list(command)
        args[args.index("-o") + 1] = str(target)
        args[args.index("-c") + 1] = str(source)
        for option in ("-MT", "-MF"):
            if option in args:
                index = args.index(option)
                del args[index : index + 2]
        args = [value for value in args if value != "-MD"]
        all_commands.append(args)
        run(args, cwd=build)

    dense = force_source.read_text()
    orbital = "auto orbital = canonical_orbital_rhs_streamed(reference, h, *provider, adjoint,\n                                                same_space_threshold, cuda, device_id);"
    assert dense.count(orbital) == 1
    dense = dense.replace(
        orbital,
        "(void)dense_orbital_rhs_plan(reference.nbf, reference.nocc, budget_bytes);\n"
        "  const auto all_orbitals = range(0, reference.nbf);\n"
        "  const auto oracle_eri = provider->get({all_orbitals, all_orbitals, all_orbitals, all_orbitals}, cuda, device_id);\n"
        "  auto orbital = canonical_orbital_rhs_with_budget(h, oracle_eri, adjoint, same_space_threshold, budget_bytes);",
    )
    weights = "auto weights = canonical_lagrangian_weights_streamed(reference, h, *provider, std::move(adjoint),\n                                                       std::move(orbital), response_result.solution,\n                                                       same_space_threshold, cuda, device_id);"
    assert dense.count(weights) == 1
    dense = dense.replace(
        weights,
        "auto weights = canonical_lagrangian_weights_with_budget(h, oracle_eri, adjoint, response_result.solution, same_space_threshold, budget_bytes);",
    )
    dense_source = directory / "dense-oracle-force.cpp"
    dense_source.write_text(dense)
    dense_object = directory / "dense-oracle-force.o"
    compile_source(force_command, dense_source, dense_object)
    libraries = {}
    for name, mode in {
        **MODES,
        "dense-oracle": 0,
        "dense-first": 1,
        "dense-oracle-helper": 5,
    }.items():
        source = directory / (name + ".cpp")
        source.write_text(variant(mode, observed=False))
        target = directory / (name + ".o")
        compile_source(compile_args, source, target)
        output = directory / (name + ".so")
        args = [
            str(target)
            if arg == common_object
            else str(dense_object)
            if name.startswith("dense-") and arg == force_object
            else arg
            for arg in link_args
        ]
        args[args.index("-o") + 1] = str(output)
        all_commands.append(args)
        run(args, cwd=build)
        libraries[name] = output
    (directory / "compiler-link-commands.json").write_text(
        json.dumps(all_commands, indent=2) + "\n"
    )
    return libraries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    launcher = shutil.which("sccache") or shutil.which("ccache")
    if not launcher:
        raise SystemExit("verified compiler cache is required")
    cache_version = run([launcher, "--version"])
    (directory / "cache-before.txt").write_text(run([launcher, "--show-stats"]))
    component = qualification_probe(directory, launcher)
    (directory / "component.json.xz").write_bytes(
        lzma.compress(json.dumps(component).encode())
    )
    libraries = library_variants(args.build.resolve(), directory)
    samples = []
    script = HERE / "endpoint.py"
    for candidate, control in [
        ("helper-retain", "retain"),
        ("first", "helper-retain"),
        ("second", "helper-retain"),
        ("third", "helper-retain"),
        ("local-view", "helper-retain"),
        ("dense-oracle", "retain"),
        ("dense-oracle-helper", "dense-oracle"),
        ("dense-first", "dense-oracle-helper"),
    ]:
        for repeat in range(2):
            for name in (control, candidate, candidate, control):
                output = directory / (f"endpoint-{len(samples):02d}-{name}.json")
                environment = {
                    **os.environ,
                    "PYTHONPATH": str(ROOT / "python") + ":" + str(ROOT),
                    "GENERATIVEQC_LIBRARY": str(libraries[name]),
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                }
                subprocess.run(
                    [sys.executable, str(script), str(output)],
                    env=environment,
                    check=True,
                    timeout=300,
                    cwd=ROOT,
                )
                samples.append(
                    {
                        "candidate": candidate,
                        "repeat": repeat,
                        "library": name,
                        "observations": json.loads(output.read_text()),
                    }
                )
    requests = []
    for site in (
        "dense-first",
        "factorized-first",
        "second",
        "third",
        "local",
        "dense-orbital-rhs",
    ):
        site_source = (
            ROOT / "src/posthf/mp2_gradient.cpp"
            if site == "dense-orbital-rhs"
            else COMMON
        )
        requests.append(
            MaterializationRuleRequest(
                origin="opt-in-native-CPU-endpoint-qualification",
                source_identity=hashlib.sha256(site_source.read_bytes()).hexdigest(),
                subject={"site": site},
                producer={"method": "MP2 analytic derivative"},
                consumers=[
                    {
                        "contract": "bounded dense orbital oracle"
                        if site == "dense-orbital-rhs"
                        else "contiguous borrowed shell cotangent"
                    }
                ],
                retained_output=site == "dense-orbital-rhs",
                allow_reassociation=False,
                precision_contract="FP64 / independent physical derivative and full E+F gates",
                constraints={"production_rewrite": False, "alternatives": list(MODES)},
            ).to_payload()
        )
    report = {
        "schema": "generativeqc.mp2-six-site-experiment.v1",
        "cache_version": cache_version,
        "compiler_rule_requests": requests,
        "library_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in libraries.items()
        },
        "samples": samples,
    }
    (directory / "qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    (directory / "cache-after.txt").write_text(run([launcher, "--show-stats"]))


if __name__ == "__main__":
    main()
