"""Execute the actual common C++ derivative pullback, with a direct tensor oracle.

This is bounded CPU correctness/allocation/work evidence, not force endpoint or
CUDA qualification. Test-only instrumentation counts executed accumulation sites;
a separately compiled, unmodified translation unit must produce identical data
and allocation receipts. Inputs and independent oracle storage are not tracked.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]
COMMON = ROOT / "src/posthf/mp2_derivative_common.cpp"
# Binding the entire source prevents newly added, uninstrumented work from being
# silently omitted. Source changes require review of all observation sites.
COMMON_SHA256 = "7cf56330b2c57adc4b4ab93ec992207da42c2befd119a19765333d11eeae1657"
# (exact original statement LHS, replacement LHS). The comma expression only
# observes execution; arithmetic and evaluation of the original RHS stay intact.
SITES = (
    ("first[((iu * n + q) * n + r) * n + s] +=", 0),
    ("first[coulomb] +=", 1),
    ("first[exchange] -=", 2),
    ("first[correlation] +=", 3),
    ("second[((iu * dj + jv) * n + r) * n + s] +=", 4),
    ("third[((iu * dj + jv) * dk + kw) * n + s] +=", 5),
    ("local[((iu * dj + jv) * dk + kw) * dl + lx] +=", 6),
    ("value *= orbit;", 8),
    ("target += reference.coefficients[nu * n + q]", 9),
    ("target += reference.coefficients[nu * n + r]", 10),
    ("target += reference.coefficients[nu * n + occupied + a]", 11),
)


def instrument(text: str) -> str:
    assert hashlib.sha256(text.encode()).hexdigest() == COMMON_SHA256, (
        "Common derivative source changed; review all probe observation sites "
        "before refreshing its SHA-256 binding"
    )
    for anchor, counter in SITES:
        assert text.count(anchor) == 1, f"Observation anchor changed: {anchor}"
        if counter == 8:
            replacement = "(++probe::counts[8], value) *= orbit;"
        elif counter >= 9:
            replacement = anchor.replace(
                "target +=", f"(++probe::counts[{counter}], target) +="
            )
        else:
            lhs, operator = anchor.rsplit(" ", 1)
            replacement = f"(++probe::counts[{counter}], {lhs}) {operator}"
        text = text.replace(anchor, replacement)
    anchor = "return weights[((a * n + b) * n + c) * n + d];"
    assert text.count(anchor) == 1
    text = text.replace(
        anchor, anchor.replace("return ", "return ++probe::counts[7], ")
    )
    return '#include "probe_counts.hpp"\n' + text


@pytest.fixture(scope="module")
def pullback_probes(
    tmp_path_factory: pytest.TempPathFactory, required_native_cxx: NativeCxx
) -> list[Path]:
    directory = tmp_path_factory.mktemp("mp2-common-pullback")
    original = COMMON.read_text()
    observed = directory / "observed.cpp"
    observed.write_text(instrument(original))
    header = directory / "probe_counts.hpp"
    header.write_text(
        "#pragma once\n#include <array>\n#include <cstddef>\n"
        "namespace probe {\ninline std::array<std::size_t,12> counts{};\n"
        "inline constexpr bool instrumented = PROBE_INSTRUMENTED;\n}\n"
    )
    args = (
        "-std=c++20",
        "-O2",
        "-ffunction-sections",
        "-fdata-sections",
        "-DGENERATIVEQC_HAS_OPENBLAS=0",
        "-I" + str(ROOT / "src"),
        "-I" + str(ROOT / "include"),
        "-I" + str(directory),
    )
    support = []
    for relative in (
        "src/molecule/basis.cpp",
        "src/tensor/cpu_linalg.cpp",
        "src/posthf/mp2_gradient.cpp",
    ):
        source = ROOT / relative
        obj = directory / (source.stem + ".o")
        required_native_cxx.compile_object(source, obj, args=args)
        support.append(obj)
    executables = []
    for mode, common in enumerate((COMMON, observed)):
        executable = directory / f"probe-{mode}"
        objects = []
        for index, source in enumerate(
            (common, ROOT / "tests/native/mp2_derivative_pullback_probe.cpp")
        ):
            obj = directory / f"probe-{mode}-{index}.o"
            required_native_cxx.compile_object(
                source, obj, args=(*args, f"-DPROBE_INSTRUMENTED={mode}")
            )
            objects.append(obj)
        required_native_cxx.link(
            (*objects, *support), executable, args=("-Wl,--gc-sections",)
        )
        executables.append(executable)
    return executables


@pytest.mark.parametrize(
    "case",
    range(8),
    ids=(
        "two-s",
        "single-p",
        "s-p",
        "p-s",
        "s-p-s-same-atom",
        "cartesian-d-s",
        "spherical-d-s",
        "single-spherical-f",
    ),
)
@pytest.mark.parametrize("factorized", (False, True), ids=("dense", "factorized"))
@pytest.mark.parametrize(
    "sparse", (False, True), ids=("full-coefficients", "zero-gates")
)
def test_actual_common_pullback(
    pullback_probes: list[Path], case: int, factorized: bool, sparse: bool
) -> None:
    receipts = []
    for executable in pullback_probes:
        result = subprocess.run(
            [str(executable), str(case), str(int(factorized)), str(int(sparse))],
            check=False,
            capture_output=True,
            text=True,
            timeout=45,
            env={**os.environ, "GENERATIVEQC_DF_PROGRESS_TRACE": ""},
        )
        assert result.returncode == 0, result.stdout + result.stderr
        receipts.append(json.loads(result.stdout))
    assert receipts[0].pop("iterations") == [0] * 12
    measured_iterations = receipts[1].pop("iterations")
    assert any(measured_iterations)
    # Includes every observed local weight, atom gradient, callback count and
    # allocation lifetime. Instrumentation must preserve these bit-for-bit.
    assert receipts[0] == receipts[1]
    if factorized:
        n = receipts[1]["n"]
        assert measured_iterations[4] == 0
        assert measured_iterations[5:7] == [n**5] * 2
    else:
        assert measured_iterations[7] == 8 * measured_iterations[0]

    if factorized:
        for executable in pullback_probes:
            staged_run = subprocess.run(
                [str(executable), str(case), "3", str(int(sparse))],
                check=True,
                capture_output=True,
                text=True,
                timeout=45,
                env={**os.environ, "GENERATIVEQC_DF_PROGRESS_TRACE": ""},
            )
            staged = json.loads(staged_run.stdout)
            assert staged["weights"] == receipts[0]["weights"]
            assert staged["gradient"] == receipts[0]["gradient"]
            assert staged["callbacks"] == receipts[0]["callbacks"]
            assert staged["local_elements"] == receipts[0]["local_elements"]
            assert receipts[0]["shell_peak_bytes"] < staged["shell_peak_bytes"]
            assert sum(measured_iterations[9:12]) < n**5
        expanded_receipts = []
        for executable in pullback_probes:
            result = subprocess.run(
                [str(executable), str(case), "2", str(int(sparse))],
                check=False,
                capture_output=True,
                text=True,
                timeout=45,
                env={**os.environ, "GENERATIVEQC_DF_PROGRESS_TRACE": ""},
            )
            assert result.returncode == 0, result.stdout + result.stderr
            expanded = json.loads(result.stdout)
            expanded.pop("iterations")
            expanded_receipts.append(expanded)
        assert expanded_receipts[0] == expanded_receipts[1]
        assert expanded_receipts[0]["gradient"] == pytest.approx(
            receipts[0]["gradient"], rel=3e-11, abs=3e-11
        )


@pytest.mark.parametrize("case", range(8))
@pytest.mark.parametrize("sparse", (False, True), ids=("full", "zero-gates"))
def test_physical_eri_derivative_pullback(
    pullback_probes: list[Path], tmp_path: Path, case: int, sparse: bool
) -> None:
    """Use independent molecular ERI center derivatives, including every slot."""
    np = pytest.importorskip("numpy")
    gto = pytest.importorskip("pyscf.gto")
    angular = ((0, 0), (1,), (0, 1), (1, 0), (0, 1, 0), (2, 0), (2, 0), (3,))[case]
    atoms = ((0, 1), (0,), (0, 1), (1, 0), (1, 0, 1), (0, 1), (0, 1), (0,))[case]
    basis = {}
    for shell, (atom, momentum) in enumerate(zip(atoms, angular, strict=True)):
        basis.setdefault(f"He{atom}", []).append([momentum, [0.8 + 0.13 * shell, 1.0]])
    molecule = gto.M(
        atom=[
            ("He0", (0.1, -0.3, 0.2)),
            ("He1", (0.8, 0.4, 1.5)),
            ("He2", (-0.4, 0.2, 2.1)),
        ],
        basis=basis,
        unit="Bohr",
        cart=case < 6,
        verbose=0,
    )
    offsets = molecule.ao_loc_nr()
    order = []
    for shell, atom in enumerate(atoms):
        matches = [
            i
            for i in range(molecule.nbas)
            if molecule.bas_atom(i) == atom
            and abs(molecule.bas_exp(i)[0] - (0.8 + 0.13 * shell)) < 1e-12
        ]
        assert len(matches) == 1
        index = matches[0]
        order.extend(range(offsets[index], offsets[index + 1]))
    # libcint differentiates the first AO with respect to its electronic
    # coordinate; a center derivative has the opposite sign. Exact ERI symmetry
    # maps it to the other three slots without a production derivative oracle.
    first = -molecule.intor("int2e_ip1", comp=3)
    first = first[np.ix_(np.arange(3), order, order, order, order)]
    slots = (
        first,
        first.transpose(0, 2, 1, 3, 4),
        first.transpose(0, 3, 4, 1, 2),
        first.transpose(0, 3, 4, 2, 1),
    )
    payload = tmp_path / "eri-center-derivatives.bin"
    np.ascontiguousarray(np.concatenate(slots), dtype=np.float64).tofile(payload)
    receipts = []
    for executable in pullback_probes:
        for representation in (1, 3, 2):
            run = subprocess.run(
                [
                    str(executable),
                    str(case),
                    str(representation),
                    str(int(sparse)),
                    str(payload),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=45,
                env={**os.environ, "GENERATIVEQC_DF_PROGRESS_TRACE": ""},
            )
            receipts.append(json.loads(run.stdout))
    for receipt in receipts:
        assert receipt["gradient"] == pytest.approx(
            receipts[0]["gradient"], rel=3e-11, abs=3e-11
        )
    assert receipts[0]["weights"] == receipts[1]["weights"]
    assert receipts[0]["gradient"] == receipts[1]["gradient"]
