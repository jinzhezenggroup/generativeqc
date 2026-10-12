"""Native optimization CLI contracts; Python is only the test driver."""

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path


def invoke(executable: str, xyz: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [executable, "optimize", xyz, *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


def main(executable: str, xyz: str) -> None:
    plan = invoke(
        executable,
        xyz,
        "--method",
        "ccsd(t)",
        "--basis",
        "sto-3g",
        "--backend",
        "cpu",
        "--plan-only",
    )
    assert plan.returncode == 0, plan.stderr
    payload = json.loads(plan.stdout)
    assert payload["status"] == "plan_only"
    assert [s["method"] for s in payload["stages"]] == [
        "gfn2-xtb",
        "pbe-rks",
        "rccsd(t)",
    ]
    assert payload["stages"][0]["backend"] == "cpu"
    assert payload["stages"][0]["precision"] == "fp64"
    assert (
        payload["stages"][-1]["force_tolerance"]
        < payload["stages"][0]["force_tolerance"]
    )

    hybrid = invoke(
        executable,
        xyz,
        "--method",
        "pbe0-rks",
        "--backend",
        "cpu",
        "--preopt-basis",
        "sto-3g",
        "--plan-only",
    )
    assert hybrid.returncode == 0, hybrid.stderr
    assert [s["method"] for s in json.loads(hybrid.stdout)["stages"]] == [
        "gfn2-xtb",
        "pbe-rks",
        "pbe0-rks",
    ]

    # A final xTB stage preserves requested backend and precision. Unsupported
    # execution modes must reach native admission rather than be substituted.
    xtb_target = invoke(
        executable,
        xyz,
        "--method",
        "gfn2-xtb",
        "--backend",
        "cuda",
        "--precision",
        "auto",
        "--plan-only",
    )
    assert xtb_target.returncode == 0, xtb_target.stderr
    xtb_plan = json.loads(xtb_target.stdout)["stages"]
    assert len(xtb_plan) == 1 and xtb_plan[0]["backend"] == "cuda"
    assert xtb_plan[0]["precision"] == "auto"
    assert xtb_plan[0]["optional_if_unsupported"] is False

    custom = invoke(
        executable, xyz, "--method", "rhf", "--stages", "rhf", "--plan-only"
    )
    assert custom.returncode == 0, custom.stderr
    assert [s["method"] for s in json.loads(custom.stdout)["stages"]] == ["rhf"]

    bad = invoke(
        executable, xyz, "--method", "rhf", "--stages", "pbe-rks", "--plan-only"
    )
    assert bad.returncode == 2 and "final --stages method" in bad.stderr
    bad = invoke(executable, xyz, "--method", "rhf", "--opt-fmax", "-1")
    assert bad.returncode == 2

    # End-to-end finite native CPU forces, no xTB/CUDA prerequisite.
    result = invoke(
        executable,
        xyz,
        "--method",
        "rhf",
        "--basis",
        "sto-3g",
        "--backend",
        "cpu",
        "--units",
        "bohr",
        "--stages",
        "rhf",
        "--opt-fmax",
        "0.02",
        "--opt-max-steps",
        "80",
        "--json",
    )
    assert result.returncode == 0, result.stderr
    value = json.loads(result.stdout)
    assert value["status"] == "converged"
    assert value["method"] == "rhf"
    assert value["max_force_hartree_per_bohr"] <= 0.02
    assert value["stages"][-1]["status"] == "converged"
    assert value["stages"][-1]["backend"] == "cpu"
    assert value["stages"][-1]["evaluations"] >= 1
    assert len(value["coordinates_bohr"]) == 2

    # The final XYZ is Angstrom and exclusively created, including symlinks.
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "optimized.xyz"
        output_args = (
            "--method",
            "rhf",
            "--basis",
            "sto-3g",
            "--backend",
            "cpu",
            "--units",
            "bohr",
            "--stages",
            "rhf",
            "--opt-fmax",
            "0.02",
            "--json",
        )
        written = invoke(executable, xyz, *output_args, "--output", str(output))
        assert written.returncode == 0, written.stderr
        contents = output.read_text(encoding="utf-8")
        rows = contents.splitlines()
        coordinates = json.loads(written.stdout)["coordinates_bohr"]
        assert int(rows[0]) == len(coordinates) == 2
        for row, atom in zip(rows[2:], coordinates, strict=True):
            fields = row.split()
            assert fields[0] == "1"
            for actual, expected in zip(fields[1:], atom, strict=True):
                assert math.isclose(
                    float(actual), expected / 1.8897261254578281, abs_tol=1e-14
                )
        existing = invoke(executable, xyz, *output_args, "--output", str(output))
        assert existing.returncode == 2 and "refusing to overwrite" in existing.stderr
        assert output.read_text(encoding="utf-8") == contents
        if sys.platform != "win32":
            dangling = Path(directory) / "dangling.xyz"
            referent = Path(directory) / "must-not-create.xyz"
            dangling.symlink_to(referent)
            rejected = invoke(executable, xyz, *output_args, "--output", str(dangling))
            assert rejected.returncode == 2
            assert "refusing to overwrite" in rejected.stderr
            assert dangling.is_symlink() and not referent.exists()

    # Exercise the real native CPU CCSD(T) analytic-gradient owner, rather
    # than only asserting that it appears in the proposed stage plan.
    ccsdt = invoke(
        executable,
        xyz,
        "--method",
        "ccsd(t)",
        "--basis",
        "sto-3g",
        "--backend",
        "cpu",
        "--units",
        "bohr",
        "--stages",
        "ccsd(t)",
        "--opt-fmax",
        "0.05",
        "--json",
    )
    assert ccsdt.returncode == 0, ccsdt.stderr
    ccsdt_value = json.loads(ccsdt.stdout)
    assert ccsdt_value["status"] == "converged"
    assert ccsdt_value["method"] == "rccsd(t)"
    assert ccsdt_value["stages"][-1]["method"] == "rccsd(t)"
    assert ccsdt_value["max_force_hartree_per_bohr"] <= 0.05


if __name__ == "__main__":
    main(*sys.argv[1:])
