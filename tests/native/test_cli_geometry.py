"""Native optimization CLI contracts; Python is only the test driver."""

import json
import subprocess
import sys


def invoke(executable, xyz, *args):
    return subprocess.run(
        [executable, "optimize", xyz, *args],
        capture_output=True, text=True, check=False, timeout=120,
    )


def main(executable, xyz):
    plan = invoke(executable, xyz, "--method", "ccsd(t)", "--basis", "sto-3g",
                  "--backend", "cpu", "--plan-only")
    assert plan.returncode == 0, plan.stderr
    payload = json.loads(plan.stdout)
    assert payload["status"] == "plan_only"
    assert [s["method"] for s in payload["stages"]] == [
        "gfn2-xtb", "pbe-rks", "rccsd(t)"
    ]
    assert payload["stages"][0]["backend"] == "cpu"
    assert payload["stages"][-1]["force_tolerance"] < payload["stages"][0]["force_tolerance"]

    hybrid = invoke(executable, xyz, "--method", "pbe0-rks", "--backend", "cpu",
                    "--preopt-basis", "sto-3g", "--plan-only")
    assert hybrid.returncode == 0, hybrid.stderr
    assert [s["method"] for s in json.loads(hybrid.stdout)["stages"]] == [
        "gfn2-xtb", "pbe-rks", "pbe0-rks"
    ]

    custom = invoke(executable, xyz, "--method", "rhf", "--stages", "rhf",
                    "--plan-only")
    assert custom.returncode == 0, custom.stderr
    assert [s["method"] for s in json.loads(custom.stdout)["stages"]] == ["rhf"]

    bad = invoke(executable, xyz, "--method", "rhf", "--stages", "pbe-rks",
                 "--plan-only")
    assert bad.returncode == 2 and "final --stages method" in bad.stderr
    bad = invoke(executable, xyz, "--method", "rhf", "--opt-fmax", "-1")
    assert bad.returncode == 2

    # End-to-end finite native CPU forces, no xTB/CUDA prerequisite.
    result = invoke(executable, xyz, "--method", "rhf", "--basis", "sto-3g",
                    "--backend", "cpu", "--units", "bohr", "--stages", "rhf",
                    "--opt-fmax", "0.02", "--opt-max-steps", "80", "--json")
    assert result.returncode == 0, result.stderr
    value = json.loads(result.stdout)
    assert value["status"] == "converged"
    assert value["method"] == "rhf"
    assert value["max_force_hartree_per_bohr"] <= 0.02
    assert value["stages"][-1]["status"] == "converged"
    assert value["stages"][-1]["evaluations"] >= 1
    assert len(value["coordinates_bohr"]) == 2


if __name__ == "__main__":
    main(*sys.argv[1:])
