"""Independent forces and fallback ownership for small cold RHF source reuse."""

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("SLURM_JOB_ID")
    or os.environ.get("GENERATIVEQC_RESOURCE_CUDA_TEST") != "1",
    reason="requires an explicitly Slurm-allocated GPU",
)


@pytest.mark.parametrize("representation", ["spherical", "cartesian"])
def test_resident_cold_warm_moved_and_capacity(
    representation: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default admission and capacity refusal preserve the final derivative density.

    Reference densities never seed native SCF. Both AO representations exercise
    final metadata production; moved/frozen replays use the ordinary fallback.
    """
    from generativeqc import Calculator
    from pyscf import gto, lib, scf

    from benchmarks.readme_hf_scaling import scaling_cases

    lib.num_threads(1)
    atoms = scaling_cases()["water-12"].atoms
    moved = [
        (s, tuple(np.asarray(x) + ([0.02, 0, 0] if i == 1 else [0, 0, 0])))
        for i, (s, x) in enumerate(atoms)
    ]
    references = []
    for geometry in (atoms, moved):
        mol = gto.M(
            atom=geometry,
            unit="Bohr",
            basis="def2-svp",
            cart=representation == "cartesian",
            verbose=0,
        )
        reference = scf.RHF(mol)
        reference.conv_tol, reference.conv_tol_grad = 1e-11, 1e-8
        reference.kernel()
        assert reference.converged
        references.append((reference.e_tot, -reference.nuc_grad_method().kernel()))

    def check(item: Any, index: int) -> None:
        assert item.converged and item.succeeded and item.executed_backend == "cuda"
        assert abs(item.energy - references[index][0]) < 1e-8
        np.testing.assert_allclose(item.forces, references[index][1], atol=1e-7, rtol=0)
        assert item.precision["operator_work_counters_valid"]

    monkeypatch.delenv("GENERATIVEQC_RHF_RESIDENT_VALUES", raising=False)
    # The one-byte request refuses optional values while retaining exact forces.
    for ceiling in ("536870912", "1"):
        trace = tmp_path / ("trace-" + ceiling + ".jsonl")
        monkeypatch.setenv("GENERATIVEQC_DF_PROGRESS_TRACE", str(trace))
        monkeypatch.setenv("GENERATIVEQC_RHF_RESIDENT_MAX_BYTES", ceiling)
        calculator = Calculator(
            method="rhf",
            basis="def2-svp",
            basis_representation=representation,
            device="cuda",
            energy_tolerance=1e-12,
            density_tolerance=1e-10,
            screening_tolerance=1e-12,
        )
        try:
            with calculator.prepare_batch([atoms], warm_start=True) as owner:
                check(
                    owner.execute(properties=("energy", "forces"), strict=True).items[
                        0
                    ],
                    0,
                )
                entries = [json.loads(line) for line in trace.read_text().splitlines()]
                built = [
                    row["value"]
                    for row in entries
                    if row.get("key") == "source_values_built"
                ]
                assert bool(max(built, default=0)) == (ceiling != "1")
                assert any(
                    row.get("key") == "selection" and row["value"] == "auto"
                    for row in entries
                )
                owner.set_warm_start_updates(False)
                check(
                    owner.execute(properties=("energy", "forces"), strict=True).items[
                        0
                    ],
                    0,
                )
                coordinates = [[x for _, x in moved]]
                check(
                    owner.execute(
                        coordinates, properties=("energy", "forces"), strict=True
                    ).items[0],
                    1,
                )
                check(
                    owner.execute(
                        coordinates, properties=("energy", "forces"), strict=True
                    ).items[0],
                    1,
                )
        finally:
            calculator.clear_cache()
