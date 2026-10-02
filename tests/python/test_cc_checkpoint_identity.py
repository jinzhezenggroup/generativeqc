"""Canonical CC checkpoints retain strict scientific and capability boundaries."""

from dataclasses import replace
from pathlib import Path

import pytest
from generativeqc import Calculator, ObservableTarget, ResolvedModel, TargetAccuracy
from generativeqc.checkpoint import CheckpointError
from generativeqc.projection import ProjectionRejected

ATOMS = [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]


@pytest.mark.parametrize(
    "method,canonical", [("rccsd", "rccsd"), ("ccsd(t)", "rccsd(t)")]
)
def test_cc_model_identity_is_closed_shell_conventional_only(
    method: str, canonical: str
) -> None:
    model = Calculator(method=method).resolved_model(ATOMS)
    assert model.method == canonical
    assert model.approximation == "conventional"
    assert ResolvedModel.from_dict(model.to_dict()) == model
    assert model.identity != Calculator(method="rhf").resolved_model(ATOMS).identity
    with pytest.raises(ValueError, match="closed-shell"):
        replace(model, multiplicity=3)
    with pytest.raises(ValueError, match="conventional"):
        replace(
            model,
            approximation="density_fitting",
            auxiliary_basis_hash="aux",
            metric_relative_threshold=1e-10,
        )
    target = TargetAccuracy(
        (ObservableTarget("energy", "absolute", "Eh", absolute=1e-6),)
    )
    with pytest.raises(NotImplementedError, match="target_accuracy"):
        Calculator(method=method, target_accuracy=target)
    with pytest.raises(NotImplementedError, match="frozen-core"):
        Calculator(method=method, ccsd_frozen_core=1)


@pytest.mark.parametrize(
    "source_method,target_method",
    [
        ("rccsd", "rhf"),
        ("rccsd", "mp2"),
        ("rccsd", "ccsd(t)"),
        ("ccsd(t)", "rhf"),
        ("ccsd(t)", "mp2"),
        ("ccsd(t)", "rccsd"),
    ],
)
def test_cc_checkpoint_rejects_cross_method_restore(
    tmp_path: Path, source_method: str, target_method: str
) -> None:
    checkpoint = tmp_path / "source.vqcp"
    with Calculator(method=source_method).prepare_batch([ATOMS]) as source:
        source.execute(properties=("energy",), strict=True)
        source.save_checkpoint(checkpoint)
    with Calculator(method=target_method).prepare_batch([ATOMS]) as target:
        with pytest.raises(CheckpointError, match="method/core/spin/provider"):
            target.load_checkpoint(checkpoint, allow_warm=True)
        assert (
            not target.execute(properties=("energy",), strict=True)
            .items[0]
            .warm_start_used
        )


@pytest.mark.parametrize("method", ["rccsd", "ccsd(t)"])
def test_cc_checkpoint_geometry_change_requires_warm_admission(
    tmp_path: Path, method: str
) -> None:
    checkpoint = tmp_path / "source.vqcp"
    moved = [("H", (0.0, 0.0, -0.8)), ("H", (0.0, 0.0, 0.8))]
    calculator = Calculator(method=method)
    with calculator.prepare_batch([ATOMS]) as source:
        source.execute(properties=("energy",), strict=True)
        source.save_checkpoint(checkpoint)
    with calculator.prepare_batch([moved]) as target:
        with pytest.raises(CheckpointError, match="geometry or numerical controls"):
            target.load_checkpoint(checkpoint)
        report = target.load_checkpoint(checkpoint, allow_warm=True)
        assert report["items"][0]["compatibility"] == "warm_start_compatible"
        warm = target.execute(properties=("energy",), strict=True).items[0]
        assert warm.warm_start_used and not warm.warm_start_fallback
    cold = calculator.singlepoint(moved, properties=("energy",))
    assert warm.energy == pytest.approx(cold.energy, abs=2e-9)


@pytest.mark.parametrize("method", ["rccsd", "ccsd(t)"])
def test_cc_identity_does_not_enable_progressive_projection(method: str) -> None:
    calculator = Calculator(method=method)
    with (
        calculator.prepare_batch([ATOMS]) as source,
        calculator.prepare_batch([ATOMS]) as target,
    ):
        source.execute(properties=("energy",), strict=True)
        with pytest.raises(ProjectionRejected, match="only RHF/UHF"):
            target.initialize_from(source)
        assert (
            not target.execute(properties=("energy",), strict=True)
            .items[0]
            .warm_start_used
        )
