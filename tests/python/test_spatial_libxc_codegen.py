"""Libxc work-domain transport into the common bounded spatial point panel."""

from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.xc.spatial_program import emit_libxc_spatial
from generativeqc_compiler.xc.spec import FunctionalSpec, UnsupportedXC
from test_libxc_automatic_point_native import _inputs, _oracle
from test_spatial_xc_codegen import host_adapter, independent_xc, spatial_case

LIBXC_CASES = ("LDA_C_PZ", "GGA_X_B88", "GGA_C_LYP", "MGGA_X_MVS", "MN15", "M06-2X")


@pytest.mark.parametrize(
    "names",
    [
        ("LDA_C_BR78", "LDA_C_WIGNER"),
        ("GGA_X_APBE", "GGA_X_PBE_SOL"),
        ("MGGA_X_KTBM_0", "MGGA_X_KTBM_1"),
    ],
)
def test_shared_parameter_package_keeps_independent_science(
    tmp_path: Path, names: tuple[str, ...]
) -> None:
    """One immutable worker/table can serve distinct owners and both spins."""
    from generativeqc_compiler.xc.parameter_cuda_package import parameter_worker_groups

    (group,) = parameter_worker_groups(names)
    points = [emit_libxc_spatial(name, shared_group=group) for name in names]
    assert points[0].kernel.source == points[1].kernel.source
    assert points[0].kernel.consumer.identity == points[1].kernel.consumer.identity
    assert points[0].metadata != points[1].metadata
    for index, point in enumerate(points):
        folder = tmp_path / str(index)
        folder.mkdir()
        evaluate = host_adapter(
            folder,
            point.spec,
            source=point.kernel.source,
            selector=point.kernel.consumer.evaluator - 2,
        )
        for restricted in (False, True):
            inputs = _inputs(names[index], 0 if restricted else 1)
            status, actual = evaluate(physical_features(inputs), restricted)
            assert status == 0
            if restricted:
                actual = actual[[0, 1, 3, 4, 5, 9]]
            np.testing.assert_allclose(
                actual.T,
                _oracle(names[index], inputs, 0 if restricted else 1),
                atol=3e-10,
                rtol=3e-10,
            )


def physical_features(inputs: np.ndarray) -> np.ndarray:
    """Pack independent physical density/Cartesian gradients/tau into grid rows."""
    features = np.zeros((13, len(inputs)))
    features[[0, 5]] = inputs[:, :2].T
    features[1:4], features[6:9] = inputs[:, 2:5].T, inputs[:, 5:8].T
    features[[4, 9]] = inputs[:, 8:10].T
    return features


@pytest.mark.parametrize("name", LIBXC_CASES)
@pytest.mark.parametrize("restricted", (False, True))
def test_libxc_adapter_interior(tmp_path: Path, name: str, restricted: bool) -> None:
    point = emit_libxc_spatial(name, spin="unpolarized" if restricted else "polarized")
    basis, grid, density = spatial_case(restricted=restricted)
    _, features, expected = independent_xc(
        point.spec, basis.evaluate(grid.points), density, grid.weights
    )
    evaluate = host_adapter(tmp_path, point.spec, source=point.kernel.source)
    status, actual = evaluate(features, restricted)
    assert status == 0
    np.testing.assert_allclose(actual, expected, atol=3e-10, rtol=3e-10)
    for row, value in ((0, -1.0), (5, np.nan), (0, np.inf)):
        invalid = features.copy()
        invalid[row, 0] = value
        assert evaluate(invalid)[0] == 1
    if "gradient" in point.kernel.consumer.ingredients:
        for value in (np.nan, 1e300):
            invalid = features.copy()
            invalid[1, 0] = value
            assert evaluate(invalid)[0] == 1
    if "tau" in point.kernel.consumer.ingredients:
        invalid = features.copy()
        invalid[4, 0] = -1.0
        assert evaluate(invalid)[0] == 1
    unequal = features.copy()
    unequal[5, 0] *= 1.01
    assert evaluate(unequal, True)[0] == 1


@pytest.mark.parametrize("name", LIBXC_CASES[:4])
@pytest.mark.parametrize("restricted", (False, True))
def test_automatic_work_boundaries(tmp_path: Path, name: str, restricted: bool) -> None:
    """Test vacuum, empty spin, floors, zero gradient and Fermi-hole clipping.

    Restricted LDA uses the independent U1 oracle, including the floor interval
    where folding the P2 worker would change energy and vxc.
    """
    from generativeqc_compiler.xc.libxc_work import automatic_work_policy

    point = emit_libxc_spatial(name)
    inputs = _inputs(name, 0 if restricted else 1)
    if restricted and point.spec.ingredients == ("rho",):
        floor = automatic_work_policy(name).density_threshold
        inputs[7:10, :2] = np.array([0.5, 0.75, 1.0])[:, None] * floor
    expected = _oracle(name, inputs, 0 if restricted else 1)
    status, actual = host_adapter(tmp_path, point.spec, source=point.kernel.source)(
        physical_features(inputs), restricted
    )
    assert status == 0
    if restricted:
        np.testing.assert_allclose(actual[1], actual[2], atol=1e-13, rtol=1e-13)
        actual = actual[[0, 1, 3, 4, 5, 9]]
    np.testing.assert_allclose(actual.T, expected, atol=3e-10, rtol=3e-10)


def test_mn15_work_boundaries(tmp_path: Path) -> None:
    """MN15's complete adversarial point set has a finite binary64 Libxc oracle."""
    from tools.validate_split_hybrid_cuda import (
        libxc_density,
        point_inputs,
        reference_points,
    )

    points, _, _ = point_inputs()
    packed = libxc_density(points)
    features = np.zeros((13, len(points)))
    for spin in range(2):
        features[5 * spin] = packed[spin, 0]
        features[5 * spin + 1 : 5 * spin + 4] = packed[spin, 1:4]
        features[5 * spin + 4] = packed[spin, 5]
    raw = reference_points("268,269", points)
    expected = np.zeros((11, len(points)))
    expected[:3] = raw[:, :3].T
    expected[3:6] = 2 * raw[:, 3] * packed[0, 1:4] + raw[:, 4] * packed[1, 1:4]
    expected[6:9] = raw[:, 4] * packed[0, 1:4] + 2 * raw[:, 5] * packed[1, 1:4]
    expected[9:11] = 0.5 * raw[:, 6:8].T
    point = emit_libxc_spatial("MN15")
    status, actual = host_adapter(tmp_path, point.spec, source=point.kernel.source)(
        features
    )
    assert status == 0
    np.testing.assert_allclose(actual, expected, atol=3e-9, rtol=3e-9)


def test_exact_composition_and_domain_identity(tmp_path: Path) -> None:
    spec = FunctionalSpec(
        "imported-mix", (("GGA_X_B88", Fraction(1, 3)), ("GGA_C_LYP", Fraction(2, 5)))
    )
    point = emit_libxc_spatial(spec)
    basis, grid, density = spatial_case()
    _, features, expected = independent_xc(
        spec, basis.evaluate(grid.points), density, grid.weights
    )
    status, actual = host_adapter(tmp_path, spec, source=point.kernel.source)(features)
    assert status == 0
    np.testing.assert_allclose(actual, expected, atol=3e-10, rtol=3e-10)
    changed = replace(
        spec, components=(("GGA_X_B88", Fraction(2, 3)), ("GGA_C_LYP", Fraction(2, 5)))
    )
    assert (
        emit_libxc_spatial(changed).kernel.consumer.identity
        != point.kernel.consumer.identity
    )
    with pytest.raises(UnsupportedXC, match="exchange operators"):
        emit_libxc_spatial(replace(spec, exact_exchange=Fraction(1, 4)))
    with pytest.raises(UnsupportedXC, match="range"):
        emit_libxc_spatial(replace(spec, range_omega=Fraction(1, 3)))
    with pytest.raises(UnsupportedXC, match="pair"):
        emit_libxc_spatial("MGGA_C_MN15")
    with pytest.raises(UnsupportedXC):
        emit_libxc_spatial("MGGA_X_BR89")  # Requires a Laplacian outside this ABI.


def test_source_generation_without_runtime_or_oracle() -> None:
    """The Libxc adapter remains usable from a compiler-only source checkout."""
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[2]
    code = f"""
import importlib.abc, sys
sys.path.insert(0, {str(root / "python")!r})
class RejectRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('generativeqc', 'pyscf', 'torch', 'cupy'):
            raise AssertionError('source generation imported runtime/oracle: ' + fullname)
sys.meta_path.insert(0, RejectRuntime())
from generativeqc_compiler.xc.spatial_program import emit_libxc_spatial
for name in {LIBXC_CASES!r}:
    point = emit_libxc_spatial(name)
    assert point.kernel.consumer.evaluator == 2
    assert point.metadata['consumer'] == point.kernel.consumer.identity
"""
    subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd="/",
        check=True,
        capture_output=True,
        text=True,
    )
