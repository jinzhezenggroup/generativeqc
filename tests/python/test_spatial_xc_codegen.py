"""Independent numerical gates for generated spatial XC point adapters."""

import ctypes as ct
import shutil
import subprocess
import threading
import typing
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from generativeqc_compiler.common.provenance import canonical_hash
from generativeqc_compiler.dft import ExplicitGrid, NativeAO
from generativeqc_compiler.dft.ao_cuda import emit_grid_source
from generativeqc_compiler.dft.xc_kernel import GRID_XC_POINT_TYPE
from generativeqc_compiler.xc import functional
from generativeqc_compiler.xc.spatial_cuda import (
    compiled_spatial_xc,
    emit_spatial_xc,
    spatial_xc_capability,
)
from generativeqc_compiler.xc.spec import (
    PUBLIC_COMPONENTS,
    FunctionalSpec,
    UnsupportedXC,
)

SPATIAL_CASES = [
    *PUBLIC_COMPONENTS,
    "LDA_XC_PW",
    "PBE",
    "SCAN",
    "R2SCAN",
    "HALF_PBE",
    "BLEND",
]


def spatial_spec(name: str) -> FunctionalSpec:
    """Include modified names and mixed families to detect name-only dispatch."""
    if name == "HALF_PBE":
        return replace(
            functional("PBE"),
            components=tuple(
                (key, weight / 2) for key, weight in functional("PBE").components
            ),
        )
    if name == "BLEND":
        return FunctionalSpec(
            "mixed-gga-mgga",
            (
                ("GGA_X_PBE", Fraction(3, 10)),
                ("GGA_C_PBE", Fraction(2, 5)),
                ("MGGA_X_R2SCAN", Fraction(7, 10)),
                ("MGGA_C_R2SCAN", Fraction(3, 5)),
            ),
        )
    return functional(name)


class AnalyticSpatialBasis(NativeAO):
    """Independent Gaussian inputs for the real grid/runtime consumer.

    Normalization is deliberately unity: both packed coefficients and analytic
    reference jets describe exp(-alpha*r^2), with no native library dependency.
    This isolates the resident XC path from unrelated molecular build artifacts.
    """

    def __init__(self, nao: int = 3) -> None:
        self._lock = threading.RLock()
        self._handle = ct.c_void_p()
        self.nao, self.natom, self.nprimitive = nao, 1, nao
        self.atoms = ((1, (0.0, 0.0, 0.0)),)
        self.charge, self.multiplicity, self.representation = 0, 1, "cartesian"
        self.shells = tuple(
            SimpleNamespace(angular_momentum=0, atom_index=0) for _ in range(nao)
        )
        self.exponents = np.linspace(0.4, 1.7, nao)
        records = np.zeros((nao, 16))
        records[:, 1] = np.arange(nao)
        records[:, 2:4] = 1
        records[:, 7] = 1
        self.packed = np.concatenate(
            (
                np.zeros(3),
                np.stack((self.exponents, np.ones(nao)), axis=1).ravel(),
                records.ravel(),
            )
        )
        self.identity = canonical_hash({"gaussian_test_basis": self.packed.tolist()})
        self.numeric_bytes = self.packed.nbytes

    def evaluate(
        self, points: np.ndarray, order: int = 1, *, ao_ids: typing.Any = None
    ) -> np.ndarray:
        assert order in (0, 1)
        values = np.exp(-np.sum(points**2, axis=1)[:, None] * self.exponents)
        jets = np.stack(
            [values]
            + [
                -2 * points[:, axis, None] * self.exponents * values
                for axis in range(3)
            ]
        )
        jets = jets[: 1 if order == 0 else 4]
        return jets if ao_ids is None else jets[:, :, ao_ids]

    def close(self) -> None:
        pass


def spatial_case(
    nao: int = 3, npoint: int = 13, restricted: bool = False
) -> tuple[AnalyticSpatialBasis, ExplicitGrid, np.ndarray]:
    rng = np.random.default_rng(1117)
    basis = AnalyticSpatialBasis(nao)
    points = rng.uniform(-0.6, 0.6, (npoint, 3))
    grid = ExplicitGrid(points, rng.uniform(0.1, 0.5, npoint), (0,) * npoint, {})
    factor = rng.normal(size=(nao, nao)) * 0.1 + np.eye(nao)
    density = np.stack(
        (factor @ factor.T, (1.0 if restricted else 0.7) * factor @ factor.T)
    )
    return basis, grid, density


def independent_xc(
    spec: FunctionalSpec, jets: np.ndarray, density: np.ndarray, weights: np.ndarray
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Libxc energy and full AO potential, with independent contractions."""
    libxc = pytest.importorskip("pyscf.dft.libxc")
    phi, gradients = jets[0], jets[1:4]
    rho = np.einsum("pi,sij,pj->sp", phi, density, phi)
    grad = 2 * np.einsum("kpi,sij,pj->skp", gradients, density, phi)
    tau = 0.5 * np.einsum("kpi,sij,kpj->sp", gradients, density, gradients)
    packed = np.zeros((2, 6, len(weights)))
    packed[:, 0], packed[:, 1:4], packed[:, 5] = rho, grad, tau
    expression = "+".join(
        f"{float(coefficient):.17g}*{name}"
        for name, coefficient in spec.components
        if coefficient
    )
    # Libxc expects family-dependent density widths; derivatives use its
    # physical spin convention independently of the generated scalar graph.
    family = libxc.xc_type(expression)
    inputs = (
        packed[:, :1]
        if family == "LDA"
        else packed[:, :4]
        if family == "GGA"
        else packed
    )
    exc, derivatives, _, _ = libxc.eval_xc(expression, inputs, spin=1, deriv=1)
    vrho = derivatives[0].T
    coefficients = np.zeros((11, len(weights)))
    coefficients[0] = rho.sum(axis=0) * exc
    coefficients[1:3] = vrho
    if family != "LDA":
        vsigma = derivatives[1].T
        coefficients[3:6] = 2 * vsigma[0] * grad[0] + vsigma[1] * grad[1]
        coefficients[6:9] = 2 * vsigma[2] * grad[1] + vsigma[1] * grad[0]
    if family == "MGGA":
        coefficients[9:11] = 0.5 * derivatives[3].T
    potential = np.einsum("p,sp,pi,pj->sij", weights, vrho, phi, phi)
    for spin in range(2):
        for axis in range(3):
            weighted = weights * coefficients[3 + 3 * spin + axis]
            leg = np.einsum("p,pi,pj->ij", weighted, gradients[axis], phi)
            potential[spin] += leg + leg.T
            potential[spin] += np.einsum(
                "p,pi,pj->ij",
                weights * coefficients[9 + spin],
                gradients[axis],
                gradients[axis],
            )
    features = np.zeros((13, len(weights)))
    for spin in range(2):
        features[5 * spin] = rho[spin]
        features[5 * spin + 1 : 5 * spin + 4] = grad[spin]
        features[5 * spin + 4] = tau[spin]
    return (
        {
            "energy": weights @ coefficients[0],
            "electrons": rho @ weights,
            "potential": potential,
        },
        features,
        coefficients,
    )


def host_adapter(tmp_path: Path, spec: FunctionalSpec) -> typing.Callable:
    """Execute the exact emitted device adapter as C++ for a CPU oracle gate."""
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    code = "#include <cmath>\nusing I = long long;\nusing std::isfinite;\n#define __device__\n"
    code += GRID_XC_POINT_TYPE + emit_spatial_xc(spec).source
    code += """
extern "C" int evaluate(const double* features, long long n, int restricted, double* output) {
  for (I p = 0; p < n; ++p) {
    const auto value = grid_custom_xc_point(restricted != 0, features, n, p);
    if (!value.valid) return 1;
    output[p] = value.energy;
    for (int s = 0; s < 2; ++s) {
      output[(1+s)*n+p] = value.rho[s];
      output[(9+s)*n+p] = value.kinetic[s];
      for (int k = 0; k < 3; ++k) output[(3+3*s+k)*n+p] = value.gradient[s][k];
    }
  }
  return 0;
}
"""
    source, library = tmp_path / "adapter.cpp", tmp_path / "adapter.so"
    source.write_text(code)
    cache = [shutil.which("ccache")] if shutil.which("ccache") else []
    subprocess.run(
        [
            *cache,
            compiler,
            "-std=c++17",
            "-O2",
            "-ffp-contract=off",
            "-shared",
            "-fPIC",
            str(source),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
    )
    lib = ct.CDLL(str(library))
    pointer = ct.POINTER(ct.c_double)
    lib.evaluate.argtypes = [pointer, ct.c_longlong, ct.c_int, pointer]

    def evaluate(
        features: np.ndarray, restricted: bool = False
    ) -> tuple[int, np.ndarray]:
        features = np.ascontiguousarray(features)
        output = np.empty((11, features.shape[1]))
        status = lib.evaluate(
            features.ctypes.data_as(pointer),
            features.shape[1],
            restricted,
            output.ctypes.data_as(pointer),
        )
        return status, output

    return evaluate


@pytest.mark.parametrize("name", SPATIAL_CASES)
def test_generated_adapter_matches_independent_libxc(tmp_path: Path, name: str) -> None:
    spec = spatial_spec(name)
    basis, grid, density = spatial_case()
    _, features, expected = independent_xc(
        spec, basis.evaluate(grid.points), density, grid.weights
    )
    status, actual = host_adapter(tmp_path, spec)(features)
    assert status == 0
    np.testing.assert_allclose(actual, expected, atol=3e-10, rtol=3e-10)


def test_device_domain_rejects_before_nonlinear_evaluation(tmp_path: Path) -> None:
    spec = functional("R2SCAN")
    evaluate = host_adapter(tmp_path, spec)
    basis, grid, density = spatial_case(restricted=True)
    _, features, _ = independent_xc(
        spec, basis.evaluate(grid.points), density, grid.weights
    )
    assert evaluate(features, restricted=True)[0] == 0
    for row, value in ((0, 0.0), (0, 1e13), (4, -1.0), (4, np.nan), (1, 1e30)):
        invalid = features.copy()
        invalid[row, 0] = value
        assert evaluate(invalid)[0] == 1
    invalid = features.copy()
    invalid[9, 0] *= 1.001
    assert evaluate(invalid, restricted=True)[0] == 1


def test_exact_composition_identity_and_unsupported_domain() -> None:
    spec = spatial_spec("HALF_PBE")
    capability = compiled_spatial_xc(spec)
    assert spatial_xc_capability(spec) is None
    assert spatial_xc_capability(spec, capability.identity) == capability
    assert (
        spatial_xc_capability(replace(spec, spin="unpolarized"), capability.identity)
        == capability
    )
    assert spatial_xc_capability(functional("R2SCAN"), capability.identity) is None
    assert spatial_xc_capability(functional("PBE"), capability.identity).evaluator == 1
    with pytest.raises(UnsupportedXC, match="domain"):
        compiled_spatial_xc(replace(spec, exact_exchange=Fraction(1, 4)))
    with pytest.raises(UnsupportedXC, match="domain"):
        compiled_spatial_xc(FunctionalSpec("B88", (("GGA_X_B88", Fraction(1)),)))
    baseline = emit_grid_source()[1]
    custom = emit_grid_source(xc_kernel=emit_spatial_xc(spec))[1]
    assert custom != baseline
    with pytest.raises(ValueError, match="native KS"):
        emit_grid_source(native_ks=True, xc_kernel=emit_spatial_xc(spec))
