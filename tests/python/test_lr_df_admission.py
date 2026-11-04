"""Native cutoff admission must precede energy and survive geometry owners."""

from __future__ import annotations

import os
import subprocess
from contextlib import closing
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from generativeqc import (
    BasisProvenance,
    BasisSet,
    BasisShell,
    Calculator,
    ElementBasis,
    GridSpec,
    KsOptions,
)
from generativeqc._ks_snapshot import NativeKsSnapshot

if TYPE_CHECKING:
    from conftest import NativeCxx


def test_native_factor_reports_response_branch_without_changing_values(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    """An independently specified diagonal spectrum distinguishes rank and safety."""
    root = Path(__file__).resolve().parents[2]
    source = tmp_path / "admission.cpp"
    source.write_text(r"""
#include "integrals/density_fitting_metric.hpp"
#include <cmath>
#include <stdexcept>
#include <vector>
int main() {
  using namespace generativeqc::integrals;
  constexpr double cutoff = 1e-10;
  for (const double smallest : {0.5e-10, 1e-10, 2e-10}) {
    const std::vector<double> metric{smallest, 0, 0, 1};
    const auto factor = factor_density_fitting_metric(metric, 2, cutoff);
    const bool safe = smallest != cutoff;
    if (factor.response_valid != safe) return 1;
    if (factor.effective_rank != (smallest > cutoff ? 2U : 1U)) return 2;
    if (factor.inverse_square_root[3] != 1) return 3;
    const double expected = smallest > cutoff ? 1 / std::sqrt(smallest) : 0;
    if (factor.inverse_square_root[0] != expected) return 4;
    bool accepted = true;
    try {
      density_fitting_metric_response(
          metric, factor.inverse_square_root, std::vector<double>{1, 0, 0, 1},
          2, cutoff, generativeqc::tensor::SymmetricMatrixFunction::inverse_sqrt);
    } catch (const std::runtime_error&) { accepted = false; }
    if (accepted != safe) return 5;
  }
}
""")
    executable = native_cxx.build_executable(
        [
            source,
            root / "src/integrals/density_fitting_metric.cpp",
            root / "src/tensor/cpu_linalg.cpp",
            root / "src/tensor/symmetric_matrix_function.cpp",
        ],
        tmp_path / "admission",
        compile_args=("-std=c++20", "-O2", f"-I{root / 'src'}"),
    )
    subprocess.run([str(executable)], check=True, timeout=30)


@pytest.mark.parametrize("device", ("cpu", "cuda"))
@pytest.mark.parametrize("initially_safe", (False, True))
@pytest.mark.parametrize("selector", (None, "auto"), ids=("default", "auto"))
def test_automatic_lr_choice_is_frozen_across_cutoff_failure_and_replay(
    device: str,
    initially_safe: bool,
    selector: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve the unsafe initial branch once, never switch on a later force error."""
    if device == "cuda" and os.environ.get("GENERATIVEQC_DFT_CUDA_TEST") != "1":
        pytest.skip("requires an explicitly scheduled CUDA device")
    if device == "cuda":
        assert os.environ.get("SLURM_JOB_ID") or os.environ.get("LSB_JOBID")
    from test_mixed_rsh_df_forces import _mixed_pyscf_energy

    unsafe, basis, threshold = _boundary_fixture()
    safe = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.7))]
    calculator = Calculator(
        method="wb97m-v-rks",
        basis=basis,
        auxiliary_basis=basis,
        device=device,
        density_fitting="auto",
        density_fitting_relative_threshold=threshold,
        initial_guess=None,
        ks_options=KsOptions(
            grid=GridSpec(radial_points=12, angular_polar=4, angular_azimuth=8)
        ),
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        screening_tolerance=1e-14,
        max_iterations=200,
    )
    if selector is None:
        monkeypatch.delenv("GENERATIVEQC_DF_RANGE_EXCHANGE", raising=False)
    else:
        monkeypatch.setenv("GENERATIVEQC_DF_RANGE_EXCHANGE", selector)
    initial = safe if initially_safe else unsafe
    with calculator.prepare_batch([initial], warm_start=True) as batch:
        monkeypatch.setenv("GENERATIVEQC_DF_RANGE_EXCHANGE", "direct")
        cold = batch.execute(strict=True, properties=("energy",)).items[0]
        with closing(NativeKsSnapshot(batch, 0)) as snapshot:
            assert snapshot.range_density_fitted == initially_safe
        assert cold.energy == pytest.approx(
            _mixed_pyscf_energy(calculator, initial, 0, fitted_range=initially_safe),
            abs=2e-8,
        )
        if initially_safe:
            assert (
                batch.execute(
                    coordinates=([position for _, position in unsafe],),
                    strict=True,
                    properties=("energy",),
                )
                .items[0]
                .converged
            )
            with closing(NativeKsSnapshot(batch, 0)) as snapshot:
                assert snapshot.range_density_fitted
                with pytest.raises(RuntimeError, match="rank crossing"):
                    snapshot.mixed_density_fitted_integral_derivatives(2, 256 << 20)
            # Both retries fail under the same fitted Hamiltonian. A fallback
            # here would make energy/forces depend on the requested properties.
            for _attempt in range(2):
                with pytest.raises((RuntimeError, ValueError)):
                    batch.execute(
                        coordinates=([position for _, position in unsafe],),
                        strict=True,
                        properties=("energy", "forces"),
                    )
        for geometry in (safe, safe, initial):
            result = batch.execute(
                coordinates=([position for _, position in geometry],),
                strict=True,
                properties=("energy", "forces"),
            ).items[0]
            assert result.converged
            with closing(NativeKsSnapshot(batch, 0)) as snapshot:
                assert snapshot.range_density_fitted == initially_safe
            assert result.energy == pytest.approx(
                _mixed_pyscf_energy(
                    calculator, geometry, 0, fitted_range=initially_safe
                ),
                abs=2e-8,
            )


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_explicit_lr_fit_does_not_borrow_automatic_exact_fallback(
    device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit fit retains the existing value-only/response-error distinction."""
    if device == "cuda" and os.environ.get("GENERATIVEQC_DFT_CUDA_TEST") != "1":
        pytest.skip("requires an explicitly scheduled CUDA device")
    if device == "cuda":
        assert os.environ.get("SLURM_JOB_ID") or os.environ.get("LSB_JOBID")
    atoms, basis, threshold = _boundary_fixture()
    monkeypatch.setenv("GENERATIVEQC_DF_RANGE_EXCHANGE", "df")
    calculator = Calculator(
        method="wb97m-v-rks",
        basis=basis,
        auxiliary_basis=basis,
        device=device,
        density_fitting="auto",
        density_fitting_relative_threshold=threshold,
        initial_guess=None,
        ks_options=KsOptions(
            grid=GridSpec(radial_points=12, angular_polar=4, angular_azimuth=8)
        ),
        max_iterations=200,
    )
    with calculator.prepare_batch([atoms], warm_start=True) as batch:
        assert batch.execute(strict=True, properties=("energy",)).items[0].converged
        with closing(NativeKsSnapshot(batch, 0)) as snapshot:
            assert snapshot.range_density_fitted
            with pytest.raises(RuntimeError, match="rank crossing"):
                snapshot.mixed_density_fitted_integral_derivatives(2, 256 << 20)
        with pytest.raises((RuntimeError, ValueError)):
            batch.execute(strict=True, properties=("energy", "forces"))


def _boundary_fixture() -> tuple[list, BasisSet, float]:
    """Share exact primitive inputs with libcint, not rounded same-name tables."""
    gto = pytest.importorskip("pyscf.gto")
    atoms = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.5))]
    basis = BasisSet(
        "synthetic-single-Gaussian-H",
        (ElementBasis(1, (BasisShell(0, ("0.7",), (("1",),)),)),),
        BasisProvenance(
            "synthetic test: H 1s exponent=0.7 coefficient=1",
            "1",
            "CC0",
            sha256(b"H 1s exponent=0.7 coefficient=1").hexdigest(),
        ),
    )
    auxiliary = gto.M(
        atom=atoms, basis={"H": [[0, [0.7, 1.0]]]}, unit="Bohr", verbose=0
    )
    with auxiliary.with_range_coulomb(0.3):
        spectrum = np.linalg.eigvalsh(auxiliary.intor("int2c2e"))
    return atoms, basis, float(spectrum[0] / spectrum[-1])
