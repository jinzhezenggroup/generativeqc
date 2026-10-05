"""DFT consumers execute canonical bounded contractions and retain work evidence."""

from __future__ import annotations

import ctypes as ct
import os
import shutil
import subprocess
import typing

if typing.TYPE_CHECKING:
    from pathlib import Path

import numpy as np
import pytest
from _cc_owner_test_support import compile_owner
from generativeqc_compiler.dft.grid_contraction import (
    emit_grid_contraction,
    grid_panel_program,
)
from generativeqc_compiler.tensor import execute


@pytest.mark.parametrize("shape", [(1, 1, 1, 1), (3, 7, 5, 2), (4, 31, 17, 9)])
def test_canonical_panel_matches_independent_jet_products(
    shape: tuple[int, ...],
) -> None:
    jets, points, active, width = shape
    rng = np.random.default_rng(182)
    ao = rng.normal(size=(jets, points, active))
    coefficients = rng.normal(size=(active, width))
    actual = execute(
        grid_panel_program(*shape), {"ao_jets": ao, "coefficients": coefficients}
    )
    expected = np.stack([panel @ coefficients for panel in ao])
    np.testing.assert_allclose(
        actual.outputs["projected"], expected, atol=1e-12, rtol=1e-12
    )


def test_bounded_domain_rejects_shape_layout_precision_and_identity_changes(
    tmp_path: Path,
) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host compiler required")
    source = tmp_path / "domain.cpp"
    source.write_text(
        emit_grid_contraction()
        + r"""
using namespace generativeqc::tensor;
using generativeqc::dft::generated::grid_panel_descriptor;
int main() {
  BoundedContractionDomain domain(grid_panel_descriptor(4,31,17,9));
  for(std::size_t jets:{1,3,4}) for(std::size_t points:{1,7,31})
    for(std::size_t active:{1,5,17}) for(std::size_t width:{1,4,9}) {
      const auto r=grid_panel_descriptor(jets,points,active,width);
      domain.validate(r);
      if(r.summands()!=jets*points*active*width) return 1;
    }
  auto rejects=[&](auto r) {
    try {domain.validate(r);} catch(const std::exception&) {return true;} return false;
  };
  if(!rejects(grid_panel_descriptor(5,1,1,1))) return 2;
  if(!rejects(grid_panel_descriptor(1,32,1,1))) return 3;
  if(!rejects(grid_panel_descriptor(1,1,18,1))) return 4;
  if(!rejects(grid_panel_descriptor(1,1,1,10))) return 5;
  auto r=grid_panel_descriptor(3,7,5,2);
  r.operands[0].strides[1]++; if(!rejects(r)) return 6;
  r=grid_panel_descriptor(3,7,5,2);
  r.beta=1; if(!rejects(r)) return 7;
  r=grid_panel_descriptor(3,7,5,2);
  r.precision.compute_dtype=generativeqc::runtime::PrecisionDtype::Fp32;
  if(!rejects(r)) return 8;
  r=grid_panel_descriptor(3,7,5,2);
  r.semantic_template_identity=std::string_view("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff");
  if(!rejects(r)) return 9;
}
"""
    )
    binary = tmp_path / "domain"
    compile_owner(compiler, tmp_path, [source], binary)
    subprocess.run([str(binary)], check=True, capture_output=True, timeout=30)


@pytest.fixture(scope="module")
def fallback_artifact(tmp_path_factory: pytest.TempPathFactory) -> typing.Any:
    if os.environ.get("GENERATIVEQC_GRID_CUDA_TEST") != "1":
        pytest.skip("requires finite Slurm real-device qualification")
    from generativeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
    from generativeqc_compiler.common.cuda_target import cuda_target_info
    from generativeqc_compiler.common.native_runtime import compile_runtime
    from generativeqc_compiler.common.provenance import find_nvcc
    from generativeqc_compiler.dft.ao_cuda import emit_grid_source

    directory = tmp_path_factory.mktemp("grid-lowering")
    source, _, headers = emit_grid_source()
    path = directory / "grid.cu"
    path.write_text("#define GENERATIVEQC_TEST_HOOKS 1\n" + source)
    compiler = find_nvcc()
    assert compiler is not None
    return compile_runtime(
        CudaCompilerAdapter(compiler, cuda_target_info("sm_120")),
        directory,
        path,
        headers=headers,
        libraries=("cublas",),
        options=(
            "-std=c++20",
            "--fmad=false",
            f"-I{headers[0].parent}",
            f"-I{headers[0].parent.parent}",
            f"-I{headers[-1].parents[1]}",
        ),
    )


@pytest.mark.parametrize("unavailable", [False, True])
@pytest.mark.parametrize("route", ["density_matrix", "orbitals"])
@pytest.mark.parametrize("mask", [1, 8, 15])
def test_real_consumers_fallback_tails_masks_and_work(
    fallback_artifact: typing.Any, unavailable: bool, route: str, mask: int
) -> None:
    from generativeqc_compiler.dft.ao import NativeAO
    from generativeqc_compiler.dft.cuda import CudaGrid
    from generativeqc_compiler.dft.fixtures import basis_arguments, load_fixture
    from test_density_cuda import factors

    library = ct.CDLL(str(fallback_artifact.library))
    library.grid_cuda_lowering_unavailable_for_test.argtypes = [ct.c_bool]
    library.grid_cuda_lowering_unavailable_for_test(unavailable)
    meta, data = load_fixture("f_spherical")
    ingredients = tuple(
        name
        for i, name in enumerate(("rho", "gradient", "sigma", "tau"))
        if mask & (1 << i)
    )
    jets = 1 if mask == 1 else 3 if mask == 8 else 4
    try:
        with NativeAO(**basis_arguments(meta)) as basis:
            source = factors(basis, (11, 3))
            with CudaGrid(
                basis,
                fallback_artifact,
                order=1,
                tile_points=7,
                active_ao_capacity=basis.nao,
                orbital_capacity=(11, 3),
                orbital_tile=4,
                ingredients=ingredients,
            ) as grid:
                grid.set_source(source, stamp=source.stamp, route=route)
                calls = summands = 0
                identity = grid.metrics()["lowering"]["candidate_identity"]
                for points, columns in ((7, [0, 2, 5]), (3, [1]), (1, []), (0, [0])):
                    xyz = data["points"][:points]
                    ids = np.array(columns, dtype=np.uintp)
                    ao = basis.evaluate(xyz, 1)[:, :, ids]
                    expected = source.features(
                        ao,
                        stamp=source.stamp,
                        route="density_matrix",
                        ao_ids=ids,
                        ingredients=ingredients,
                    )
                    actual = grid.evaluate(xyz, ao_ids=ids, stamp=source.stamp)
                    for name in expected:
                        np.testing.assert_allclose(
                            actual[name], expected[name], atol=1e-11, rtol=1e-10
                        )
                    if points and len(ids):
                        calls += 2 if route == "density_matrix" else 4
                        summands += (
                            jets
                            * points
                            * len(ids)
                            * (2 * len(ids) if route == "density_matrix" else 14)
                        )
                    metrics = grid.metrics()
                    lowering = metrics["lowering"]
                    assert lowering["provider"] == (
                        "generated.cuda" if unavailable else "cublas"
                    )
                    assert lowering["candidate_identity"] == identity
                    assert lowering["preparations"] == 1
                    assert lowering["calls"] == calls
                    assert lowering["summands"] == summands
                    assert (
                        lowering["binding_bytes"]
                        <= grid.plan.provider_bytes + lowering["host_bytes"]
                    )
                    assert (
                        metrics["provider_retained_bytes"] <= grid.plan.provider_bytes
                    )
                    assert lowering["prepare_seconds"] >= 0
    finally:
        library.grid_cuda_lowering_unavailable_for_test(False)
