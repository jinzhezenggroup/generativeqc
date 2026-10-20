"""Independent host science and acceptance-accounting gates for the MD engine.

The compiled scalar composition checks generated R plus retained Hermite E
against libcint. It does not claim to execute CUDA scheduling, resident layout,
public spherical projection, or a native SCF endpoint; those are separate gates.
"""

from __future__ import annotations

import ctypes as ct
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from _md_j_engine_oracle import contracted_quartet, screened_j

ROOT = Path(__file__).resolve().parents[2]

BRIDGE = r"""
#define __host__
#define __device__
#define __constant__
#define __forceinline__ inline
#include "generated_md_j_engine.cuh"
#include "scf/cuda/hermite_recurrence.cuh"
using namespace generativeqc::scf::cuda_execution;

static unsigned index8(unsigned x, unsigned y, unsigned z) {
  unsigned result=0;
  for(unsigned i=0;i<x;++i) result+=(9-i)*(10-i)/2;
  for(unsigned j=0;j<y;++j) result+=9-x-j;
  return result+z;
}
extern "C" double primitive_eri(const double* alpha,const double* centers,
                                const unsigned* angular) {
  const double p=alpha[0]+alpha[1], q=alpha[2]+alpha[3];
  Vec3<double> P{}, Q{};
  double product[2][3]{};
  HermiteCoefficients<double> E[2][3];
  for(unsigned pair=0;pair<2;++pair) {
    const unsigned a=pair*2,b=a+1;
    for(unsigned axis=0;axis<3;++axis) {
      product[pair][axis]=(alpha[a]*centers[3*a+axis]+alpha[b]*centers[3*b+axis])/
                          (alpha[a]+alpha[b]);
      fill_hermite(angular[3*a+axis],angular[3*b+axis],product[pair][axis],
                   centers[3*a+axis],centers[3*b+axis],alpha[a],alpha[b],E[pair][axis]);
    }
  }
  P={product[0][0],product[0][1],product[0][2]};
  Q={product[1][0],product[1][1],product[1][2]};
  double radial[165];
  md_engine_radial<8>(p*q/(p+q),P,Q,radial);
  unsigned bra[3],ket[3];
  for(unsigned i=0;i<3;++i) {
    bra[i]=angular[i]+angular[3+i];
    ket[i]=angular[6+i]+angular[9+i];
  }
  double value=0;
  for(unsigned x=0;x<=bra[0];++x) for(unsigned y=0;y<=bra[1];++y)
  for(unsigned z=0;z<=bra[2];++z) {
    double a=E[0][0].at(angular[0],angular[3],x)*
             E[0][1].at(angular[1],angular[4],y)*
             E[0][2].at(angular[2],angular[5],z);
    for(unsigned u=0;u<=ket[0];++u) for(unsigned v=0;v<=ket[1];++v)
    for(unsigned w=0;w<=ket[2];++w) {
      double b=E[1][0].at(angular[6],angular[9],u)*
               E[1][1].at(angular[7],angular[10],v)*
               E[1][2].at(angular[8],angular[11],w);
      value+=a*b*((u+v+w)%2 ? -1.0 : 1.0)*radial[index8(x+u,y+v,z+w)];
    }
  }
  return value*2.0*std::pow(kPi,2.5)/(p*q*std::sqrt(p+q));
}
"""


@pytest.fixture(scope="module")
def primitive_eri(native_cxx: Any, tmp_path_factory: Any) -> Any:
    """Compile actual generated radial + actual retained transforms, no CUDA SDK."""
    from generativeqc_compiler.integral.md_j_engine_cuda import emit_md_j_engine_header

    directory = tmp_path_factory.mktemp("md-j-engine-libcint")
    (directory / "generated_md_j_engine.cuh").write_text(emit_md_j_engine_header())
    (directory / "cuda_runtime_api.h").write_text(
        "#pragma once\ntypedef int cudaError_t; typedef void* cudaStream_t;\n"
    )
    source, library = directory / "bridge.cpp", directory / "bridge.so"
    source.write_text(BRIDGE)
    native_cxx.build_shared(
        [source],
        library,
        compile_args=[
            "-std=c++20",
            "-O2",
            f"-I{directory}",
            f"-I{ROOT / 'src'}",
            f"-I{ROOT / 'include'}",
        ],
    )
    owner = ct.CDLL(str(library))
    function = owner.primitive_eri
    function.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.uint32, flags="C_CONTIGUOUS"),
    ]
    function.restype = ct.c_double
    return function


@pytest.mark.parametrize(
    "angular", [(0, 0, 0, 0), (1, 0, 1, 0), (2, 0, 1, 1), (2, 1, 2, 0), (2, 2, 2, 2)]
)
@pytest.mark.parametrize("geometry", ["separated", "coincident"])
def test_generated_radial_and_retained_transform_against_libcint(
    primitive_eri: Any, angular: tuple, geometry: str, record_property: Any
) -> None:
    """All Cartesian components, signed contractions, same-center and dddd roots."""
    pyscf = pytest.importorskip("pyscf")
    from pyscf import gto, lib

    lib.num_threads(1)
    centers = np.asarray(
        [(0.0, 0.0, 0.0), (0.7, -0.2, 0.4), (-0.3, 0.5, 1.1), (0.2, 0.9, -0.6)]
    )
    if geometry == "coincident":
        centers[:] = 0.0
    primitives = (
        ((1.7, 0.71), (0.43, -0.22)),
        ((0.9, -0.16), (0.31, 0.83)),
        ((1.2, 0.51), (0.22, 0.37)),
        ((2.1, 0.63), (0.55, -0.18)),
    )
    molecule = gto.M(
        atom=[(f"H{index}", center) for index, center in enumerate(centers)],
        basis={
            f"H{index}": [[order, *primitives[index]]]
            for index, order in enumerate(angular)
        },
        unit="Bohr",
        cart=True,
        verbose=0,
    )
    offsets = molecule.ao_loc_nr()
    scales = 1 / np.sqrt(np.diag(molecule.intor("int1e_ovlp")))
    expected = molecule.intor_by_shell("int2e", (0, 1, 2, 3)) * np.einsum(
        "i,j,k,l->ijkl", *(scales[offsets[i] : offsets[i + 1]] for i in range(4))
    )
    observed = contracted_quartet(primitive_eri, angular, centers, primitives)
    np.testing.assert_allclose(observed, expected, atol=3e-12, rtol=3e-11)
    record_property("oracle", f"libcint via PySCF {pyscf.__version__}")
    record_property("maximum_eri_error", float(np.max(np.abs(observed - expected))))


def test_public_mask_boundary_and_unrestricted_density_oracle() -> None:
    """The oracle itself retains actual FP64 product and both spin directions."""
    bounds = np.asarray([[1.0, 0.3], [0.3, 0.7]])
    eri = np.ones((2, 2, 2, 2))
    density = np.asarray([[[1.0, 2.0], [-0.3, 0.7]], [[-0.5, 0.1], [0.6, 0.2]]])
    cutoff = float(bounds[0, 1] * bounds[1, 1])
    accepted = screened_j(eri, bounds, density, cutoff)
    rejected = screened_j(eri, bounds, density, np.nextafter(cutoff, np.inf))
    assert accepted[0, 1] - rejected[0, 1] == pytest.approx(density[:, 1, 1].sum())
    np.testing.assert_allclose(
        accepted, screened_j(eri, bounds, density.sum(axis=0), cutoff)
    )


def test_acceptance_never_promotes_skipped_or_missing_cuda(tmp_path: Path) -> None:
    from benchmarks.md_j_engine_acceptance import junit_gate, summarize

    xml = tmp_path / "gate.xml"
    xml.write_text(
        '<testsuites><testsuite tests="2" failures="0" errors="0" skipped="1"><testcase name="a" time="0.1"/><testcase name="b" time="0"><skipped message="no GPU"/></testcase></testsuite></testsuites>'
    )
    partial = junit_gate(xml, returncode=0)
    assert partial["status"] == "skipped"
    assert partial["passed"] == 1 and partial["skipped"] == 1
    report = summarize(
        {
            "host": partial,
            "raw_j_cuda": {"status": "not-run"},
            "endpoints_cuda": {"status": "not-run"},
        }
    )
    assert not report["correctness_qualified"]
    assert not report["default_promotion_qualified"]
    assert json.loads(json.dumps(report))["gates"]["raw_j_cuda"]["status"] == "not-run"


def test_acceptance_requires_real_test_results(tmp_path: Path) -> None:
    from benchmarks.md_j_engine_acceptance import junit_gate

    missing = junit_gate(tmp_path / "missing.xml", returncode=0)
    assert missing["status"] == "failed"
    xml = tmp_path / "empty.xml"
    xml.write_text('<testsuites><testsuite tests="0"/></testsuites>')
    assert junit_gate(xml, returncode=0)["status"] == "not-run"
    xml.write_text(
        '<testsuites><testsuite tests="1"><testcase name="failure"><failure message="bad J"/></testcase></testsuite></testsuites>'
    )
    assert junit_gate(xml, returncode=1)["status"] == "failed"


def test_acceptance_all_scientific_gates_still_does_not_promote_default() -> None:
    from benchmarks.md_j_engine_acceptance import summarize

    report = summarize(
        {
            name: {"status": "passed"}
            for name in ("host", "raw_j_cuda", "endpoints_cuda", "cuda_graph_replay")
        }
    )
    assert report["correctness_qualified"]
    assert not report["default_promotion_qualified"]
    report["gates"]["source_provenance"] = {"status": "failed"}
    assert not summarize(report["gates"])["correctness_qualified"]


@pytest.mark.parametrize("status", ["not-run", "skipped", "failed"])
def test_acceptance_requires_executed_graph_capture(status: str) -> None:
    from benchmarks.md_j_engine_acceptance import REQUIRED_GATES, summarize

    gates = {name: {"status": "passed"} for name in REQUIRED_GATES}
    gates["cuda_graph_replay"] = {"status": status}
    assert not summarize(gates)["correctness_qualified"]


@pytest.mark.parametrize("graph_status", ["passed", "skipped", "failed"])
def test_gpu_acceptance_executes_and_enforces_graph_gate(
    graph_status: str, tmp_path: Path, monkeypatch: Any
) -> None:
    from benchmarks import md_j_engine_acceptance as acceptance

    output = tmp_path / "acceptance.json"
    executed: dict[str, list[str]] = {}

    def run_gate(
        name: str,
        selection: list[str],
        output: Path,
        environment: dict[str, str],
        timeout: int,
    ) -> dict[str, str]:
        executed[name] = selection
        assert environment["GENERATIVEQC_PROFILE"] == "off"
        return {"status": graph_status if name == "cuda_graph_replay" else "passed"}

    monkeypatch.setattr(acceptance, "ROOT", tmp_path)
    monkeypatch.setattr(acceptance, "_run_gate", run_gate)
    monkeypatch.setattr(acceptance, "_cuda_preflight", lambda: {"test_double": True})
    monkeypatch.setenv("GENERATIVEQC_PROFILE", "unrelated-cached-profile")
    monkeypatch.setattr(
        acceptance.sys, "argv", ["acceptance", "--gpu", "--output", str(output)]
    )
    if graph_status == "passed":
        acceptance.main()
    else:
        with pytest.raises(SystemExit) as error:
            acceptance.main()
        assert error.value.code == 1
    report = json.loads(output.read_text())
    assert report["gates"]["cuda_graph_replay"]["status"] == graph_status
    assert report["correctness_qualified"] == (graph_status == "passed")
    assert "graph_capture" in executed["cuda_graph_replay"]
    assert "not graph_capture" in executed["raw_j_cuda"][-1]


def test_acceptance_rejects_stale_xml_and_retains_distinct_gate_paths(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from types import SimpleNamespace

    from benchmarks import md_j_engine_acceptance as acceptance

    # A dotted output basename must not make each gate overwrite the same XML.
    output = tmp_path / "evidence.v1.json"
    stale = tmp_path / "evidence.v1-host.xml"
    stale.write_text(
        '<testsuites><testsuite><testcase name="old"/></testsuite></testsuites>'
    )
    monkeypatch.setattr(
        acceptance.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="", stderr="", returncode=0),
    )
    result = acceptance._run_gate("host", ["unused"], output, {}, 1)
    assert result["status"] == "failed"
    assert result["junit"] == str(stale)
    assert not stale.exists()
    assert (tmp_path / "evidence.v1-host.stdout.txt").is_file()


def test_optional_upstream_rejects_wrong_or_modified_commit(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from types import SimpleNamespace

    from benchmarks import md_j_engine_acceptance as acceptance

    responses = iter(["wrong-revision\n"])
    monkeypatch.setattr(
        acceptance.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=next(responses)),
    )
    with pytest.raises(ValueError, match="must be pinned"):
        acceptance.verify_upstream(tmp_path)
    responses = iter(
        [acceptance.UPSTREAM_COMMIT + "\n", " M gpu4pyscf/scf/j_engine.py\n"]
    )
    with pytest.raises(ValueError, match="tracked modifications"):
        acceptance.verify_upstream(tmp_path)


@pytest.fixture(scope="module")
def production_pipeline(native_cxx: Any, tmp_path_factory: Any) -> Any:
    from _md_j_engine_pipeline_probe import build_pipeline

    return build_pipeline(
        native_cxx, tmp_path_factory.mktemp("md-j-production-pipeline")
    )


@pytest.mark.parametrize("representation", ["cartesian", "spherical"])
@pytest.mark.parametrize("spin", ["restricted", "unrestricted"])
@pytest.mark.parametrize(
    "general", [False, True], ids=["contracted", "general-contraction-columns"]
)
@pytest.mark.parametrize(
    "density_kind", ["nonsymmetric", "diagonal", "antisymmetric", "zero"]
)
def test_production_transform_density_radial_project_against_libcint(
    production_pipeline: Any,
    representation: str,
    spin: str,
    general: bool,
    density_kind: str,
    record_property: Any,
) -> None:
    """Execute actual E, sorted pair_loc, D and reverse-E production bodies.

    Only the source traversal is scalar test orchestration. Device scheduling,
    allocation, residual screening and graph replay are not inferred from this
    host result; those have separately identified acceptance gates.
    """
    pytest.importorskip("pyscf")
    from _md_j_engine_oracle import packed_pipeline_fixture
    from pyscf import lib

    lib.num_threads(1)
    molecule, scales, arrays = packed_pipeline_fixture(representation, general)
    nao = molecule.nao_nr()
    generator = np.random.default_rng(91271)
    alpha = generator.normal(size=(nao, nao)) * 0.08
    beta = generator.normal(size=(nao, nao)) * 0.03
    if density_kind == "diagonal":
        alpha, beta = np.diag(alpha.diagonal()), np.diag(beta.diagonal())
    elif density_kind == "antisymmetric":
        alpha, beta = alpha - alpha.T, beta - beta.T
    elif density_kind == "zero":
        alpha[:], beta[:] = 0, 0
    alpha, beta = np.ascontiguousarray(alpha), np.ascontiguousarray(beta)
    total = alpha + beta if spin == "unrestricted" else alpha
    integrals = molecule.intor("int2e") * np.einsum(
        "i,j,k,l->ijkl", scales, scales, scales, scales
    )
    expected = np.einsum("ijkl,kl->ij", integrals, total, optimize=True)
    observed = np.full((nao, nao), np.nan)
    stages = np.full(3, np.nan)
    status = production_pipeline.pipeline_j(
        molecule.nbas,
        nao,
        *(value.ctypes.data for value in [*arrays, alpha, beta]),
        spin == "unrestricted",
        observed.ctypes.data,
        stages.ctypes.data,
    )
    assert status == 0, f"production pipeline returned status {status}"
    np.testing.assert_allclose(observed, expected, atol=3e-11, rtol=3e-11)
    assert np.isfinite(stages).all() and stages[0] > 0
    if density_kind == "zero":
        np.testing.assert_array_equal(stages[1:], 0.0)
    record_property("maximum_j_error", float(np.max(np.abs(observed - expected))))
    record_property(
        "stages",
        json.dumps(
            {
                "transform_maximum": stages[0],
                "hermite_density_maximum": stages[1],
                "potential_maximum": stages[2],
            }
        ),
    )
    record_property(
        "scope",
        "actual host-executed md_j_transforms/md_j_density/md_j_project/md_j_engine_pair_loc with compiled generated md_engine_contract; scalar canonical traversal",
    )
