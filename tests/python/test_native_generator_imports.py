"""Native AOT families reuse canonical, dependency-light compiler packages."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GENERATORS = (
    "gfn2_pair_native",
    "scf_array_native",
    "scf_density_cuda",
    "cosx_derivative_native",
    "cosx_contractions",
    "nonlocal_pair_native",
    "rccsd_native",
    "gfn2_aes2_native",
    "gfn2_electronic_native",
    "gfn2_electronic_cuda",
    "gfn2_es2_native",
    "gfn2_es3_native",
    "gfn2_h0_native",
    "gfn2_scc_free_energy_native",
    "gfn2_spin_native",
)


DFT_EXPORTS = {
    "NativeAO": "ao",
    "directional_ao_jets": "ao",
    "jet_indices": "ao",
    "DensitySource": "density_source",
    "DensityStamp": "density_source",
    "density_features": "features",
    "orbital_features": "features",
    "spin_densities": "features",
    "ExplicitGrid": "grid",
    "GridPolicy": "grid",
    "GridProfile": "grid",
    "GridSpec": "grid",
    "MolecularGrid": "grid",
    "grid_policy_provenance": "grid",
    "partition_weights": "grid",
    "FixedDensityNonlocalCorrelation": "nonlocal_integration",
    "NonlocalGeometry": "nonlocal_integration",
    "NonlocalIntegral": "nonlocal_integration",
    "assemble_nonlocal_potential_reference": "nonlocal_reference",
    "nonlocal_energy_density_reference": "nonlocal_reference",
    "nonlocal_energy_reference": "nonlocal_reference",
    "nonlocal_explicit_geometry_derivatives_reference": "nonlocal_reference",
    "nonlocal_feature_derivatives_reference": "nonlocal_reference",
    "nonlocal_kernel_matrix_reference": "nonlocal_reference",
    "DEVICE_FUSED": "xc_schedule",
    "HOST_UNFUSED": "xc_schedule",
    "GridXcCandidateAssessment": "xc_schedule",
    "GridXcCandidateLimits": "xc_schedule",
    "GridXcCandidateShape": "xc_schedule",
    "GridXcExecutionSchedule": "xc_schedule",
    "GridXcScheduleCandidate": "xc_schedule",
    "GridXcScientificIdentity": "xc_schedule",
    "assess_grid_xc_schedule": "xc_schedule",
    "grid_xc_schedule": "xc_schedule",
    "rank_grid_xc_candidates": "xc_schedule",
    "rank_grid_xc_schedules": "xc_schedule",
    "PreparedGrid": "prepared",
    "PreparedGridBatch": "prepared",
}


def _run(script: str, *, without_site: bool) -> None:
    result = subprocess.run(
        [sys.executable, *(["-S"] if without_site else []), "-c", script],
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join((str(ROOT / "python"), str(ROOT))),
        },
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("generator", GENERATORS)
@pytest.mark.parametrize("preload", (False, True))
def test_generator_keeps_canonical_packages_without_runtime(
    generator: str,
    preload: bool,
) -> None:
    _run(
        f"""
import importlib
import importlib.abc
import runpy
import sys

class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{'numpy', 'generativeqc', 'pyscf', 'torch', 'cupy'}}:
            raise AssertionError('source generation imported ' + fullname)

sys.meta_path.insert(0, BlockRuntime())
packages = ('generativeqc_compiler.tensor', 'generativeqc_compiler.method',
            'generativeqc_compiler.dft')
before = {{name: importlib.import_module(name) for name in packages}} if {preload!r} else {{}}
runpy.run_path('tools/generate_{generator}.py', run_name='import_generator')
import generativeqc_compiler
for name in packages:
    package = importlib.import_module(name)
    assert before.get(name, package) is package
    assert getattr(generativeqc_compiler, name.rsplit('.', 1)[1]) is package
    assert package.__spec__.name == name
    assert package.__file__.endswith('/__init__.py')
from generativeqc_compiler.tensor import Program, optimize
from generativeqc_compiler.tensor.program import Program as CanonicalProgram
from generativeqc_compiler.tensor.optimize import optimize as canonical_optimize
assert Program is CanonicalProgram
assert optimize is canonical_optimize
assert 'numpy' not in sys.modules
""",
        without_site=True,
    )


@pytest.mark.parametrize("preload", (False, True))
def test_generators_preserve_reference_exports_in_one_process(preload: bool) -> None:
    """AOT imports must not replace existing or later interpreter/packing APIs."""
    _run(
        f"""
import importlib
import runpy
tensor = importlib.import_module("generativeqc_compiler.tensor") if {preload!r} else None
before = (tensor.execute, tensor.PackedLayout) if tensor is not None else None
for generator in {GENERATORS!r}:
    runpy.run_path('tools/generate_' + generator + '.py', run_name='import_generator')
after = importlib.import_module('generativeqc_compiler.tensor')
assert tensor is None or after is tensor
from generativeqc_compiler.tensor import execute, PackedLayout, Program, constant
from generativeqc_compiler.tensor.interpreter import execute as canonical_execute
from generativeqc_compiler.tensor.packing import PackedLayout as CanonicalPackedLayout
assert execute is canonical_execute
assert PackedLayout is CanonicalPackedLayout
assert before is None or before == (execute, PackedLayout)
assert float(execute(Program({{'value': constant(2)}}), {{}}).outputs['value']) == 2.0
from generativeqc_compiler.cc.equations import amplitude_layouts
assert all(isinstance(layout, PackedLayout) for layout in amplitude_layouts(1, 2))
""",
        without_site=False,
    )


@pytest.mark.parametrize("generator", ("cosx_derivative_native", "cosx_contractions"))
@pytest.mark.parametrize("preload", (False, True))
def test_cosx_emission_without_runtime(generator: str, preload: bool) -> None:
    """Exercise actual emission, not only deferred generator-module imports."""
    _run(
        f"""
import importlib
import importlib.abc
import runpy
import sys
class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{'numpy', 'generativeqc', 'pyscf', 'torch', 'cupy'}}:
            raise AssertionError('source generation imported ' + fullname)
sys.meta_path.insert(0, BlockRuntime())
package = importlib.import_module('generativeqc_compiler.dft') if {preload!r} else None
namespace = runpy.run_path('tools/generate_{generator}.py', run_name='import_generator')
emit = namespace['native_header' if {generator!r} == 'cosx_derivative_native'
                 else 'emit_cosx_contractions']
source = emit()
assert source == emit() and 'ContractionSite' in source
assert package is None or package is importlib.import_module('generativeqc_compiler.dft')
assert 'numpy' not in sys.modules
""",
        without_site=True,
    )


@pytest.mark.parametrize("preload", (False, True))
def test_cosx_generators_preserve_all_dft_exports(preload: bool) -> None:
    """Retain the complete previous package API before and after AOT imports."""
    _run(
        f"""
import importlib
import runpy
owners = {DFT_EXPORTS!r}
package = importlib.import_module('generativeqc_compiler.dft')
assert set(package.__all__) == set(owners)
assert set(owners) <= set(dir(package))
before = {{name: getattr(package, name) for name in owners}} if {preload!r} else {{}}
for generator in ('cosx_derivative_native', 'cosx_contractions'):
    runpy.run_path('tools/generate_' + generator + '.py', run_name='import_generator')
assert package is importlib.import_module('generativeqc_compiler.dft')
for name, owner in owners.items():
    actual = getattr(package, name)
    assert actual is getattr(importlib.import_module('generativeqc_compiler.dft.' + owner), name)
    assert before.get(name, actual) is actual
try:
    package.no_such_dft_export
except AttributeError:
    pass
else:
    raise AssertionError('unknown package export was accepted')
""",
        without_site=False,
    )
