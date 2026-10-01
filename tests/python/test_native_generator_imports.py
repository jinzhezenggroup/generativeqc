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
packages = ('generativeqc_compiler.tensor', 'generativeqc_compiler.method')
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
