"""Independent contraction, resource and transactional CPU factor-jet contracts."""

import json
import math
import shutil
import subprocess
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pytest
import tomllib
from generativeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from generativeqc_compiler.integral.factorized_host import FactorizedERIHost
from test_factorized_second import DIRECTION, EXPONENTS, POSITIONS, reference

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def host_directory(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Retain one artifact cache for independent synchronous host owners."""
    return tmp_path_factory.mktemp("factorized-host")


@pytest.fixture(scope="module")
def host(host_directory: Path) -> FactorizedERIHost:
    """Compile one shared CPU program, with a deliberately small primitive page."""
    pytest.importorskip("pyscf")
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler is unavailable")
    owner = FactorizedERIHost(
        CppCompilerAdapter(Path(compiler), compile_timeout=600),
        host_directory,
        record_capacity=3,
    )
    yield owner
    owner.close()


@pytest.mark.parametrize("angular", ((0, 0, 0, 0), (2, 1, 0, 1), (3, 3, 3, 3)))
def test_streamed_first_and_weighted_hvp_match_independent_libcint(
    host: FactorizedERIHost, angular: tuple[int, ...]
) -> None:
    """Vary exponents and signed coefficients across real page boundaries."""
    primitives = tuple(
        ((float(exponent), 0.7), (float(exponent * 1.3), -0.2))
        for exponent in EXPONENTS
    )
    size = math.prod((order + 1) * (order + 2) // 2 for order in angular)
    components = tuple(np.linspace(0, size - 1, min(size, 64), dtype=int).tolist())
    weights = np.linspace(-0.4, 0.6, len(components))
    expected_first = np.zeros((len(components), 13))
    expected_hvp = np.zeros((4, 3))
    for combination in product(*primitives):
        exponents, coefficients = zip(*combination, strict=True)
        scale = math.prod(coefficients)
        value, gradient = reference(0, angular, POSITIONS, exponents=exponents)
        expected_first += scale * np.column_stack(
            (value[list(components)], gradient[:, list(components)].T)
        )
        differences = []
        for step in (2e-4, 1e-4):
            displaced = [
                reference(
                    0,
                    angular,
                    POSITIONS + sign * step * DIRECTION,
                    exponents=exponents,
                )[1]
                for sign in (-1, 1)
            ]
            differences.append((displaced[1] - displaced[0]) / (2 * step))
        independent_jvp = (4 * differences[1] - differences[0]) / 3
        expected_hvp += scale * (
            independent_jvp[:, list(components)] @ weights
        ).reshape(4, 3)
    actual_first = host.raw_first(angular, components, primitives, POSITIONS)
    actual_hvp, diagnostics = host.weighted_hvp(
        angular, components, primitives, POSITIONS, weights, DIRECTION
    )
    actual_direction, first_diagnostics = host.first_direction(
        angular, components, primitives, POSITIONS, DIRECTION
    )
    expected_direction = np.column_stack(
        (expected_first[:, 0], expected_first[:, 1:] @ DIRECTION.ravel())
    )
    np.testing.assert_allclose(actual_first, expected_first, atol=3e-10, rtol=2e-9)
    np.testing.assert_allclose(actual_hvp, expected_hvp, atol=1e-9, rtol=2e-9)
    np.testing.assert_allclose(
        actual_direction, expected_direction, atol=3e-10, rtol=2e-9
    )
    assert first_diagnostics["records"] == 16
    assert first_diagnostics["component_evaluations"] == 16 * len(components)
    assert first_diagnostics["native_artifact"] != diagnostics["native_artifact"]
    zero, zero_diagnostics = host.first_direction(
        angular, components, primitives, POSITIONS, np.zeros((4, 3))
    )
    np.testing.assert_allclose(zero[:, 0], expected_first[:, 0], atol=3e-10, rtol=2e-9)
    np.testing.assert_array_equal(zero[:, 1], 0)
    assert zero_diagnostics["program_identity"] == first_diagnostics["program_identity"]
    np.testing.assert_allclose(
        actual_first[:, 1:].reshape(-1, 4, 3).sum(axis=1), 0, atol=3e-12
    )
    np.testing.assert_allclose(actual_hvp.sum(axis=0), 0, atol=3e-12)
    assert diagnostics["records"] == 16
    assert diagnostics["component_evaluations"] == 16 * len(components)
    assert diagnostics["resources"]["peak_bytes"]["host"] == host.numeric_bytes


def test_factorized_host_native_failure_does_not_publish_partial_results(
    host: FactorizedERIHost,
) -> None:
    """A late invalid record or finite-input overflow must be transactional."""
    angular = np.zeros(4, dtype=np.uint32)
    components = np.zeros(1, dtype=np.uint32)
    records = np.tile(np.r_[EXPONENTS, POSITIONS.ravel(), 1.0], (2, 1))
    weights = np.ones(1)
    direction = np.ascontiguousarray(DIRECTION)
    for operation, mode, failure in (
        (1, "first", "late-invalid"),
        (2, "full", "late-overflow"),
        (3, "directional", "late-invalid"),
        (3, "directional", "late-overflow"),
    ):
        inputs = records.copy()
        if failure == "late-invalid":
            inputs[1, 0] = np.nan
        else:
            inputs[1, -1] = 1e308
            weights[0] = 1e308
            inputs[1, :4] = 0.001
        output = np.full({1: 13, 2: 12, 3: 2}[operation], 7.25)
        visits = np.full(2, 17, dtype=np.uint64)
        status = host._binding(mode)[2](
            operation,
            angular.ctypes.data,
            components.ctypes.data,
            1,
            inputs.ctypes.data,
            len(inputs),
            17,
            weights.ctypes.data,
            direction.ctypes.data,
            output.ctypes.data,
            output.size,
            visits.ctypes.data,
        )
        assert status == (1 if failure == "late-invalid" else 5)
        np.testing.assert_array_equal(output, 7.25)
        np.testing.assert_array_equal(visits, 17)


def test_factorized_host_budget_inputs_and_closed_owner(
    host: FactorizedERIHost, host_directory: Path, tmp_path: Path
) -> None:
    """Validate normal guards before allocating or using a closed owner."""
    compiler = CppCompilerAdapter(Path(shutil.which("c++")))
    with pytest.raises(MemoryError, match="budget"):
        FactorizedERIHost(compiler, tmp_path, budget_bytes=1)
    primitives = tuple(((float(exponent), 1.0),) for exponent in EXPONENTS)
    for components in ((0, 0), (-1,), (1,)):
        with pytest.raises(ValueError, match="components"):
            host.raw_first((0, 0, 0, 0), components, primitives, POSITIONS)
    with pytest.raises(ValueError, match="direction"):
        host.weighted_hvp((0, 0, 0, 0), (0,), primitives, POSITIONS, (1.0,), None)
    for direction in (None, np.zeros(12), np.full((4, 3), np.nan)):
        with pytest.raises(ValueError, match="direction"):
            host.first_direction((0, 0, 0, 0), (0,), primitives, POSITIONS, direction)
    closed = FactorizedERIHost(compiler, host_directory)
    closed.close()
    with pytest.raises(RuntimeError, match="closed"):
        closed.raw_first((0, 0, 0, 0), (0,), primitives, POSITIONS)


def test_first_direction_admission_and_failure_preserve_owner(
    host_directory: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No compile before input admission; failed native pages publish no work."""
    compiler = CppCompilerAdapter(Path(shutil.which("c++")))
    owner = FactorizedERIHost(compiler, host_directory)
    primitives = tuple(((float(exponent), 1.0),) for exponent in EXPONENTS)

    def unexpected_artifact(*args: object, **kwargs: object) -> None:
        pytest.fail("invalid input compiled a factorized artifact")

    with monkeypatch.context() as context:
        context.setattr(
            "generativeqc_compiler.integral.factorized_host._artifact",
            unexpected_artifact,
        )
        for components, positions, motion in (
            ((0, 0), POSITIONS, DIRECTION),
            ((0,), np.full((4, 3), np.inf), DIRECTION),
            ((0,), POSITIONS, np.full((4, 3), np.nan)),
        ):
            with pytest.raises(ValueError):
                owner.first_direction(
                    (0, 0, 0, 0), components, primitives, positions, motion
                )

    before = dict(owner.statistics)

    def late_failure(*args: object) -> int:
        return 5

    binding = owner._binding("directional")
    owner._bindings["directional"] = (*binding[:2], late_failure)
    with pytest.raises(FloatingPointError, match="native status 5"):
        owner.first_direction((0, 0, 0, 0), (0,), primitives, POSITIONS, DIRECTION)
    assert owner.statistics == before
    owner._bindings["directional"] = binding
    recovered, _ = owner.first_direction(
        (0, 0, 0, 0), (0,), primitives, POSITIONS, DIRECTION
    )
    assert np.isfinite(recovered).all()
    assert owner.statistics["records"] > 0
    owner.close()


@pytest.mark.parametrize("operation", (0, 1, 2, 3, 4))
def test_directional_native_abi_admits_only_matching_operation_and_capacity(
    host: FactorizedERIHost, operation: int
) -> None:
    """Explicit packing and finite direction gates remain transactional."""
    angular = np.zeros(4, dtype=np.uint32)
    components = np.zeros(1, dtype=np.uint32)
    record = np.r_[EXPONENTS, POSITIONS.ravel(), 1.0]
    motion = np.ascontiguousarray(DIRECTION)
    run = host._binding("directional")[2]
    for count, size, direction in (
        (129, 2, motion),
        (1, 13, motion),
        (1, 2, np.full((4, 3), np.nan)),
    ):
        output = np.full(13, 7.25)
        visits = np.full(2, 17, dtype=np.uint64)
        status = run(
            operation,
            angular.ctypes.data,
            components.ctypes.data,
            1,
            record.ctypes.data,
            count,
            17,
            None,
            direction.ctypes.data,
            output.ctypes.data,
            size,
            visits.ctypes.data,
        )
        assert status == 1
        np.testing.assert_array_equal(output, 7.25)
        np.testing.assert_array_equal(visits, 17)


def test_staged_wheel_factor_consumer_has_no_checkout_or_runtime_dependency(
    tmp_path: Path,
) -> None:
    """Exercise package destination headers in an isolated compiler-only import."""
    pytest.importorskip("pyscf")
    if shutil.which("c++") is None:
        pytest.skip("C++ compiler is unavailable")
    package_root = tmp_path / "site"
    shutil.copytree(
        ROOT / "python/generativeqc_compiler",
        package_root / "generativeqc_compiler",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "assets"),
    )
    mapping = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"][
        "scikit-build"
    ]["wheel"]["force-include"]
    for name in (
        "factorized_host_runtime.hpp",
        "eri_geometry.hpp",
        "range_moments.hpp",
    ):
        relative = "src/integrals/" + name
        destination = package_root / mapping[relative]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    value, gradient = reference(0, (0, 0, 0, 0), POSITIONS)
    inputs = {
        "exponents": EXPONENTS.tolist(),
        "centers": POSITIONS.tolist(),
        "expected": np.r_[value[0], gradient[:, 0]].tolist(),
    }
    code = r"""
import importlib.abc
import json
import shutil
import sys
from pathlib import Path
package_root = Path(sys.argv[1]).resolve()
# Supply the installed NumPy files without executing site-packages .pth hooks.
sys.path[:0] = [str(package_root), sys.argv[4]]
class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in ("generativeqc", "pyscf", "tools") or fullname.startswith(("generativeqc.", "pyscf.", "tools.")):
            raise AssertionError("compiler consumer imported " + fullname)
sys.meta_path.insert(0, BlockRuntime())
import numpy as np
from generativeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from generativeqc_compiler.common.paths import PACKAGE, asset_path, source_root
from generativeqc_compiler.integral.factorized_host import FactorizedERIHost
assert PACKAGE == package_root / "generativeqc_compiler"
for name in ("factorized_host_runtime.hpp", "eri_geometry.hpp", "range_moments.hpp"):
    relative = "src/integrals/" + name
    assert asset_path(relative) == PACKAGE / "assets" / relative
try:
    source_root()
except ValueError:
    pass
else:
    raise AssertionError("staged compiler unexpectedly found a checkout")
data = json.loads(sys.argv[3])
owner = FactorizedERIHost(CppCompilerAdapter(Path(shutil.which("c++"))), Path(sys.argv[2]))
try:
    actual = owner.raw_first((0,0,0,0), (0,), tuple(((exponent,1.0),) for exponent in data["exponents"]), data["centers"])
    np.testing.assert_allclose(actual[0], data["expected"], atol=3e-10, rtol=2e-9)
    motion = np.arange(12, dtype=float).reshape(4, 3) * 0.01
    directed, diagnostic = owner.first_direction((0,0,0,0), (0,), tuple(((exponent,1.0),) for exponent in data["exponents"]), data["centers"], motion)
    independent = np.asarray(data["expected"])
    np.testing.assert_allclose(directed[0], (independent[0], independent[1:] @ motion.ravel()), atol=3e-10, rtol=2e-9)
    assert diagnostic["records"] == diagnostic["component_evaluations"] == 1
    for name, module in tuple(sys.modules.items()):
        if name == "generativeqc_compiler" or name.startswith("generativeqc_compiler."):
            assert Path(module.__file__).resolve().is_relative_to(PACKAGE), name
finally:
    owner.close()
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            # Isolated mode alone still runs editable-install .pth import hooks.
            "-S",
            "-c",
            code,
            str(package_root),
            str(tmp_path / "cache"),
            json.dumps(inputs),
            str(Path(np.__file__).resolve().parents[1]),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=450,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
