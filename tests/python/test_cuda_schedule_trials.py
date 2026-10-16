"""Execute source-variant planning and detached-worktree ownership, without CUDA."""

import ast
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import pytest

from tools.cuda_schedule_trials import (
    BATCH_PATH,
    EMITTER_PATH,
    MATRIX_PATH,
    NATIVE_PATH,
    POINT_ANCHOR,
    POINT_PATH,
    POINT_POLICY_ANCHOR,
    RESOURCE_PATH,
    CudaScheduleTrial,
    candidates,
    prepare_trial,
    render_trial,
)


@pytest.fixture
def sources() -> dict[str, str]:
    return {
        RESOURCE_PATH: '''"""Unicode 惟 fixture: byte offsets must remain correct."""
GEOMETRY_THREADS = 32
BECKE_COOPERATIVE_THREADS = 32
BECKE_PAIR_TILE_ROWS = 4
GEOMETRY_MAX_SCRATCH_BYTES = 8 << 20
UNRELATED_WARP_WIDTH = 32
''',
        EMITTER_PATH: """from x.stationary_resources import (
    GEOMETRY_THREADS, BECKE_COOPERATIVE_THREADS, BECKE_PAIR_TILE_ROWS,
    GEOMETRY_MAX_SCRATCH_BYTES,
)
def _runtime_layout_cuda(plan):
    return (
        f"constexpr size_t stationary_geometry_max_threads = {GEOMETRY_THREADS};",
        f"constexpr size_t stationary_becke_threads = {BECKE_COOPERATIVE_THREADS};",
        f"constexpr size_t stationary_becke_pair_tile_rows = {BECKE_PAIR_TILE_ROWS};",
        f"constexpr size_t stationary_geometry_max_scratch_bytes = {GEOMETRY_MAX_SCRATCH_BYTES};",
    )
""",
        NATIVE_PATH: """// Contract fixture: actual CUDA execution is not simulated.
auto a = stationary_geometry_max_scratch_bytes;
auto b = stationary_geometry_max_threads;
auto c = stationary_becke_pair_tile_rows;
auto d = stationary_becke_threads;
""",
        BATCH_PATH: '''class PreparedBatch:
    def _public_dft_cuda_force(self, index, atoms):
        """Unicode 尾 before the edited expression."""
        composite_force = bool(index)
        if composite_force:
            max_device_bytes, max_host_bytes = 1 << 30, 2 << 30
            tile_policy, policy_tile_points = "budget-auto", None
        else:
            max_device_bytes, max_host_bytes = 512 << 20, 256 << 20
            tile_policy, policy_tile_points = "fixed", 256
        workload = (tile_policy, policy_tile_points, max_device_bytes, max_host_bytes)
        if composite_force:
            return workload, "composite"
        kwargs = {
            "max_device_bytes": max_device_bytes,
            "max_host_bytes": max_host_bytes,
            "resident_ao_cutoff": 1e-16,
        }
        return workload, kwargs

    def untouched(self):
        return 32, 128, 256
''',
        MATRIX_PATH: "DEFAULT_XC_MATRIX_SCHEDULE = XcMatrixSchedule(16)\nUNCHANGED = 128\n",
        POINT_PATH: 'SOURCE = r"""\nconstexpr I threads =\n    '
        + POINT_ANCHOR
        + '\n"""\n',
    }


def test_incumbent_is_byte_identical(sources: dict[str, str]) -> None:
    assert render_trial(sources, CudaScheduleTrial()) == {}


def test_geometry_constants_remain_coupled(sources: dict[str, str]) -> None:
    changed = render_trial(
        sources,
        CudaScheduleTrial(
            geometry_threads=64,
            becke_threads=128,
            becke_pair_rows=2,
            geometry_scratch_mib=16,
        ),
    )
    assert set(changed) == {RESOURCE_PATH}
    namespace = {}
    exec(changed[RESOURCE_PATH], namespace)  # noqa: S102 - controlled fixture code
    assert namespace["GEOMETRY_THREADS"] == 64
    assert namespace["BECKE_COOPERATIVE_THREADS"] == 128
    assert namespace["BECKE_PAIR_TILE_ROWS"] == 2
    assert namespace["GEOMETRY_MAX_SCRATCH_BYTES"] == 16 * 1024 * 1024
    assert namespace["UNRELATED_WARP_WIDTH"] == 32
    assert sources[RESOURCE_PATH].endswith("UNRELATED_WARP_WIDTH = 32\n")


@pytest.mark.parametrize(
    "path,old,new",
    [
        (EMITTER_PATH, "from x.stationary_resources", "from x.unrelated"),
        (EMITTER_PATH, "def _runtime_layout_cuda", "def disconnected"),
        (NATIVE_PATH, "stationary_becke_threads", "unrelated_threads"),
        (
            RESOURCE_PATH,
            "BECKE_COOPERATIVE_THREADS = 32",
            "BECKE_COOPERATIVE_THREADS = choose()",
        ),
        (
            RESOURCE_PATH,
            "BECKE_COOPERATIVE_THREADS = 32",
            'BECKE_COOPERATIVE_THREADS = "32"',
        ),
    ],
)
def test_source_drift_is_rejected(
    sources: dict[str, str], path: str, old: str, new: str
) -> None:
    sources[path] = sources[path].replace(old, new)
    with pytest.raises(ValueError):
        render_trial(sources, CudaScheduleTrial(becke_threads=128))


def test_grid_binds_policy_and_execution_preserving_composite(
    sources: dict[str, str],
) -> None:
    trial = CudaScheduleTrial(
        grid_tile_points=1024, host_budget_mib=1024, device_budget_mib=1024
    )
    changed = render_trial(sources, trial)
    assert set(changed) == {BATCH_PATH}
    original, modified = {}, {}
    exec(sources[BATCH_PATH], original)  # noqa: S102 - controlled fixture code
    exec(changed[BATCH_PATH], modified)  # noqa: S102 - controlled fixture code
    before = original["PreparedBatch"]()
    after = modified["PreparedBatch"]()
    assert before._public_dft_cuda_force(1, None) == after._public_dft_cuda_force(
        1, None
    )
    workload, kwargs = after._public_dft_cuda_force(0, None)
    assert workload == ("fixed", 1024, 1 << 30, 1 << 30)
    assert kwargs["tile_points"] == 1024
    assert kwargs["max_device_bytes"] == kwargs["max_host_bytes"] == 1 << 30
    assert kwargs["resident_ao_cutoff"] == 1e-16
    assert after.untouched() == before.untouched()


def test_existing_tile_binding_is_updated_without_duplicate(
    sources: dict[str, str],
) -> None:
    sources[BATCH_PATH] = sources[BATCH_PATH].replace(
        "kwargs = {", 'kwargs = {"tile_points": 256,'
    )
    text = render_trial(sources, CudaScheduleTrial(grid_tile_points=512))[BATCH_PATH]
    assert text.count('"tile_points"') == 1
    namespace = {}
    exec(text, namespace)  # noqa: S102 - controlled fixture code
    assert (
        namespace["PreparedBatch"]()._public_dft_cuda_force(0, None)[1]["tile_points"]
        == 512
    )


def test_budget_only_trial_does_not_change_tile_policy(sources: dict[str, str]) -> None:
    text = render_trial(sources, CudaScheduleTrial(device_budget_mib=1024))[BATCH_PATH]
    namespace = {}
    exec(text, namespace)  # noqa: S102 - controlled fixture code
    workload, kwargs = namespace["PreparedBatch"]()._public_dft_cuda_force(0, None)
    assert workload == ("fixed", 256, 1 << 30, 256 << 20)
    assert "tile_points" not in kwargs


def shared_request_source(source: str) -> str:
    """Executable fixture for the current shared explicit-request binding."""
    source = source.replace(
        "def _public_dft_cuda_force(self, index, atoms):",
        "def _public_dft_cuda_force(self, index, atoms, *, tile_points=None):",
    )
    source = source.replace(
        '            tile_policy, policy_tile_points = "budget-auto", None\n', ""
    )
    source = source.replace(
        '            tile_policy, policy_tile_points = "fixed", 256\n', ""
    )
    return source.replace(
        "        workload = ",
        '        tile_policy = "budget-auto" if tile_points is None else "fixed"\n'
        "        policy_tile_points = tile_points\n"
        "        workload = ",
    )


def test_legacy_policy_expression_is_not_silently_replaced(
    sources: dict[str, str],
) -> None:
    sources[BATCH_PATH] = sources[BATCH_PATH].replace(
        'tile_policy, policy_tile_points = "fixed", 256',
        "tile_policy, policy_tile_points = choose_policy()",
    )
    with pytest.raises(ValueError, match="unknown legacy"):
        render_trial(sources, CudaScheduleTrial(grid_tile_points=512))


@pytest.mark.parametrize("composite", (False, True))
@pytest.mark.parametrize("explicit_tile", (None, 256, 768))
def test_shared_request_trial_changes_only_ordinary_unspecified_tile(
    sources: dict[str, str], composite: bool, explicit_tile: int | None
) -> None:
    sources[BATCH_PATH] = shared_request_source(sources[BATCH_PATH])
    text = render_trial(sources, CudaScheduleTrial(grid_tile_points=512))[BATCH_PATH]
    namespace = {}
    exec(text, namespace)  # noqa: S102 - controlled executable policy fixture
    workload, execution = namespace["PreparedBatch"]()._public_dft_cuda_force(
        int(composite), None, tile_points=explicit_tile
    )
    expected = explicit_tile if explicit_tile is not None or composite else 512
    assert workload[:2] == ("budget-auto" if expected is None else "fixed", expected)
    if composite:
        assert execution == "composite" and workload[2:] == (1 << 30, 2 << 30)
    else:
        assert execution["tile_points"] == expected


@pytest.mark.parametrize("mutation", ("policy", "request", "duplicate"))
def test_shared_policy_trial_rejects_unknown_bindings(
    sources: dict[str, str], mutation: str
) -> None:
    text = shared_request_source(sources[BATCH_PATH])
    if mutation == "policy":
        text = text.replace(
            '"budget-auto" if tile_points is None else "fixed"', "choose(tile_points)"
        )
    elif mutation == "request":
        text = text.replace(
            "policy_tile_points = tile_points", "policy_tile_points = None"
        )
    else:
        text = text.replace(
            "        workload = ",
            "        policy_tile_points = None\n        workload = ",
        )
    sources[BATCH_PATH] = text
    with pytest.raises(ValueError):
        render_trial(sources, CudaScheduleTrial(grid_tile_points=512))


def test_foreign_grid_structure_rejects_before_partial_changes(
    sources: dict[str, str],
) -> None:
    original = dict(sources)
    sources[BATCH_PATH] = sources[BATCH_PATH].replace(
        "kwargs = {", "kwargs = {**other,"
    )
    with pytest.raises(ValueError):
        render_trial(
            sources, CudaScheduleTrial(grid_tile_points=1024, becke_threads=128)
        )
    assert sources[RESOURCE_PATH] == original[RESOURCE_PATH]


@pytest.mark.parametrize("width", [32, 64, 128])
def test_point_width_is_scoped_to_physical_pbe(
    sources: dict[str, str], width: int
) -> None:
    changed = render_trial(sources, CudaScheduleTrial(pbe_point_threads=width))
    text = changed.get(POINT_PATH, sources[POINT_PATH])
    assert f"!response ? {width} : 128;" in text
    assert ast.parse(text)


def test_ambiguous_point_expression_rejected(sources: dict[str, str]) -> None:
    sources[POINT_PATH] += "# " + POINT_ANCHOR + "\n"
    with pytest.raises(ValueError):
        render_trial(sources, CudaScheduleTrial(pbe_point_threads=64))


@pytest.mark.parametrize("width", [32, 64, 128])
def test_ingredient_point_trial_preserves_other_schedules(
    sources: dict[str, str], width: int
) -> None:
    source = sources[POINT_PATH].replace(
        "constexpr I threads =\n    " + POINT_ANCHOR, POINT_POLICY_ANCHOR
    )
    sources[POINT_PATH] = source
    changed = render_trial(sources, CudaScheduleTrial(pbe_point_threads=width))[
        POINT_PATH
    ]
    replacement = POINT_POLICY_ANCHOR.replace(
        "point_threads(feature_terms, response);",
        "functional == static_cast<I>(SemilocalFamily::Pbe) && !response "
        f"? {width} : point_threads(feature_terms, response);",
    )
    assert changed == source.replace(POINT_POLICY_ANCHOR, replacement)
    assert ast.parse(changed)


@pytest.mark.parametrize(
    "mutation",
    ["missing", "duplicate", "same-string", "mixed", "feature-policy", "response"],
)
def test_ingredient_point_trial_rejects_source_drift(
    sources: dict[str, str], mutation: str
) -> None:
    source = sources[POINT_PATH].replace(
        "constexpr I threads =\n    " + POINT_ANCHOR, POINT_POLICY_ANCHOR
    )
    if mutation == "missing":
        source = source.replace(POINT_POLICY_ANCHOR, "constexpr I threads = 32;")
    elif mutation == "duplicate":
        source += 'SECOND = r"""' + POINT_POLICY_ANCHOR + '"""\n'
    elif mutation == "same-string":
        source = source.replace(
            POINT_POLICY_ANCHOR, POINT_POLICY_ANCHOR + "\n" + POINT_POLICY_ANCHOR
        )
    elif mutation == "mixed":
        source += 'LEGACY = "' + POINT_ANCHOR + '"\n'
    elif mutation == "feature-policy":
        source = source.replace(
            "point_feature_terms(functional)", "unknown(functional)"
        )
    else:
        source = source.replace(
            "point_threads(feature_terms, response)",
            "point_threads(feature_terms, false)",
        )
    sources[POINT_PATH] = source
    with pytest.raises(ValueError, match="point launch expression"):
        render_trial(sources, CudaScheduleTrial(pbe_point_threads=64))


def test_actual_ingredient_point_trial_executes_only_pbe_override(
    tmp_path: Path, native_cxx: object
) -> None:
    import importlib.util
    import re

    from generativeqc_compiler.dft._generated_native_semilocal import SEMILOCAL_FAMILIES
    from generativeqc_compiler.dft._generated_split_hybrid_consumers import (
        SPLIT_HYBRID_POINT_CONSUMERS,
    )

    root = Path(__file__).resolve().parents[2]
    source = (root / POINT_PATH).read_text()
    changed = render_trial(
        {POINT_PATH: source}, CudaScheduleTrial(pbe_point_threads=64)
    )[POINT_PATH]

    # Execute only this trusted repository source variant. Relative compiler
    # imports retain their canonical package objects; no runtime or GPU is used.
    variant_path = tmp_path / "point_schedule_trial.py"
    variant_path.write_text(changed, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(
        "generativeqc_compiler.dft._point_schedule_trial", variant_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "cuda_runtime_api.h").write_text(
        "#pragma once\nusing cudaStream_t = void*;\n", encoding="utf-8"
    )
    (tmp_path / "point_program.hpp").write_text(
        re.sub(
            r"([A-Za-z_]\w*<[^;]*?>)\s*<<<([^>]*?)>>>\s*\(",
            r"host_launch(\1, \2)(",
            module.emit_native_xc_point_program(),
        ),
        encoding="utf-8",
    )
    (tmp_path / "point_dispatch.hpp").write_text(
        '#include "point_program.hpp"\n'
        + module.emit_native_xc_point_dispatch()
        + module.emit_native_xc_point_batch_dispatch(),
        encoding="utf-8",
    )
    checks = []
    for record in SEMILOCAL_FAMILIES:
        if not record["cuda_ks"]:
            continue
        width = 5 if record["requires_tau"] else 4 if record["requires_gradient"] else 1
        expected = 64 if record["symbol"] == "Pbe" else 32 if width == 4 else 128
        checks.append(
            f"check({record['code']}, {expected}, {32 if width == 4 else 128});"
        )
    for code, width in SPLIT_HYBRID_POINT_CONSUMERS:
        checks.append(f"probe_split_codes.push_back({code});")
        if width == 5:
            checks.append(f"probe_split_mgga.push_back({code});")
        checks.append(
            f"check({code}, {32 if width == 4 else 128}, {32 if width == 4 else 128});"
        )
    probe = tmp_path / "point_trial.cpp"
    probe.write_text(
        r"""
#define main point_arithmetic_contract_main
#include "contract.cpp"
#undef main
void check(unsigned functional, unsigned physical_threads, unsigned batch_threads) {
  double features[10]{}, weights[1]{1}, coefficients[10]{}, totals[3]{};
  int error = 0;
  for (const char* equal : {"0", "1"}) for (const char* family : {"0", "1"}) {
    policy("GENERATIVEQC_CUDA_XC_EQUAL_SPIN", equal);
    policy("GENERATIVEQC_CUDA_XC_PBE_POINT_SPECIALIZATION", family);
    for (std::size_t spins : {1U, 2U}) {
      auto launch = resolve_point_launcher(functional, false);
      launch(nullptr, features, weights, 1, spins, coefficients, totals, &error, functional, 1, 1, nullptr);
      require(observed_threads == physical_threads, "trial changed wrong physical geometry");
      auto batch = resolve_point_batch_launcher(functional, launch);
      batch(nullptr, features, weights, 1, spins, coefficients, totals, &error, functional, 1, 1, 1);
      require(observed_threads == batch_threads, "trial changed batch geometry");
      if (functional < 2) {
        auto response = resolve_point_launcher(functional, true);
        response(nullptr, features, weights, 1, spins, coefficients, totals, &error, functional, 1, 1, features);
        require(observed_threads == 128, "trial changed response geometry");
      }
    }
  }
}
int main() {
  execute_kernels = false; // Execute real dispatch; no numerical/GPU claim here.
"""
        + "\n".join(checks)
        + "\n}\n",
        encoding="utf-8",
    )
    try:
        binary = native_cxx.build_executable(
            [probe],
            tmp_path / "point_trial",
            compile_args=(
                "-std=c++17",
                f"-I{root / 'src'}",
                f"-I{tmp_path}",
                f"-I{root / 'tests/native/fixtures/xc_point_policy'}",
            ),
        )
    except subprocess.CalledProcessError as exc:
        # A captured compile error otherwise hides its diagnostic from pytest.
        # Keep the exact inputs and complete streams in the existing failure artifact.
        evidence = root / ".artifacts/point-schedule-trial-native-failure"
        evidence.mkdir(parents=True, exist_ok=True)
        inputs = {}
        for source_file in (
            probe,
            variant_path,
            tmp_path / "point_program.hpp",
            tmp_path / "point_dispatch.hpp",
            tmp_path / "cuda_runtime_api.h",
            root / "tests/native/fixtures/xc_point_policy/contract.cpp",
        ):
            data = source_file.read_bytes()
            (evidence / source_file.name).write_bytes(data)
            inputs[source_file.name] = hashlib.sha256(data).hexdigest()
        record = {
            "command": exc.cmd,
            "returncode": exc.returncode,
            "stdout": exc.stdout,
            "stderr": exc.stderr,
            "input_sha256": inputs,
        }
        (evidence / "failure.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
        diagnostic = f"Native probe evidence: {evidence}\n{(exc.stderr or '')[:6000]}"
        if hasattr(exc, "add_note"):
            exc.add_note(diagnostic)
        else:
            print(diagnostic)
        raise
    subprocess.run([str(binary)], check=True, timeout=30)


@pytest.mark.parametrize("tile", [8, 16, 32])
def test_matrix_tile_only_edits_constructor(sources: dict[str, str], tile: int) -> None:
    changed = render_trial(sources, CudaScheduleTrial(xc_matrix_tile=tile))
    text = changed.get(MATRIX_PATH, sources[MATRIX_PATH])
    assert (
        text
        == f"DEFAULT_XC_MATRIX_SCHEDULE = XcMatrixSchedule({tile})\nUNCHANGED = 128\n"
    )


@pytest.mark.parametrize("field", list(CudaScheduleTrial.__dataclass_fields__))
@pytest.mark.parametrize("bad", [True, -1, "128", 1.5])
def test_invalid_parameters_rejected(field: str, bad: object) -> None:
    with pytest.raises(ValueError):
        CudaScheduleTrial.from_payload({field: bad})


@pytest.mark.parametrize(
    "payload",
    [[], None, {"tuning_maximum_shared_bytes": 101376}, {"direct_warp_width": 128}],
)
def test_unsafe_or_unknown_axes_rejected(payload: object) -> None:
    with pytest.raises(ValueError):
        CudaScheduleTrial.from_payload(payload)


def test_every_finite_candidate_is_renderable(sources: dict[str, str]) -> None:
    table = candidates()
    assert 1 < len(table) < 64
    for payload in table.values():
        render_trial(sources, CudaScheduleTrial.from_payload(payload))
    assert table["incumbent"] == asdict(CudaScheduleTrial())


def test_crlf_and_unicode_are_preserved(sources: dict[str, str]) -> None:
    sources[RESOURCE_PATH] = sources[RESOURCE_PATH].replace("\n", "\r\n")
    text = render_trial(sources, CudaScheduleTrial(becke_threads=64))[RESOURCE_PATH]
    assert text == sources[RESOURCE_PATH].replace(
        "BECKE_COOPERATIVE_THREADS = 32", "BECKE_COOPERATIVE_THREADS = 64"
    )


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
    ).stdout.decode()


@pytest.fixture
def repository(tmp_path: Path, sources: dict[str, str]) -> Path:
    root = tmp_path / "repository"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.name", "CUDA trial test")
    git(root, "config", "user.email", "cuda-trial-test@example.invalid")
    (root / ".gitignore").write_text(".artifacts/\n")
    for path, text in sources.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(text.encode("utf-8"))
    git(root, "add", ".")
    git(root, "commit", "-qm", "Synthetic source-contract fixture")
    return root


def test_real_detached_worktree_preserves_invoking_checkout(
    repository: Path, tmp_path: Path
) -> None:
    original = git(repository, "rev-parse", "HEAD").strip()
    target = tmp_path / "detached"
    result = prepare_trial(
        repository, target, CudaScheduleTrial(becke_threads=128, xc_matrix_tile=8)
    )
    assert git(repository, "status", "--porcelain").strip() == ""
    assert result["base_commit"] == original
    assert result["validation"]["complete_endpoints"] == "not-run"
    assert result["promotion"] == "not-evaluated"
    assert set(result["changed_files"]) == {RESOURCE_PATH, MATRIX_PATH}
    patch = (target / ".artifacts/cuda-schedule-trial/source.patch").read_bytes()
    assert result["patch_sha256"] == hashlib.sha256(patch).hexdigest()
    for path, hashes in result["changed_files"].items():
        assert (
            hashes["after"] == hashlib.sha256((target / path).read_bytes()).hexdigest()
        )
    manifest = json.loads(
        (target / ".artifacts/cuda-schedule-trial/manifest.json").read_text()
    )
    assert manifest == result
    with pytest.raises(ValueError, match="already exists"):
        prepare_trial(repository, target, CudaScheduleTrial())


def test_dirty_checkout_rejected_before_creation(
    repository: Path, tmp_path: Path
) -> None:
    (repository / RESOURCE_PATH).write_text("changed\n")
    target = tmp_path / "not-created"
    with pytest.raises(ValueError, match="committed"):
        prepare_trial(repository, target, CudaScheduleTrial())
    assert not target.exists()


def test_non_artifact_and_publication_outputs_rejected(repository: Path) -> None:
    for path in ("new-trial", "benchmarks/results/new-trial"):
        with pytest.raises(ValueError):
            prepare_trial(repository, repository / path, CudaScheduleTrial())


def test_in_repository_ignored_worktree_is_supported(repository: Path) -> None:
    result = prepare_trial(
        repository, repository / ".artifacts" / "trial", CudaScheduleTrial()
    )
    assert result["changed_files"] == {}
    assert git(repository, "status", "--porcelain").strip() == ""


def test_actual_repository_source_contracts() -> None:
    from tools.cuda_schedule_trials import required_paths

    root = Path(__file__).resolve().parents[2]
    trial = CudaScheduleTrial(
        geometry_threads=64,
        becke_threads=128,
        becke_pair_rows=2,
        geometry_scratch_mib=16,
        grid_tile_points=1024,
        host_budget_mib=1024,
        device_budget_mib=1024,
        xc_matrix_tile=8,
        pbe_point_threads=64,
    )
    inputs = {
        path: (root / path).read_bytes().decode("utf-8")
        for path in required_paths(trial)
    }
    result = render_trial(inputs, trial)
    assert set(result) == {RESOURCE_PATH, BATCH_PATH, MATRIX_PATH, POINT_PATH}
    for path, text in result.items():
        ast.parse(text)
        assert inputs[path] == (root / path).read_bytes().decode("utf-8")
