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
        RESOURCE_PATH: '''"""Unicode Ω fixture: byte offsets must remain correct."""
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
        """Unicode β before the edited expression."""
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
    from generativeqc_compiler.dft._generated_native_semilocal import SEMILOCAL_FAMILIES
    from generativeqc_compiler.dft._generated_split_hybrid_consumers import (
        SPLIT_HYBRID_POINT_CONSUMERS,
    )
    from generativeqc_compiler.dft.ao_cuda import (
        emit_native_xc_contraction_kernels,
        emit_native_xc_point_feature_terms,
    )

    root = Path(__file__).resolve().parents[2]
    source = (root / POINT_PATH).read_text()
    changed = render_trial(
        {POINT_PATH: source}, CudaScheduleTrial(pbe_point_threads=64)
    )[POINT_PATH]

    def kernel_template(text: str) -> str:
        values = [
            ast.literal_eval(node.value)
            for node in ast.parse(text).body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "_NATIVE_XC_CONTRACTION_KERNELS"
                for target in node.targets
            )
        ]
        assert len(values) == 1
        return values[0]

    original = kernel_template(source)
    candidate = kernel_template(changed)
    begin = candidate.index(
        "constexpr I feature_terms = point_feature_terms(functional);"
    )
    end = candidate.index("\n  if constexpr (!response)", begin)
    declaration = candidate[begin:end]
    assert candidate.replace(declaration, POINT_POLICY_ANCHOR) == original
    start = original.index("constexpr I point_threads(")
    stop = original.index("\n}", start) + 2
    helper = original[start:stop]
    split_start = candidate.index("void launch_split_hybrid_points(")
    split_width = "constexpr I threads = point_threads(feature_terms);"
    assert candidate.index(split_width, split_start) > split_start
    generated = emit_native_xc_contraction_kernels()
    batch_start = generated.index(
        "CudaXcPointBatchLauncher resolve_point_batch_launcher("
    )
    batch_stop = generated.index("\n}", batch_start) + 2
    batch = generated[batch_start:batch_stop]
    checks = []
    batch_checks = []
    for record in SEMILOCAL_FAMILIES:
        if not record["cuda_ks"]:
            continue
        width = 5 if record["requires_tau"] else 4 if record["requires_gradient"] else 1
        expected = 64 if record["symbol"] == "Pbe" else 32 if width == 4 else 128
        checks.append(
            f"static_assert(observe<{record['code']}, false>() == {expected});"
        )
        checks.append(f"static_assert(observe<{record['code']}, true>() == 128);")
        batch_checks.append(
            f"assert(resolve_point_batch_launcher({record['code']})() == {32 if width == 4 else 128});"
        )
    split_codes = " || ".join(
        f"functional == {code}" for code, _ in SPLIT_HYBRID_POINT_CONSUMERS
    )
    split_mgga = " || ".join(
        f"functional == {code}"
        for code, width in SPLIT_HYBRID_POINT_CONSUMERS
        if width == 5
    )
    batch_checks.extend(
        f"assert(resolve_point_batch_launcher({code})() == {32 if width == 4 else 128});"
        for code, width in SPLIT_HYBRID_POINT_CONSUMERS
    )
    probe = tmp_path / "point_trial.cpp"
    probe.write_text(
        '#include <cassert>\n#include <cstdint>\n#include <stdexcept>\n#include "dft/semilocal_family.hpp"\n'
        "using I = long;\nusing generativeqc::dft::SemilocalFamily;\n"
        "using generativeqc::dft::semilocal_family_code;\n"
        + emit_native_xc_point_feature_terms()
        + "\n"
        + helper
        + "\ntemplate<I functional, bool response> constexpr I observe() {\n"
        + declaration
        + "\nreturn threads;\n}\n"
        + "template<I feature_terms> constexpr I split_observe() {\n"
        + split_width
        + "\nreturn threads;\n}\n"
        + "static_assert(split_observe<4>() == 32);\n"
        + "static_assert(split_observe<5>() == 128);\n"
        + "\n".join(checks)
        + "\nusing CudaXcPointBatchLauncher = I(*)();\n"
        + "template<I width, I threads> I launch_point_batches() { return threads; }\n"
        + "namespace generated {\n"
        + f"bool split_hybrid_registered(unsigned functional) {{ return {split_codes}; }}\n"
        + f"bool split_hybrid_is_mgga(unsigned functional) {{ return {split_mgga}; }}\n"
        + "}\n"
        + batch
        + "\nint main() {\n"
        + "\n".join(batch_checks)
        + "\n}\n"
    )
    binary = native_cxx.build_executable(
        [probe],
        tmp_path / "point_trial",
        compile_args=("-std=c++17", f"-I{root / 'src'}"),
    )
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
        target.write_text(text)
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
