"""Prepare isolated, source-identified CUDA schedule trials; never promote them.

Examples (from a clean checkout):
  python -m tools.cuda_schedule_trials --list-candidates
  python -m tools.cuda_schedule_trials --trial trial.json --output /tmp/qc-trial

The second command creates a detached worktree, not a change to the invoking
checkout. Build and qualify that exact worktree with the usual Slurm/ccache
protocol. Compilation, device correctness and endpoint timing remain not-run
until actually performed; this tool never manufactures a performance verdict.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

RESOURCE_PATH = "python/generativeqc_compiler/method/stationary_resources.py"
EMITTER_PATH = "python/generativeqc_compiler/method/stationary_cuda.py"
NATIVE_PATH = "src/dft/stationary_gradient_cuda.cuh"
BATCH_PATH = "python/generativeqc/batch.py"
MATRIX_PATH = "python/generativeqc_compiler/dft/xc_contraction_cuda.py"
POINT_PATH = "python/generativeqc_compiler/dft/ao_cuda.py"
POINT_ANCHOR = (
    "functional == static_cast<I>(SemilocalFamily::Pbe) && !response ? 32 : 128;"
)


@dataclass(frozen=True, slots=True)
class CudaScheduleTrial:
    """None means byte-for-byte incumbent behavior, not an assumed default.

    These are source experiments, not new public runtime/environment switches.
    Geometry constants are shared by resource planning and generated native
    admission. Ordinary grid trials forward their exact tile into both workload
    selection and execution; composite tile selection retains its own policy.
    """

    geometry_threads: int | None = None
    becke_threads: int | None = None
    becke_pair_rows: int | None = None
    geometry_scratch_mib: int | None = None
    grid_tile_points: int | None = None
    host_budget_mib: int | None = None
    device_budget_mib: int | None = None
    xc_matrix_tile: int | None = None
    pbe_point_threads: int | None = None

    def __post_init__(self) -> None:
        choices = {
            "geometry_threads": (32, 64, 128),
            "becke_threads": (32, 64, 128),
            "becke_pair_rows": (2, 4),
            "geometry_scratch_mib": (8, 16),
            "grid_tile_points": (256, 512, 1024),
            "host_budget_mib": (256, 512, 1024, 2048),
            "device_budget_mib": (512, 1024, 2048),
            "xc_matrix_tile": (8, 16, 32),
            "pbe_point_threads": (32, 64, 128),
        }
        for name, allowed in choices.items():
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value not in allowed):
                raise ValueError(f"{name} must be None or one of {allowed}")

    @classmethod
    def from_payload(cls, payload: Any) -> CudaScheduleTrial:
        if not isinstance(payload, dict):
            raise ValueError("trial must be a JSON object")
        unknown = set(payload) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unsupported trial fields: {sorted(unknown)}")
        return cls(**payload)


def _only(nodes: list[Any], description: str) -> Any:
    if len(nodes) != 1:
        raise ValueError(
            f"expected one {description}, found {len(nodes)}; source drift"
        )
    return nodes[0]


def _span(source: str, node: ast.expr) -> tuple[int, int]:
    # Python AST columns are UTF-8 byte offsets, not Unicode character offsets.
    lines = source.encode("utf-8").splitlines(keepends=True)
    if node.end_lineno is None or node.end_col_offset is None:
        raise ValueError("AST source span unavailable")
    return (
        sum(map(len, lines[: node.lineno - 1])) + node.col_offset,
        sum(map(len, lines[: node.end_lineno - 1])) + node.end_col_offset,
    )


def _rewrite(source: str, edits: list[tuple[ast.expr, str]]) -> str:
    spans = sorted((*_span(source, node), value) for node, value in edits)
    if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
        raise ValueError("overlapping schedule edits")
    result = source.encode("utf-8")
    for begin, end, value in reversed(spans):
        result = result[:begin] + value.encode("utf-8") + result[end:]
    text = result.decode("utf-8")
    ast.parse(text)
    return text


def _assignment(body: list[ast.stmt], name: str) -> ast.Assign:
    return _only(
        [
            node
            for node in body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        ],
        f"assignment to {name}",
    )


def _constant_edits(source: str, values: dict[str, int]) -> str:
    tree = ast.parse(source)
    edits = []
    for name, value in values.items():
        assignment = _assignment(tree.body, name)
        # Only integer constant expressions, never calls or executable policy.
        allowed = (ast.Constant, ast.BinOp, ast.LShift)
        if any(
            not isinstance(node, allowed)
            or isinstance(node, ast.Constant)
            and type(node.value) is not int
            for node in ast.walk(assignment.value)
        ):
            raise ValueError(f"{name} is no longer a constant expression")
        edits.append((assignment.value, str(value)))
    return _rewrite(source, edits)


def _verify_geometry_coupling(sources: dict[str, str], names: set[str]) -> None:
    tree = ast.parse(sources[EMITTER_PATH])
    function = _only(
        [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_runtime_layout_cuda"
        ],
        "stationary native-layout emitter",
    )
    coupled = {node.id for node in ast.walk(function) if isinstance(node, ast.Name)}
    imports = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and node.module.endswith("stationary_resources")
        for alias in node.names
    }
    if not names <= coupled & imports:
        raise ValueError("stationary planner/emitter schedule coupling changed")
    native = sources[NATIVE_PATH]
    for required in (
        "stationary_geometry_max_scratch_bytes",
        "stationary_geometry_max_threads",
        "stationary_becke_pair_tile_rows",
        "stationary_becke_threads",
    ):
        if required not in native:
            raise ValueError(f"native schedule contract missing {required}")


def _ordinary_grid(source: str, trial: CudaScheduleTrial) -> str:
    tree = ast.parse(source)
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_public_dft_cuda_force"
    ]
    function = _only(functions, "public CUDA force entry")
    branches = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "composite_force"
        and any(
            isinstance(stmt, ast.Assign)
            and any(
                isinstance(target, ast.Tuple)
                and [item.id for item in target.elts if isinstance(item, ast.Name)]
                == ["max_device_bytes", "max_host_bytes"]
                for target in stmt.targets
            )
            for stmt in node.orelse
        )
    ]
    branch = _only(branches, "ordinary/composite resource split")

    def tuple_assignment(names: list[str]) -> ast.Assign:
        return _only(
            [
                node
                for node in branch.orelse
                if isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Tuple)
                and [
                    item.id
                    for item in node.targets[0].elts
                    if isinstance(item, ast.Name)
                ]
                == names
            ],
            f"ordinary assignment {names}",
        )

    edits: list[tuple[ast.expr, str]] = []
    budget = tuple_assignment(["max_device_bytes", "max_host_bytes"])
    if not isinstance(budget.value, ast.Tuple) or len(budget.value.elts) != 2:
        raise ValueError("ordinary budget expression changed")
    for node, requested in zip(
        budget.value.elts, (trial.device_budget_mib, trial.host_budget_mib)
    ):
        if requested is not None:
            edits.append((node, str(requested << 20)))
    if trial.grid_tile_points is not None:
        policy = tuple_assignment(["tile_policy", "policy_tile_points"])
        edits.append((policy.value, f'("fixed", {trial.grid_tile_points})'))
        kwargs = _only(
            [
                node
                for node in ast.walk(function)
                if isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "kwargs"
            ],
            "ordinary force kwargs",
        )
        if not isinstance(kwargs.value, ast.Dict):
            raise ValueError("ordinary force kwargs is not a dictionary")
        if any(key is None for key in kwargs.value.keys):
            raise ValueError("cannot authenticate expanded force kwargs")
        keys = [
            key.value if isinstance(key, ast.Constant) else None
            for key in kwargs.value.keys
        ]
        if "tile_points" in keys:
            index = keys.index("tile_points")
            edits.append((kwargs.value.values[index], "policy_tile_points"))
        else:
            # Add the explicit execution binding without rewriting other values,
            # comments, budget ownership or the prepared-state lifetime.
            begin, end = _span(source, kwargs.value)
            fragment = source.encode("utf-8")[begin:end].decode("utf-8")
            edits.append(
                (kwargs.value, '{"tile_points": policy_tile_points, ' + fragment[1:])
            )
    return _rewrite(source, edits)


def required_paths(trial: CudaScheduleTrial) -> tuple[str, ...]:
    paths: list[str] = []
    if any(
        getattr(trial, key) is not None
        for key in (
            "geometry_threads",
            "becke_threads",
            "becke_pair_rows",
            "geometry_scratch_mib",
        )
    ):
        paths += [RESOURCE_PATH, EMITTER_PATH, NATIVE_PATH]
    if any(
        getattr(trial, key) is not None
        for key in (
            "grid_tile_points",
            "host_budget_mib",
            "device_budget_mib",
        )
    ):
        paths.append(BATCH_PATH)
    if trial.xc_matrix_tile is not None:
        paths.append(MATRIX_PATH)
    if trial.pbe_point_threads is not None:
        paths.append(POINT_PATH)
    return tuple(paths)


def render_trial(sources: dict[str, str], trial: CudaScheduleTrial) -> dict[str, str]:
    """Return only changed files; validate all edits before writing anything."""
    missing = set(required_paths(trial)) - set(sources)
    if missing:
        raise ValueError(f"missing source inputs: {sorted(missing)}")
    changed: dict[str, str] = {}
    constants = {
        name: value
        for name, value in (
            ("GEOMETRY_THREADS", trial.geometry_threads),
            ("BECKE_COOPERATIVE_THREADS", trial.becke_threads),
            ("BECKE_PAIR_TILE_ROWS", trial.becke_pair_rows),
            (
                "GEOMETRY_MAX_SCRATCH_BYTES",
                None
                if trial.geometry_scratch_mib is None
                else trial.geometry_scratch_mib << 20,
            ),
        )
        if value is not None
    }
    if constants:
        _verify_geometry_coupling(sources, set(constants))
        changed[RESOURCE_PATH] = _constant_edits(sources[RESOURCE_PATH], constants)
    if BATCH_PATH in required_paths(trial):
        changed[BATCH_PATH] = _ordinary_grid(sources[BATCH_PATH], trial)
    if trial.xc_matrix_tile is not None:
        source = sources[MATRIX_PATH]
        assignment = _assignment(ast.parse(source).body, "DEFAULT_XC_MATRIX_SCHEDULE")
        call = assignment.value
        if not (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "XcMatrixSchedule"
            and len(call.args) == 1
            and not call.keywords
        ):
            raise ValueError("XC matrix schedule constructor changed")
        changed[MATRIX_PATH] = _rewrite(
            source, [(call.args[0], str(trial.xc_matrix_tile))]
        )
    if trial.pbe_point_threads is not None:
        source = sources[POINT_PATH]
        strings = [
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and POINT_ANCHOR in node.value
        ]
        _only(strings, "physical-PBE point launch expression")
        if source.count(POINT_ANCHOR) != 1:
            raise ValueError("ambiguous physical-PBE point launch expression")
        candidate = source.replace(
            POINT_ANCHOR,
            POINT_ANCHOR.replace("? 32 : 128;", f"? {trial.pbe_point_threads} : 128;"),
        )
        ast.parse(candidate)
        changed[POINT_PATH] = candidate
    return {path: text for path, text in changed.items() if text != sources[path]}


def candidates() -> dict[str, dict[str, int | None]]:
    """Staged finite search, not a predicted performance ranking."""
    trials = {"incumbent": CudaScheduleTrial()}
    for threads, rows, scratch in product((32, 64, 128), (2, 4), (8, 16)):
        trials[f"becke-t{threads}-r{rows}-m{scratch}"] = CudaScheduleTrial(
            becke_threads=threads,
            becke_pair_rows=rows,
            geometry_scratch_mib=scratch,
        )
    for tile, matrix in product((256, 512, 1024), (8, 16, 32)):
        trials[f"grid{tile}-matrix{matrix}"] = CudaScheduleTrial(
            grid_tile_points=tile,
            xc_matrix_tile=matrix,
            host_budget_mib=1024,
            device_budget_mib=1024,
        )
    for threads in (32, 64, 128):
        trials[f"pbe-point-{threads}"] = CudaScheduleTrial(pbe_point_threads=threads)
    trials["joint-becke128-rows2-scratch16-grid1024"] = CudaScheduleTrial(
        becke_threads=128,
        becke_pair_rows=2,
        geometry_scratch_mib=16,
        grid_tile_points=1024,
        host_budget_mib=1024,
        device_budget_mib=1024,
    )
    return {name: asdict(trial) for name, trial in trials.items()}


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
    ).stdout.decode("utf-8")


def prepare_trial(
    root: Path, output: Path, trial: CudaScheduleTrial, ref: str = "HEAD"
) -> dict[str, Any]:
    """Create a detached source trial without builds, GPU commands or promotion."""
    root = root.resolve()
    if Path(_git(root, "rev-parse", "--show-toplevel").strip()).resolve() != root:
        raise ValueError("root must be the repository root")
    if _git(root, "status", "--porcelain", "--untracked-files=no").strip():
        raise ValueError("tracked checkout changes must be committed before a trial")
    if output.exists() or output.is_symlink():
        raise ValueError("trial output already exists")
    output = output.resolve()
    if (
        output == root
        or root in output.parents
        and output.relative_to(root).parts[0] != ".artifacts"
    ):
        raise ValueError("in-repository trials must live under ignored .artifacts")
    if "benchmarks" in output.parts and "results" in output.parts:
        raise ValueError("raw trial worktrees are not reviewed evidence publications")
    if root in output.parents:
        _git(root, "check-ignore", "--quiet", str(output))
    revision = _git(
        root, "rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}"
    ).strip()
    sources = {
        path: _git(root, "show", f"{revision}:{path}") for path in required_paths(trial)
    }
    changed = render_trial(sources, trial)
    output.parent.mkdir(parents=True, exist_ok=True)
    _git(root, "worktree", "add", "--detach", str(output), revision)
    for path, text in changed.items():
        destination = output / path
        if destination.is_symlink() or output not in destination.resolve().parents:
            raise ValueError(f"source file escapes the detached worktree: {path}")
        destination.write_text(text, encoding="utf-8")
    patch = _git(output, "diff", "--no-ext-diff", "--binary", "--")
    metadata = output / ".artifacts" / "cuda-schedule-trial"
    metadata.mkdir(parents=True, exist_ok=False)
    (metadata / "source.patch").write_text(patch, encoding="utf-8")

    def digest(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    result = {
        "schema": "generativeqc.cuda-schedule-trial.v1",
        "base_commit": revision,
        "trial": asdict(trial),
        "source_inputs": {path: digest(text) for path, text in sources.items()},
        "changed_files": {
            path: {"before": digest(sources[path]), "after": digest(text)}
            for path, text in changed.items()
        },
        "patch_sha256": digest(patch),
        "validation": {
            "source_preparation": "pass",
            "cuda_compile": "not-run",
            "numerical": "not-run",
            "complete_endpoints": "not-run",
        },
        "promotion": "not-evaluated",
        "notes": [
            "Direct-force CTA promotion remains in #2026; no generic warp-width edits.",
            "Static shared ceilings are unchanged; dynamic opt-in requires native support and a granted kernel limit.",
            "Grid/byte-budget changes may change selected AO maps and work; record actual routing and counts.",
            "Build with verified ccache; run device work only in a finite Slurm allocation.",
            "Retain this exact patch, source/binary identities, all negative samples and actual solver histories.",
        ],
    }
    (metadata / "manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--base-ref", default="HEAD")
    parser.add_argument("--trial", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--list-candidates", action="store_true")
    args = parser.parse_args()
    try:
        if args.list_candidates:
            if args.trial is not None or args.output is not None:
                parser.error("--list-candidates cannot prepare a trial")
            result = candidates()
        else:
            if args.trial is None or args.output is None:
                parser.error("--trial and --output are required")
            trial = CudaScheduleTrial.from_payload(
                json.loads(args.trial.read_text(encoding="utf-8"))
            )
            result = prepare_trial(args.root, args.output, trial, args.base_ref)
        print(json.dumps(result, indent=2, sort_keys=True))
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        parser.exit(2, f"CUDA schedule trial failed: {error}\n")


if __name__ == "__main__":
    main()
