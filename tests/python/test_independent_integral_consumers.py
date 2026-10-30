"""Independent value/derivative schedules retain one scientific request and ABI."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.integral.ir import KernelConsumer
from generativeqc_compiler.integral.production_bundle import (
    write_production_bundle,
    write_production_bundles,
)
from generativeqc_compiler.integral.production_profile import resolve_production_profile
from generativeqc_compiler.integral.production_selection import (
    split_independent_consumers,
)
from generativeqc_compiler.integral.tuning import policy as tuning_policy
from generativeqc_compiler.integral.tuning.manifest import update_manifest_payload

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _manifest(path: Path) -> Path:
    """Exercise root folding, scalar force support, and the resident force ABI."""
    rows = []
    for name, components in (("ppps", 27), ("dpss", 18)):
        rows.append(
            {
                "shell_class": name,
                "consumers": ["fock", "force"],
                "recurrence": "rys3",
                "fock_recurrence": "rys2",
                "capabilities": ["streaming_fock", "local_packed_streaming_fock"],
                "schedule": {
                    "kind": "thread_tasks",
                    "block_threads": 32,
                    "component_tile": components,
                    "tasks_per_warp": 32,
                    "shared_coulomb": False,
                },
                "fock_schedule": {
                    "kind": "packed_tasks",
                    "block_threads": 128,
                    "component_tile": components,
                    "tasks_per_warp": 32,
                    "shared_coulomb": False,
                },
            }
        )
    rows[0]["resident_force_recurrence"] = "rys3"
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "architectures": {
                    "portable_cuda": {
                        "kind": "portable",
                        "optional_fock_variants": False,
                        "kernels": rows,
                    }
                },
            }
        )
    )
    return path


@pytest.mark.parametrize(
    "architecture",
    ("sm_70", "sm_75", "sm_80", "sm_86", "sm_89", "sm_90", "sm_100", "sm_120"),
)
def test_independent_consumers_preserve_scientific_outputs(
    tmp_path: Path, architecture: str
) -> None:
    """Resource validation uses each actual target; portability is not tuning."""
    profile = resolve_production_profile(
        _manifest(tmp_path / "manifest.json"), architecture
    )
    assert profile.portable and not profile.tuned
    for selection in profile.selections:
        force, fock = split_independent_consumers(selection)
        assert force.consumers == (KernelConsumer.FORCE,)
        assert fock.consumers == (KernelConsumer.FOCK,)
        assert force.integral.operator == fock.integral.operator
        assert force.integral.spec == fock.integral.spec == selection.spec
        assert force.integral.derivative.order == 1
        assert fock.integral.derivative is None
        assert force.integral.required_rys_roots == 3
        assert fock.integral.required_rys_roots == 2
        assert fock.schedule == selection.fock_schedule
        assert force.schedule == selection.schedule
        assert not force.capabilities
        assert fock.capabilities == selection.capabilities
        assert force.resident_force_recurrence == selection.resident_force_recurrence
        assert fock.resident_force_recurrence is None


@pytest.mark.parametrize(
    "recurrence", (True, [], "rys1", "rys3", "subset_wick", "invalid")
)
def test_invalid_independent_value_recurrence_fails_at_selection(
    tmp_path: Path, recurrence: object
) -> None:
    """Both programmatic and JSON entry points reject wrong value root counts."""
    manifest = _manifest(tmp_path / "manifest.json")
    selection = resolve_production_profile(manifest, "sm_80").selections[0]
    with pytest.raises(ValueError):
        replace(selection, fock_recurrence=recurrence)
    payload = json.loads(manifest.read_text())
    payload["architectures"]["portable_cuda"]["kernels"][0]["fock_recurrence"] = (
        recurrence
    )
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        resolve_production_profile(manifest, "sm_80")


def test_independent_value_rejects_target_block_overflow(tmp_path: Path) -> None:
    """A legal force schedule cannot hide an illegal independent value mapping."""
    manifest = _manifest(tmp_path / "manifest.json")
    payload = json.loads(manifest.read_text())
    payload["architectures"]["portable_cuda"]["kernels"][0]["fock_schedule"][
        "block_threads"
    ] = 2048
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        resolve_production_profile(manifest, "sm_80")


@pytest.mark.parametrize("consumer", ("fock", "force"))
def test_schedule_only_tuner_cannot_publish_an_unmeasured_recurrence(
    tmp_path: Path, consumer: str
) -> None:
    """A rejected promotion leaves the entire input manifest intact."""
    manifest = _manifest(tmp_path / "manifest.json")
    payload = json.loads(manifest.read_text())
    original = json.dumps(payload, sort_keys=True)
    selection = resolve_production_profile(manifest, "sm_80").selections[0]
    with pytest.raises(ValueError, match="schedule-only tuning"):
        update_manifest_payload(
            payload,
            "portable_cuda",
            {selection.spec.name: selection.schedule},
            consumer,
        )
    assert json.dumps(payload, sort_keys=True) == original


@pytest.mark.parametrize("multiple", (False, True))
def test_independent_consumers_compile_and_link(tmp_path: Path, multiple: bool) -> None:
    """Resolve every legacy/profile, streaming, and resident C wrapper at link."""
    nvcc = os.environ.get("GENERATIVEQC_NVCC")
    if nvcc is None:
        pytest.skip("set GENERATIVEQC_NVCC for the compiled ABI gate")
    architecture = os.environ.get("GENERATIVEQC_CUDA_ARCH", "sm_80")
    manifest = _manifest(tmp_path / "manifest.json")
    output = tmp_path / "generated"
    if multiple:
        paths = write_production_bundles(manifest, output, 1, (architecture,))
    else:
        paths = write_production_bundle(manifest, output, 1, architecture)
    objects = []
    for source in paths:
        if source.suffix != ".cu":
            continue
        obj = source.with_suffix(".o")
        result = subprocess.run(
            [
                nvcc,
                "-std=c++20",
                f"-arch={architecture}",
                f"-I{REPOSITORY_ROOT / 'src'}",
                f"-I{output}",
                "-Xcompiler=-fPIC",
                "-c",
                str(source),
                "-o",
                str(obj),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=240,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        objects.append(obj)
    result = subprocess.run(
        [
            nvcc,
            "-shared",
            "-Xlinker=--no-undefined",
            *map(str, objects),
            "-o",
            str(tmp_path / "libindependent.so"),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("consumer", (KernelConsumer.FOCK, KernelConsumer.FORCE))
def test_tuning_carries_the_measured_independent_integral_to_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, consumer: KernelConsumer
) -> None:
    """Retuning geometry keeps the actual value/derivative mathematics and peer."""
    manifest = _manifest(tmp_path / "manifest.json")
    selection = resolve_production_profile(manifest, "sm_80").selections[0]
    force, fock = split_independent_consumers(selection)
    selected = fock if consumer == KernelConsumer.FOCK else force
    monkeypatch.setattr(tuning_policy, "_PRODUCTION_MANIFEST_PATH", manifest)
    trials = tuning_policy.supported_schedule_trials(
        selection.spec, consumer, cuda_target_info("sm_80")
    )
    assert trials and all(trial.integral == selected.integral for trial in trials)
    payload = json.loads(manifest.read_text())
    update_manifest_payload(
        payload,
        "portable_cuda",
        {selection.spec.name: selected.schedule},
        consumer,
        qualified_integrals={selection.spec.name: selected.integral},
    )
    manifest.write_text(json.dumps(payload))
    updated = resolve_production_profile(manifest, "sm_80").selections[0]
    assert split_independent_consumers(updated) == (force, fock)
    with pytest.raises(ValueError, match="qualified tuning integral differs"):
        update_manifest_payload(
            payload,
            "portable_cuda",
            {selection.spec.name: selected.schedule},
            consumer,
            qualified_integrals={
                selection.spec.name: (
                    force if consumer == KernelConsumer.FOCK else fock
                ).integral
            },
        )

def test_portable_uniform_warp_force_unroll_pragmas_target_loops() -> None:
    """CuMetal requires every emitted unroll pragma to precede a real loop."""
    from generativeqc_compiler.integral.fused_schedule import build_fused_shell_plan
    from generativeqc_compiler.integral.lowering.force_rys_uniform import (
        _emit_rys_uniform_warp_force_consumer_cuda,
    )

    profile = resolve_production_profile(
        REPOSITORY_ROOT
        / "python/generativeqc_compiler/integral/production_shell_classes.json",
        "sm_80",
        "portable",
    )
    selection = next(item for item in profile.selections if item.spec.name == "pppp")
    parts = split_independent_consumers(selection)
    assert parts is not None
    force, _ = parts
    plan = build_fused_shell_plan(
        force.spec,
        schedule=force.schedule,
        integral=force.integral,
        target=profile.target,
    )
    source = _emit_rys_uniform_warp_force_consumer_cuda(
        force.spec, plan, force.schedule.minimum_blocks_per_sm
    )
    assert "const double fourth_0" in source
    lines = source.splitlines()
    for index, line in enumerate(lines):
        if line.strip().startswith("#pragma unroll"):
            assert index + 1 < len(lines)
            assert lines[index + 1].lstrip().startswith(("for (", "for(")), (
                index,
                lines[index + 1],
            )
