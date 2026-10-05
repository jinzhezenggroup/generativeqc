"""Device-free execution/ABI gates for the actual indexed-grid work census."""

import ctypes as ct
import shutil
import subprocess
from pathlib import Path
from threading import RLock
from types import SimpleNamespace

import pytest
from generativeqc._stationary_cuda import _grid_metric_delta
from generativeqc_compiler.dft.cuda import CudaGrid, _AoGridWork

ROOT = Path(__file__).resolve().parents[2]


def test_native_work_counters_and_ctypes_layout(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler is unavailable")
    assertions = "\n".join(
        f"static_assert(offsetof(AoGridWork, {name}) == {_AoGridWork.__dict__[name].offset});"
        for name, _ in _AoGridWork._fields_
    )
    source = tmp_path / "work.cpp"
    source.write_text(
        '#include "dft/ao_grid_work.hpp"\n#include <cstddef>\n'
        "using generativeqc::dft::AoGridWork;\n"
        f"static_assert(sizeof(AoGridWork) == {ct.sizeof(_AoGridWork)});\n"
        + assertions
        + r"""
int main() {
  AoGridWork work;
  work.record_ao(7, 3, 11, 10);
  work.record_ao(5, 11, 11, 4);
  work.record_ao(7, 11, 11, 10, true);
  work.record_ao(0, 3, 11, 4);
  work.record_ao(7, 0, 11, 10);
  if (work.evaluation_passes != 2 || work.deriv1_passes != 1 || work.deriv2_passes != 1)
    return 1;
  if (work.active_point_ao != 76 || work.dense_point_ao != 209 || work.ao_jet_values != 430)
    return 2;
  if (work.deriv1_point_ao != 55 || work.deriv2_point_ao != 21 ||
      work.evaluation_tiles != 3 || work.evaluation_points != 19) return 10;
  if (work.discovery_passes != 1 || work.discovery_point_ao != 77 ||
      work.discovery_ao_jet_values != 770) return 3;
  work.record_projection(7, 3, 1, 4, true);
  work.record_projection(5, 11, 2, 1, false);
  work.record_projection(7, 0, 2, 4, false);
  if (work.projection_passes != 3 || work.projection_matrices != 6 ||
      work.projection_output_values != 194 || work.projection_fma_pairs != 1462 ||
      work.identical_spin_copy_bytes != 672) return 4;
  for (unsigned jets : {1U, 4U, 10U, 20U}) work.record_ao(1, 1, 1, jets);
  if (work.deriv0_passes != 1 || work.deriv1_passes != 2 ||
      work.deriv2_passes != 2 || work.deriv3_passes != 1) return 5;
  try { work.record_ao(1, 1, 1, 2); return 6; }
  catch (const std::invalid_argument&) {}
  try { work.record_ao(1, 2, 1, 4); return 7; }
  catch (const std::invalid_argument&) {}
  try { work.record_ao(UINT64_MAX, 2, 2, 4); return 8; }
  catch (const std::overflow_error&) {}
  work.evaluation_passes = UINT64_MAX;
  try { work.record_ao(1, 1, 1, 1); return 9; }
  catch (const std::overflow_error&) {}
}
"""
    )
    binary = tmp_path / "work"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            f"-I{ROOT / 'src'}",
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
    )
    subprocess.run([str(binary)], check=True)


def test_prepared_endpoint_subtracts_cumulative_ao_work() -> None:
    before = dict.fromkeys(
        ("device_ms", "input_ms", "output_ms", "packing_ms", "library_ms", "kernel_ms"),
        10.0,
    )
    after = dict.fromkeys(before, 13.0)
    before["ao_grid_work"] = {"deriv2_passes": 7, "ao_stage_ms": 3.0}
    after["ao_grid_work"] = {"deriv2_passes": 12, "ao_stage_ms": 4.5}
    result = _grid_metric_delta(after, before)
    assert result["ao_grid_work"] == {"deriv2_passes": 5, "ao_stage_ms": 1.5}
    assert after["ao_grid_work"]["deriv2_passes"] == 12
    legacy = {name: value for name, value in before.items() if name != "ao_grid_work"}
    assert (
        _grid_metric_delta(dict.fromkeys(legacy, 3.0), dict.fromkeys(legacy, 1.0))[
            "kernel_ms"
        ]
        == 2.0
    )


def test_profiling_uses_one_event_pair_and_is_opt_in() -> None:
    source = (ROOT / "src/dft/cuda_grid.cu").read_text()
    selected = source.split("static int grid_cuda_run_selected_impl", 1)[1].split(
        "int grid_cuda_run_selected_v1", 1
    )[0]
    assert "detailed_profile = detailed_profile || p.profile_stages;" in selected
    assert "ctx.section(p.profile_stages," not in selected
    for stage in ("ao_stage_ms", "density_gather_ms", "projection_ms", "feature_ms"):
        assert f"p.work_metrics.{stage} +=" in selected
    assert "bool profile_stages = false;" in source


def test_older_grid_library_does_not_invent_work_counters() -> None:
    """Retain the shared lowering ABI when optional work exports are absent."""
    calls = []

    def call(name: str, *args: object) -> None:
        calls.append(name)
        if name == "grid_cuda_lowering_v1":
            labels = ct.cast(args[1], ct.POINTER(ct.c_char_p))
            for position, label in enumerate(
                (b"retained", b"candidate", b"fp64", b"semantic")
            ):
                labels[position] = label

    owner = SimpleNamespace(
        _lock=RLock(),
        _check_open=lambda: None,
        _handle=ct.c_void_p(1),
        _library=SimpleNamespace(),
        _call=call,
    )
    result = CudaGrid.metrics(owner)
    assert "ao_grid_work" not in result
    assert result["lowering"]["provider"] == "retained"
    assert calls == ["grid_cuda_metrics_v1", "grid_cuda_lowering_v1"]
    owner._borrowed = False
    with pytest.raises(NotImplementedError, match="lacks stage profiling"):
        CudaGrid.profile_stages(owner)


def test_stage_profile_admission_and_boolean_contract() -> None:
    calls = []
    owner = SimpleNamespace(
        _lock=RLock(),
        _check_open=lambda: None,
        _handle=ct.c_void_p(1),
        _library=SimpleNamespace(grid_cuda_profile_stages_v1=object()),
        _borrowed=False,
        _call=lambda name, handle, enabled: calls.append((name, enabled)),
    )
    with pytest.raises(TypeError, match="Boolean"):
        CudaGrid.profile_stages(owner, 1)
    owner._borrowed = True
    with pytest.raises(RuntimeError, match="leased"):
        CudaGrid.profile_stages(owner)
    assert calls == []
    owner._borrowed = False
    CudaGrid.profile_stages(owner)
    CudaGrid.profile_stages(owner, False)
    assert calls == [
        ("grid_cuda_profile_stages_v1", 1),
        ("grid_cuda_profile_stages_v1", 0),
    ]
