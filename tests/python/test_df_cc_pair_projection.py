"""GPU projection metadata against exact Fractions of actual stored FP64 values."""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
from fractions import Fraction
from pathlib import Path

import pytest
from _cc_owner_test_support import write_df_cpu_headers

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def projection_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if os.environ.get("GENERATIVEQC_DF_CC_CUDA_TEST") != "1":
        pytest.skip("requires explicit finite Slurm GPU qualification")
    assert os.environ.get("SLURM_JOB_ID"), "GPU probes must run through Slurm"
    compiler, cache = shutil.which("nvcc"), shutil.which("ccache")
    assert compiler and cache
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    directory = tmp_path_factory.mktemp("df-pair-projection")
    write_df_cpu_headers(directory)
    obj, executable = directory / "probe.o", directory / "probe"
    subprocess.run(
        [
            cache,
            compiler,
            "-std=c++20",
            "-O2",
            "-arch=sm_120",
            "-I" + str(ROOT / "src"),
            "-I" + str(ROOT / "include"),
            "-I" + str(directory),
            "-c",
            str(ROOT / "tests/native/df_cc_pair_projection_probe.cu"),
            "-o",
            str(obj),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "CCACHE_BASEDIR": str(ROOT)},
    )
    subprocess.run(
        [compiler, str(obj), "-lcublas", "-o", str(executable)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return executable


def _real(bits: int) -> Fraction:
    return Fraction.from_float(struct.unpack("d", struct.pack("Q", bits))[0])


def _run(
    probe: Path, occupied: int, virtuals: int, singles: list[int], tau: list[int]
) -> tuple[list[int], list[int]]:
    text = " ".join(map(str, [occupied, virtuals, *singles, *tau])) + "\n"
    if os.environ.get("GENERATIVEQC_DF_CC_SAVE_INPUTS") == "1":
        (probe.parent / f"case-{occupied}-{virtuals}.txt").write_text(text)
    result = subprocess.run(
        [str(probe)], input=text, capture_output=True, text=True, check=True, timeout=30
    )
    first, second = result.stdout.splitlines()
    return list(map(int, first.split())), list(map(int, second.split()))


@pytest.mark.parametrize("occupied,virtuals", [(1, 3), (2, 3), (3, 2)])
def test_outward_projection_covers_actual_error_and_diagonal_reflection(
    projection_probe: Path, occupied: int, virtuals: int
) -> None:
    patterns = [
        1,
        2,
        3,
        19,
        0x0010000000000000,
        0x3FF0000000000000,
        0x3FF0000000000001,
        0x3FD123456789ABCD,
        0xBFD23456789ABCDE,
    ]
    singles = [patterns[index % len(patterns)] for index in range(occupied * virtuals)]
    tau = [
        patterns[index % len(patterns)] for index in range((occupied * virtuals) ** 2)
    ]
    metadata, paired = _run(projection_probe, occupied, virtuals, singles, tau)
    assert metadata[0] == 0
    assert metadata[2] == max(bits & 0x7FFFFFFFFFFFFFFF for bits in singles)
    assert metadata[3] == max(bits & 0x7FFFFFFFFFFFFFFF for bits in [*tau, *paired])
    error = _real(metadata[1])
    cursor = 0
    for first_occupied in range(occupied):
        for second_occupied in range(first_occupied, occupied):
            start = cursor
            for first_virtual in range(virtuals):
                for second_virtual in range(virtuals):
                    flat = (
                        (first_occupied * occupied + second_occupied) * virtuals
                        + first_virtual
                    ) * virtuals + second_virtual
                    mate = (
                        (second_occupied * occupied + first_occupied) * virtuals
                        + second_virtual
                    ) * virtuals + first_virtual
                    actual = _real(paired[cursor])
                    assert abs(_real(tau[flat]) - actual) <= error
                    assert abs(_real(tau[mate]) - actual) <= error
                    if first_occupied == second_occupied:
                        assert (
                            paired[cursor]
                            == paired[start + second_virtual * virtuals + first_virtual]
                        )
                    cursor += 1
    assert cursor == len(paired)


def test_equal_subnormal_tau_is_bitwise_preserved(projection_probe: Path) -> None:
    metadata, paired = _run(projection_probe, 2, 3, [1] * 6, [1] * 36)
    assert metadata == [0, 0, 1, 1]
    assert paired == [1] * 27


@pytest.mark.parametrize("nonfinite", [0x7FF0000000000000, 0x7FF8000000000001])
def test_nonfinite_metadata_refuses_without_modifying_original(
    projection_probe: Path, nonfinite: int
) -> None:
    metadata, _ = _run(projection_probe, 1, 2, [nonfinite, 0], [0] * 4)
    assert metadata[0] == 1
    metadata, _ = _run(projection_probe, 1, 2, [0, 0], [nonfinite] * 4)
    assert metadata[0] == 1
