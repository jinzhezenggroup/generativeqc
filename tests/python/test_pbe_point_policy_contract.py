"""Execute composed family/equal-spin policies through production point ABIs."""

import hashlib
import json
import re
import subprocess
from pathlib import Path

from generativeqc_compiler.dft.ao_cuda import (
    emit_native_xc_point_batch_dispatch,
    emit_native_xc_point_dispatch,
    emit_native_xc_point_program,
)

ROOT = Path(__file__).resolve().parents[2]


def test_compiled_family_equal_spin_point_contract(
    tmp_path: Path, native_cxx: object
) -> None:
    """Real canonical PBE math; host scheduling stubs supply no GPU evidence."""
    (tmp_path / "cuda_runtime_api.h").write_text(
        "#pragma once\nusing cudaStream_t = void*;\n", encoding="utf-8"
    )
    (tmp_path / "point_program.hpp").write_text(
        re.sub(
            r"([A-Za-z_]\w*<[^;]*?>)\s*<<<([^>]*?)>>>\s*\(",
            r"host_launch(\1, \2)(",
            emit_native_xc_point_program(),
        ),
        encoding="utf-8",
    )
    (tmp_path / "point_dispatch.hpp").write_text(
        '#include "point_program.hpp"\n'
        + emit_native_xc_point_dispatch()
        + emit_native_xc_point_batch_dispatch(),
        encoding="utf-8",
    )
    binary = tmp_path / "point-policy-contract"
    try:
        native_cxx.build_executable(
            [ROOT / "tests/native/fixtures/xc_point_policy/contract.cpp"],
            binary,
            compile_args=(
                "-std=c++17",
                "-O2",
                "-ffp-contract=off",
                "-fno-fast-math",
                f"-I{tmp_path}",
                f"-I{ROOT / 'src'}",
            ),
        )
    except subprocess.CalledProcessError as exc:
        evidence = ROOT / ".artifacts/point-policy-contract-native-failure"
        evidence.mkdir(parents=True, exist_ok=True)
        inputs = {}
        for source_file in (
            tmp_path / "point_program.hpp",
            tmp_path / "point_dispatch.hpp",
            tmp_path / "cuda_runtime_api.h",
            ROOT / "tests/native/fixtures/xc_point_policy/contract.cpp",
        ):
            data = source_file.read_bytes()
            (evidence / source_file.name).write_bytes(data)
            inputs[source_file.name] = hashlib.sha256(data).hexdigest()
        (evidence / "failure.json").write_text(
            json.dumps(
                {
                    "command": exc.cmd,
                    "returncode": exc.returncode,
                    "stdout": exc.stdout,
                    "stderr": exc.stderr,
                    "input_sha256": inputs,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        diagnostic = f"Native probe evidence: {evidence}\n{(exc.stderr or '')[:6000]}"
        if hasattr(exc, "add_note"):
            exc.add_note(diagnostic)
        else:
            print(diagnostic)
        raise
    completed = subprocess.run(
        [
            str(binary),
            str(ROOT / "tests/data/xc/scf_domain.tsv"),
            str(ROOT / "tests/data/xc/pbe0_restricted_point.tsv"),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    report = ROOT / ".artifacts/pbe-point-policy-contract-report.txt"
    report.parent.mkdir(exist_ok=True)
    report.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    assert completed.returncode == 0, completed.stdout + completed.stderr
