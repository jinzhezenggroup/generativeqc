"""Compare the selected provider against exact retained PTX, without any recapture."""

from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path

from diagnose_cumetal_ptx import KERNEL, file_sha256, run_command

FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures/cumetal/primitive_pair_cache.json"
)
INPUT_SHA256 = "4247e547d34e4764cd51ae38ebc82619da07646c53b59e2e0620e9370f755037"


def compare(output: Path, env: dict[str, str], fixture: Path = FIXTURE) -> int:
    output.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + 90
    reference = json.loads(fixture.read_text(encoding="utf-8"))
    ptx = output / "retained-input.ptx"
    ptx.write_bytes(base64.b64decode(reference["input_base64"], validate=True))
    actual_hash = file_sha256(ptx)
    if (
        actual_hash != INPUT_SHA256
        or actual_hash != reference["input_sha256"]
        or ptx.stat().st_size != reference["input_bytes"]
        or reference["entry"] != KERNEL
    ):
        raise ValueError("retained PTX fixture does not match the captured input")
    compiler = Path(env["CUMETAL_PREFIX"]) / "bin/cumetalc"
    report = {
        "purpose": "exact-input compiler comparison; not QC acceptance",
        "reference": {
            key: value for key, value in reference.items() if key != "input_base64"
        },
        "input_sha256": actual_hash,
        "selected_provider": env["CUMETAL_COMMIT"],
        "tested_commit": env.get("GITHUB_SHA"),
        "compiler_sha256": file_sha256(compiler),
        "developer_dir": env.get("DEVELOPER_DIR"),
        "backend": "cumetal-ir",
        "fp64_mode": "fast48",
        "suite_budget_seconds": 90,
        "commands": {},
    }

    def save() -> None:
        (output / "comparison.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )

    def run(name: str, command: list[str], timeout: int) -> dict[str, object]:
        report["commands"][name] = {"command": command, "status": "starting"}
        save()
        result = run_command(command, name, output, env, timeout, deadline)
        report["commands"][name] = result
        save()
        return result

    for name, command in (
        ("provider-version", [str(compiler), "--version"]),
        ("provider-head", ["git", "-C", env["CUMETAL_SOURCE"], "rev-parse", "HEAD"]),
        (
            "provider-status",
            [
                "git",
                "-C",
                env["CUMETAL_SOURCE"],
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
        ),
        ("xcode-version", ["xcodebuild", "-version"]),
        ("metal-version", ["xcrun", "metal", "--version"]),
    ):
        if run(name, command, 5)["returncode"] != 0:
            return 1
    if (output / "provider-head.stdout.txt").read_text().strip() != env[
        "CUMETAL_COMMIT"
    ] or (output / "provider-status.stdout.txt").read_text().strip():
        report["status"] = (
            "provider checkout is modified or does not match the selected pin"
        )
        save()
        return 1
    result = run(
        "typed-compiler",
        [
            str(compiler),
            str(ptx),
            "--backend=cumetal-ir",
            "--fp64=fast48",
            "--ptx-strict",
            "--entry",
            KERNEL,
            "--emit=msl",
            "--no-link",
            "-o",
            str(output / "typed-output.metal"),
        ],
        60,
    )
    report["status"] = (
        "compiler timed out"
        if result["timed_out"]
        else "retained input accepted by compiler"
        if result["returncode"] == 0
        else "retained input rejected by compiler"
    )
    save()
    print(
        f"Retained PTX {actual_hash}: typed compiler exit={result['returncode']}; not QC acceptance"
    )
    return 0 if result["returncode"] == 0 and not result["timed_out"] else 1


if __name__ == "__main__":
    raise SystemExit(
        compare(Path(os.environ["CUMETAL_COMPARISON_OUTPUT"]), dict(os.environ))
    )
