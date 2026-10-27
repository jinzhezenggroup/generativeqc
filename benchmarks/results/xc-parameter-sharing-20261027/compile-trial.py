"""Compare complete packages with a verified cache launcher and forced recaches.

Only this controlled compile-time experiment requests recompilation. Existing
cache contents are retained and refreshed; no input identities are falsified.
Code generation and compiler execution are timed separately on both revisions.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import shlex
import subprocess
import time
from pathlib import Path


def main() -> None:
    root = Path("/home/jzzeng/tmp/qc-xc-parameter-reuse-20261027")
    baseline = Path("/home/jzzeng/tmp/qc-df-schedule-reuse-20261027/baseline-snapshot")
    python = "/home/jzzeng/tmp/qc-1123-shared-aot-20261027/.venv/bin/python"
    launcher = str(Path(python).with_name("sccache"))
    version = subprocess.check_output([launcher, "--version"], text=True).strip()
    catalog = json.loads(
        (
            root / "source/python/generativeqc_compiler/xc/libxc_bulk_catalog.json"
        ).read_text()
    )
    names = [
        r["name"]
        for r in catalog["registrations"]
        if r["entry"] == "maple/mgga_exc/mgga_x_ktbm.mpl"
    ]
    commands = json.loads((root / "evidence/xc-build-baseline.json").read_text())[
        "commands"
    ]
    old_directory = str(
        root / "source/build/pro6000-cuda/generated/libxc_semilocal_cuda"
    )
    records = {}
    for arm, source in (("baseline", baseline), ("candidate", root / "source")):
        trial = root / "compile-trial" / arm
        generated = trial / "generated"
        objects = trial / "objects"
        objects.mkdir(parents=True, exist_ok=True)
        args = [
            python,
            str(source / "tools/generate_libxc_semilocal_cuda_registry.py"),
            "--output-directory",
            str(generated),
        ]
        if arm == "candidate":
            args.append("--share-parameters")
        for name in names:
            args.extend(("--functional", name))
        began = time.perf_counter()
        subprocess.run(args, check=True, timeout=120)
        generation_seconds = time.perf_counter() - began
        env = {**os.environ, "SCCACHE_RECACHE": "1"}
        before = subprocess.check_output([launcher, "--show-stats"], text=True)

        def compile_one(
            row: dict,
            *,
            generated: Path = generated,
            objects: Path = objects,
            env: dict = env,
        ) -> dict:
            command = shlex.split(row["command"])
            command = [x.replace(old_directory, str(generated)) for x in command]
            output = (
                objects
                / Path(row["file"]).with_suffix(Path(row["file"]).suffix + ".o").name
            )
            command[command.index("-o") + 1] = str(output)
            command = [launcher, *command]
            started = time.perf_counter()
            completed = subprocess.run(
                command,
                env=env,
                cwd=row["directory"],
                capture_output=True,
                text=True,
                timeout=240,
                check=False,
            )
            if completed.returncode:
                raise RuntimeError(completed.stderr)
            with output.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            return {
                "file": Path(row["file"]).name,
                "bytes": output.stat().st_size,
                "sha256": digest,
                "seconds": time.perf_counter() - started,
                "command": command,
            }

        began = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            outputs = list(pool.map(compile_one, commands))
        compile_seconds = time.perf_counter() - began
        after = subprocess.check_output([launcher, "--show-stats"], text=True)
        records[arm] = {
            "generation_seconds": generation_seconds,
            "compile_seconds": compile_seconds,
            "objects": outputs,
            "cache_before": before,
            "cache_after": after,
        }
        print(
            arm,
            generation_seconds,
            compile_seconds,
            sum(r["bytes"] for r in outputs),
            flush=True,
        )
    (root / "evidence/compile-trial.json").write_text(
        json.dumps(
            {
                "cache_launcher": launcher,
                "cache_version": version,
                "recache": True,
                "concurrent_jobs": 8,
                "records": records,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
