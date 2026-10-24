"""Index and collect explicitly selected completed cache-state receipts.

The full source archive, installed binaries and opaque CUDA/compiler cache bodies
remain in the owned persistent root. Their inventories/hashes, all scientific
inputs/outputs, commands, diagnostics and unsuccessful preparation records are
selected here. No credentials or arbitrary working directory are copied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if not args.pilot_only and not (root / "receipts/measurement-finished").is_file():
        raise ValueError("full campaign is not independently replayed and finished")
    if args.output.exists():
        raise FileExistsError(args.output)
    selected = {root / "source-ececb48-manifest.json"}
    for name in ("receipts", "frozen-inputs"):
        selected.update(path for path in (root / name).iterdir() if path.is_file())
    for name in ("host-observer-probe", "cuda-observer-probe"):
        selected.update(
            path
            for path in (root / name).iterdir()
            if path.is_file()
            and path.suffix in (".json", ".py", ".cpp", ".cu", ".sha256")
        )
    for name in (
        "build-sdk.sh",
        "resume-sdk.sh",
        "freeze_environment.py",
        "prepare-inputs.py",
        "probe-download.py",
        "build-13561.log",
        "build-13562.log",
        "freeze-13569.txt",
        "gpu-13581.txt",
        "host-observer-probe.log",
        "cuda-observer-probe.txt",
        "source-archive.sha256",
    ):
        path = root / name
        if path.is_file():
            selected.add(path)
    selected.update(
        path
        for path in (root / "preparation-failure").iterdir()
        if path.is_file() and path.suffix in (".json", ".txt", ".sha256")
    )
    # Record the executing harness, not a later packaging checkout's spelling.
    selected.add(root / "harness/benchmarks/runtime_cache_state.py")
    selected.update(
        path
        for path in (
            root / "harness/benchmarks/results/runtime-cache-state-20261023"
        ).iterdir()
        if path.is_file() and path.suffix in (".py", ".sh")
    )
    for cohort in ("pilot",) if args.pilot_only else ("pilot", "formal"):
        folder = root / cohort
        selected.update(path for path in folder.iterdir() if path.is_file())
        for process in folder.glob("process-*"):
            selected.update(path for path in process.iterdir() if path.is_file())
    if not args.pilot_only:
        selected.add(root / "formal-independent-replay.json")
    args.output.mkdir(parents=True, exist_ok=False)
    files = []
    for path in sorted(selected):
        relative = path.relative_to(root)
        if (
            path.is_symlink()
            or ".env" in relative.parts
            or path.name.startswith(".env.")
        ):
            raise ValueError("unsafe raw selection")
        raw = path.read_bytes()
        target = args.output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        files.append(
            {
                "path": relative.as_posix(),
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "mode": f"{path.stat().st_mode & 0o777:04o}",
            }
        )
    index = {
        "schema": "generativeqc.runtime-cache-selected-raw.v1",
        "files": files,
        "source_files": 12353,
        "cohort_scope": "pilot only"
        if args.pilot_only
        else "pilot plus complete formal",
        "excluded": {
            "source.tar/source-ececb48.tar.gz": "Git LF source reconstructible from revision and complete indexed input manifest",
            "installed SDK and cache binary bodies": "immutable original binaries persist in owned root; complete identities/census retained",
            "qm7.mat": "complete consumed C2 geometry, basis snapshots and pinned QM7 identity/index retained; other geometries are unused",
            "preparation download prefix": "unsuccessful package transfer is not a scientific input; prefix identity and original diagnostics retained",
        },
    }
    index_path = args.output / "raw-index.json"
    index_path.touch(exist_ok=False)
    index["control_file_mode"] = f"{index_path.stat().st_mode & 0o777:04o}"
    index_path.write_text(json.dumps(index, indent=2) + "\n")
    print(
        json.dumps(
            {
                "files": len(files),
                "bytes": sum(row["bytes"] for row in files),
                "scope": index["cohort_scope"],
            }
        )
    )


if __name__ == "__main__":
    main()
