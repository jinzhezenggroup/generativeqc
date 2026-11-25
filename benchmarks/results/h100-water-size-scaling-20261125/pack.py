"""Pack the text receipts losslessly after copying a stopped or terminal job."""

import argparse
import json
import lzma
import re
from pathlib import Path


def write(path, contents):
    encoded = json.dumps(contents, ensure_ascii=True, allow_nan=False).encode()
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(lzma.compress(encoded, preset=6))
    temporary.replace(path)


def raw(files):
    return {path.name: path.read_bytes().decode("utf-8") for path in sorted(files)}


def endpoint_files(root):
    return [
        p
        for p in (root / "gpu").iterdir()
        if p.is_file() and p.suffix in {".json", ".txt"}
    ]


def pack(root, output, arm):
    if arm == "cpu":
        files = root / "cpu"
        write(
            output / "cpu-raw.json.xz", raw(p for p in files.iterdir() if p.is_file())
        )
        return
    files = endpoint_files(root)
    run_log = root / "gpu-run.txt"
    launcher_logs = [run_log] if run_log.is_file() else []
    if arm in ("pilot", "legacy"):
        write(output / f"gpu-{arm}.json.xz", raw([*launcher_logs, *files]))
        if arm == "legacy":
            cpu_files = root / "cpu"
            write(
                output / "cpu-legacy.json.xz",
                raw(p for p in cpu_files.iterdir() if p.is_file()),
            )
        return
    groups = {}
    for path in files:
        match = re.fullmatch(
            r"(prewarm|formal)-\d+-(\d+)-[a-z0-9-]+\.(json|txt)", path.name
        )
        key = match[2] if match else "journal"
        groups.setdefault(key, []).append(path)
    groups["journal"].extend(launcher_logs)
    for key, paths in sorted(groups.items()):
        write(output / f"gpu-{arm}-{key}.json.xz", raw(paths))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=("pilot", "legacy", "main", "large", "cpu"))
    parser.add_argument("readback", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    pack(args.readback, args.output, args.arm)
