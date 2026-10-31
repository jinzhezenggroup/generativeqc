"""Offline compiled artifact/resource census; never initializes CUDA."""

import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

sys.path[:0] = [".", "python"]
from tools.validate_direct_jk_aot_resources import cuobjdump_resources

root = Path.cwd()
build = root / "build/full-auto-cuda"
e = root / "evidence"


def item(p: Path) -> dict[str, Any]:
    """Bind an actual inspected file to its bytes."""
    return {
        "path": str(p.relative_to(root)),
        "bytes": p.stat().st_size,
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
    }


def group(paths: Iterable[Path]) -> dict[str, Any]:
    """Keep per-object records locally and aggregate package growth."""
    rows = [item(p) for p in sorted(paths)]
    return {
        "count": len(rows),
        "total_bytes": sum(r["bytes"] for r in rows),
        "max_bytes": max((r["bytes"] for r in rows), default=0),
        "rows": rows,
    }


r = {}
for label, path, marker in [
    ("point", e / "full-point-resources.txt", "automatic_"),
    ("force", e / "full-native-force-resources.txt", "generativeqc_native_libxc_"),
]:
    rows = cuobjdump_resources(
        "\n".join(
            m[0]
            for m in re.finditer(r"[ \t]*Function (\w+):\n[^\n]+", path.read_text())
            if marker in m[1]
        )
    )
    assert rows
    limits = {"REG": 255, "STACK": 4096, "LOCAL": 4096, "SHARED": 49152}
    r[label] = {
        "records": len(rows),
        "maxima": {k: max(v[k] for v in rows.values()) for k in limits},
        "violations": sum(
            any(v[k] > limit for k, limit in limits.items()) for v in rows.values()
        ),
        "raw": item(path),
    }
    assert r[label]["violations"] == 0
ninja = []
build_started = (e / "full-cache-before.txt").stat().st_mtime_ns
for line in (build / ".ninja_log").read_text().splitlines()[1:]:
    start, end, timestamp, output, _ = line.split("\t")
    # The tree is reused. Previous invocation timings have a different zero;
    # only outputs newer than this build's pre-cache receipt enter its span.
    if int(timestamp) >= build_started:
        ninja.append((int(start), int(end), output))
assert ninja
r["build_elapsed_seconds"] = (
    max(v[1] for v in ninja) - min(v[0] for v in ninja)
) / 1000
r["force_objects"] = group(
    build.glob("CMakeFiles/generativeqc_native_stationary.dir/**/native_libxc_*.cu.o")
)
r["point_objects"] = group(
    build.glob("CMakeFiles/generativeqc_libxc_semilocal_cuda.dir/**/*.o")
)
r["shared_objects"] = group(
    p
    for p in build.glob("CMakeFiles/generativeqc_native_stationary.dir/**/*.o")
    if "native_libxc_" not in p.name
)
r["archive"] = item(build / "libgenerativeqc_native_stationary.a")
commands = subprocess.run(
    [
        "/home/jzzeng/.local/bin/ninja",
        "-C",
        str(build),
        "-t",
        "commands",
        "generativeqc_native_stationary",
        "generativeqc_libxc_semilocal_cuda",
    ],
    capture_output=True,
    text=True,
    check=True,
    timeout=60,
).stdout
compiles = [line for line in commands.splitlines() if " -c " in line]
assert compiles and all("/home/jzzeng/.local/bin/sccache " in line for line in compiles)
r["compiler_launcher"] = {
    "commands": len(compiles),
    "verified_launcher_commands": len(compiles),
    "command_stream_sha256": hashlib.sha256(commands.encode()).hexdigest(),
}
r["installed"] = {
    name: item(root / path)
    for name, path in [
        ("library", "install/lib/libgenerativeqc.so"),
        ("cli", "install/bin/generativeqc"),
        ("build_library", "build/native-cuda/libgenerativeqc.so"),
    ]
}
(e / "full-object-census.json").write_text(json.dumps(r, indent=2) + "\n")
print(
    json.dumps(
        {
            k: (
                {n: v for n, v in val.items() if n != "rows"}
                if isinstance(val, dict)
                else val
            )
            for k, val in r.items()
        },
        indent=2,
    )
)
