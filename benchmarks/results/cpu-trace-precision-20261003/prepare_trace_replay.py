"""Stage a NEW replay layout and provenance only; never run scientific endpoints."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--expanded-trace", type=Path, required=True)
p.add_argument("--expanded-cpu", type=Path, required=True)
p.add_argument("--source", type=Path, required=True)
p.add_argument("--library", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
if a.output.exists():
    raise ValueError("output must not exist")
source = a.source.resolve(strict=True)
library = a.library.resolve(strict=True)
if not library.is_relative_to(source):
    raise ValueError("library must come from the selected pinned source checkout/build")
if subprocess.check_output(
    ["git", "-C", str(source), "status", "--porcelain"], text=True
).strip():
    raise ValueError("source must be clean")


def identity(path: Path) -> dict[str, int | str]:
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def git(value: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", value], text=True
    ).strip()


def canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


repro = json.loads((a.expanded_trace / "reproduction.json").read_bytes())
# The published alias is independently verified to have the exact measured tree.
accepted_revisions = {
    repro["source_revision"],
    "e4cc2787c02959ea287eb81e4541d009fe860f8c",
}
if git("HEAD") not in accepted_revisions or git("HEAD^{tree}") != repro["source_tree"]:
    raise ValueError("choose the measured source or its exact-tree public alias")
root = a.output.resolve()
q = root / "revalidation/receipts-trace-replay/qualification/new-run"
q.mkdir(parents=True)
harnesses = sorted((a.expanded_trace / "harnesses").glob("supplemental*.py"))
if not harnesses:
    raise ValueError("decoded trace harnesses are required")
for f in harnesses:
    shutil.copyfile(f, q / f.name)
d = root / "revalidation/diagnostics"
d.mkdir()
f = a.expanded_cpu / "reproduction/diagnostics/pbe96_failure_history.py.txt"
expected = repro["dependencies"]["existing_source_snapshots"][
    "reproduction/diagnostics/pbe96_failure_history.py.txt"
]
if identity(f) != expected:
    raise ValueError("CPU diagnostic source dependency changed")
shutil.copyfile(f, d / "pbe96_failure_history.py")
o = root / "revalidation/oracles"
o.mkdir()
f = a.expanded_cpu / "oracles/pbe96-v2.json"
if (
    hashlib.sha256(canonical(json.loads(f.read_bytes()))).hexdigest()
    != "73c6f41ac263cef592824c8c65b8ec4141e5d736be8b6be65af9ce8acf813ee3"
):
    raise ValueError("CPU diagnostic oracle scientific values changed")
shutil.copyfile(f, o / f.name)
assert q.parents[2] / "oracles" == o
old = repro["dependencies"]["oracle"]
env = {
    "WORKTREE": str(source),
    "PYTHONPATH": str(source / "python") + ":" + str(source),
    "GENERATIVEQC_LIBRARY": str(library),
    "GENERATIVEQC_PROFILE": "off",
    **{
        k: "1"
        for k in [
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "BLIS_NUM_THREADS",
        ]
    },
}
manifest = {
    "schema": "trace-replay-provenance-v1",
    "status": "prepared-not-run",
    "source_revision": git("HEAD"),
    "source_tree": git("HEAD^{tree}"),
    "source_dirty": False,
    "native_library": str(library),
    "native_identity": identity(library),
    "environment": env,
    "harnesses": {f.name: identity(f) for f in sorted(q.glob("*.py"))},
    "oracle": {
        "path": str(o / f.name),
        "new_file_identity": identity(o / f.name),
        "original_identity": old,
        "canonical_sha256": hashlib.sha256(
            canonical(json.loads((o / f.name).read_bytes()))
        ).hexdigest(),
    },
    "note": "Prepared only. Record new Python/package versions, commands, native identity and completion outcomes for actual replay. Expanded oracle has original scientific values but may have different whitespace and raw file hash.",
}
(q / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
(root / "environment.sh").write_text(
    "\n".join(
        "export " + k + "=" + __import__("shlex").quote(v) for k, v in env.items()
    )
    + "\n"
)
print(
    json.dumps(
        {
            "prepared": str(q),
            "scientific_endpoints_executed": False,
            "manifest": str(q / "manifest.json"),
        }
    )
)
