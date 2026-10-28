"""Retain byte-identical narrow audit inputs before expanding the active build."""

import hashlib
import json
import shutil
from pathlib import Path

root = Path.cwd()
build = root / "build/pro6000-cuda"
snapshot = root / "evidence/narrow-snapshot"
assert not snapshot.exists()
new = snapshot / "build/pro6000-cuda"
for relative in (
    "generated/stationary_force",
    "generated/libxc_semilocal_cuda",
    "CMakeFiles/generativeqc_native_stationary.dir",
    "CMakeFiles/generativeqc_libxc_semilocal_cuda.dir",
):
    shutil.copytree(build / relative, new / relative)
for name in ("generativeqc_dft_cuda_tests", "generativeqc_ks_cuda_tests"):
    shutil.copy2(build / name, new / name)
lib = snapshot / "lib"
lib.mkdir()
shutil.copy2(build / "libgenerativeqc.so", lib / "libgenerativeqc.so.0")
(lib / "libgenerativeqc.so").symlink_to("libgenerativeqc.so.0")
files = {
    str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in snapshot.rglob("*")
    if p.is_file() and not p.is_symlink()
}
(root / "evidence/narrow-snapshot-identity.json").write_text(
    json.dumps(dict(files=files), sort_keys=True) + "\n"
)
print("frozen narrow audit/execution inputs", len(files), flush=True)
